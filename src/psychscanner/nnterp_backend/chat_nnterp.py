"""nnterp-backed chat model that captures internal activations per call.

Wraps ``nnterp.StandardizedTransformer`` (an nnsight ``LanguageModel``
subclass that renames every HuggingFace architecture to a common
``layers`` / ``self_attn`` / ``mlp`` shape) behind the same ``BaseChatModel``
contract every other psychscanner backend uses, so it drops into
``model_provider.llm_chat_model`` and the rest of the scanning pipeline
unchanged. Unlike the ``nnsight_backend`` counterpart, activations are
addressed by layer index + kind (``layers``/``attentions``/``mlps``)
instead of hand-written dotted module paths, so the same call works across
architectures without per-model tuning.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import click
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun

NNTERP_IMPORT_MSG = (
    "The nnterp backend requires the optional 'nnterp' and 'torch' "
    "packages. Install them with: pip install psychscanner[nnterp]"
)


class ChatNNterpModel(BaseChatModel):
    """Chat model that generates via a local (or NDIF-remote) HF model through nnterp.

    Alongside the usual generated reply, every call runs an ``nnterp`` trace
    over the prompt and writes captured layer activations to
    ``activations_dir/<uuid>.pt``; the path is attached to the returned
    ``AIMessage.additional_kwargs["activations_path"]``.

    Example:

        .. code-block:: python

            model = ChatNNterpModel(model="openai-community/gpt2")
            result = model.invoke([HumanMessage(content="hello")])
            print(result.additional_kwargs["activations_path"])
    """

    model_name: str = Field(alias="model")
    device_map: str = "auto"
    dtype: Optional[str] = None
    trust_remote_code: bool = False
    remote: bool = False

    max_new_tokens: int = 128
    temperature: Optional[float] = None
    do_sample: bool = False

    # Which standardized accessor(s) to read: any of "layers", "attentions", "mlps".
    capture_kinds: List[str] = ["layers"]
    # None captures every layer; pass indices (e.g. [5, 10]) to limit disk usage.
    capture_layers: Optional[List[int]] = None
    activations_dir: str = "nnterp_activations"

    _nn_model: Any = PrivateAttr(default=None)

    def _get_model(self) -> Any:
        if self._nn_model is None:
            try:
                from nnterp import StandardizedTransformer
            except ImportError as exc:
                raise ImportError(NNTERP_IMPORT_MSG) from exc

            kwargs: dict[str, Any] = {
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
            }
            if self.dtype:
                import torch

                kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._nn_model = StandardizedTransformer(self.model_name, **kwargs)
        return self._nn_model

    def _render_prompt(self, messages: list[BaseMessage]) -> str:
        nn_model = self._get_model()
        openai_messages = convert_to_openai_messages(messages)
        return nn_model.tokenizer.apply_chat_template(
            openai_messages, tokenize=False, add_generation_prompt=True
        )

    def _capture_activations(self, nn_model: Any, prompt: str) -> Path:
        # ponytail: activations are captured on the prompt's forward pass
        # only, matching the nnsight_backend behavior. Same upgrade path
        # applies if per-generation-step capture is ever needed.
        layers = (
            self.capture_layers if self.capture_layers is not None else range(nn_model.num_layers)
        )
        with nn_model.trace(prompt, remote=self.remote) as tracer:
            saved = {
                f"{kind}.{i}": getattr(nn_model, f"{kind}_output")[i].save()
                for kind in self.capture_kinds
                for i in layers
            }

        import torch

        tensors = {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in saved.items()
        }
        out_dir = Path(self.activations_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.pt"
        torch.save(
            {"model": self.model_name, "prompt": prompt, "activations": tensors},
            out_path,
        )
        return out_path

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: "CallbackManagerForLLMRun | None" = None,
        **kwargs: Any,
    ) -> ChatResult:
        nn_model = self._get_model()
        prompt = self._render_prompt(messages)

        activations_path = self._capture_activations(nn_model, prompt)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature
        if stop:
            gen_kwargs["stop_strings"] = stop
            gen_kwargs["tokenizer"] = nn_model.tokenizer

        with nn_model.generate(prompt, remote=self.remote, **gen_kwargs) as tracer:
            out_ids = nn_model.generator.output.save()

        prompt_len = nn_model.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
        text = nn_model.tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

        message = AIMessage(
            content=text,
            additional_kwargs={
                "activations_path": str(activations_path),
                "capture_kinds": self.capture_kinds,
                "capture_layers": self.capture_layers,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "nnterp-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "remote": self.remote}


if __name__ == "__main__":
    # ponytail: smoke check only, requires `pip install psychscanner[nnterp]`
    # and network access to fetch a tiny model; not runnable in CI/sandbox.
    from langchain_core.messages import HumanMessage

    model = ChatNNterpModel(model="hf-internal-testing/tiny-random-gpt2", max_new_tokens=4)
    result = model.invoke([HumanMessage(content="hello!")])
    assert "activations_path" in result.additional_kwargs
    assert Path(result.additional_kwargs["activations_path"]).exists()
    click.echo(result)
