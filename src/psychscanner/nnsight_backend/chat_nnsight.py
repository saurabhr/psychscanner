"""nnsight-backed chat model that captures internal activations per call.

Wraps an ``nnsight.LanguageModel`` (a thin interpretability layer over a
HuggingFace ``transformers`` model) behind the same ``BaseChatModel``
contract every other psychscanner backend uses, so it drops into
``model_provider.llm_chat_model`` and the rest of the scanning pipeline
unchanged. Every call additionally runs a cached forward pass over the
prompt and writes the captured module activations to disk, referencing the
file from the returned message instead of embedding tensors in it (mirrors
how ``media_store.py`` externalizes base64 blobs rather than inlining them).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

import click
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun

NNSIGHT_IMPORT_MSG = (
    "The nnsight backend requires the optional 'nnsight' and 'torch' "
    "packages. Install them with: pip install psychscanner[nnsight]"
)


def _resolve_module(root: Any, dotted_path: str) -> Any:
    """Resolve a dotted/indexed path like ``"model.layers.5.mlp"`` on *root*."""
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _cache_to_tensor_dict(cache: Any) -> dict[str, Any]:
    """Flatten an nnsight ``CacheDict`` into ``{module_path: output_tensor}``."""
    tensors = {}
    for module_path, entry in dict(cache).items():
        value = getattr(entry, "output", entry)
        tensors[module_path] = value.detach().cpu() if hasattr(value, "detach") else value
    return tensors


class ChatNNsightModel(BaseChatModel):
    """Chat model that generates via a local (or NDIF-remote) HF model through nnsight.

    Alongside the usual generated reply, every call runs an ``nnsight`` trace
    over the prompt and writes captured module activations to
    ``activations_dir/<uuid>.pt``; the path is attached to the returned
    ``AIMessage.additional_kwargs["activations_path"]``.

    Example:

        .. code-block:: python

            model = ChatNNsightModel(model="openai-community/gpt2")
            result = model.invoke([HumanMessage(content="hello")])
            print(result.additional_kwargs["activations_path"])
    """

    model_name: str = Field(alias="model")
    device_map: str = "auto"
    dtype: str | None = None
    dispatch: bool = True
    remote: bool = False

    max_new_tokens: int = 128
    temperature: float | None = None
    do_sample: bool = False

    # None captures every module's output (nnsight's default); pass dotted
    # paths (e.g. ["model.layers.5", "model.layers.10"]) to limit disk usage.
    capture_modules: list[str] | None = None
    activations_dir: str = "nnsight_activations"

    _nn_model: Any = PrivateAttr(default=None)

    def _get_model(self) -> Any:
        if self._nn_model is None:
            try:
                from nnsight import LanguageModel
            except ImportError as exc:
                raise ImportError(NNSIGHT_IMPORT_MSG) from exc

            kwargs: dict[str, Any] = {"device_map": self.device_map, "dispatch": self.dispatch}
            if self.dtype:
                import torch

                kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._nn_model = LanguageModel(self.model_name, **kwargs)
        return self._nn_model

    def _render_prompt(self, messages: list[BaseMessage]) -> str:
        nn_model = self._get_model()
        openai_messages = convert_to_openai_messages(messages)
        return nn_model.tokenizer.apply_chat_template(
            openai_messages, tokenize=False, add_generation_prompt=True
        )

    def _capture_activations(self, nn_model: Any, prompt: str) -> Path:
        # ponytail: activations are captured on the prompt's forward pass
        # only, not on each token generated afterward (that needs
        # tracer.all()/tracer.iter[:] inside .generate(), a fussier API).
        # Upgrade to per-generation-step capture if a use case needs it.
        modules = (
            [_resolve_module(nn_model, path) for path in self.capture_modules]
            if self.capture_modules
            else None
        )
        with nn_model.trace(prompt, remote=self.remote) as tracer:
            cache = (tracer.cache(modules=modules) if modules else tracer.cache()).save()

        import torch

        out_dir = Path(self.activations_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.pt"
        torch.save(
            {"model": self.model_name, "prompt": prompt, "activations": _cache_to_tensor_dict(cache)},
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
                "capture_modules": self.capture_modules,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "nnsight-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "remote": self.remote}


if __name__ == "__main__":
    # ponytail: smoke check only, requires `pip install psychscanner[nnsight]`
    # and network access to fetch a tiny model; not runnable in CI/sandbox.
    from langchain_core.messages import HumanMessage

    model = ChatNNsightModel(model="hf-internal-testing/tiny-random-gpt2", max_new_tokens=4)
    result = model.invoke([HumanMessage(content="hello!")])
    assert "activations_path" in result.additional_kwargs
    assert Path(result.additional_kwargs["activations_path"]).exists()
    click.echo(result)
