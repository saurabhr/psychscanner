"""nnsight-backed chat model that captures VLM activations per call.

Replaces ``vlm_lens_backend`` (removed): unlike vlm-lens, ``nnsight``'s
``VisionLanguageModel`` (a ``LanguageModel`` subclass with an
``AutoProcessor`` for image+text input, supporting LLaVA, Qwen2-VL, and
other HuggingFace VLMs) is an ordinary importable package — no subprocess,
no per-model venv, no sqlite round-trip. It also generates real text
alongside the captured activations, which vlm-lens could not do.

Wraps it behind the same ``BaseChatModel`` contract every other psychscanner
backend uses, so it drops into ``model_provider.llm_chat_model`` and the
rest of the scanning pipeline unchanged. Activation capture mirrors
``nnsight_backend``'s ``_capture_activations``/``_resolve_module`` exactly
(reused from there, since ``VisionLanguageModel`` shares the same trace/cache
API as ``LanguageModel``); this module only adds image handling.

Example:

    .. code-block:: python

        model = ChatVLMModel(model="Qwen/Qwen2-VL-2B-Instruct")
        result = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe the color in this image in one word."},
                {"type": "image", "path": "/path/to/image.png"},
            ])
        ])
        print(result.content, result.additional_kwargs["activations_path"])
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import click
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun

VLM_IMPORT_MSG = (
    "The vlm backend requires the optional 'nnsight', 'torch' and 'pillow' "
    "packages. Install them with: pip install psychscanner[vlm]"
)


def _extract_text_and_images(messages: list[BaseMessage]) -> tuple[str, list]:
    """Flatten every message's content into (prompt_text, PIL images)."""
    from PIL import Image

    text_parts: list[str] = []
    images = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                if "path" in block:
                    image = Image.open(block["path"])
                else:
                    image = Image.open(io.BytesIO(base64.b64decode(block["base64"])))
                images.append(image.convert("RGB"))
    return " ".join(p for p in text_parts if p), images


class ChatVLMModel(BaseChatModel):
    """Chat model that generates via a local (or NDIF-remote) VLM through nnsight.

    Alongside the usual generated reply, every call runs an ``nnsight`` trace
    over the prompt and images, writing captured module activations to
    ``activations_dir/<uuid>.pt``; the path is attached to the returned
    ``AIMessage.additional_kwargs["activations_path"]``.
    """

    model_name: str = Field(alias="model")
    device_map: str = "auto"
    dtype: Optional[str] = None
    dispatch: bool = True
    remote: bool = False

    max_new_tokens: int = 128
    temperature: Optional[float] = None
    do_sample: bool = False

    # None captures every module's output (nnsight's default); pass dotted
    # paths (e.g. ["model.layers.5", "model.layers.10"]) to limit disk usage.
    capture_modules: Optional[List[str]] = None
    activations_dir: str = "vlm_activations"

    _nn_model: Any = PrivateAttr(default=None)

    def _get_model(self) -> Any:
        if self._nn_model is None:
            try:
                from nnsight import VisionLanguageModel
            except ImportError as exc:
                raise ImportError(VLM_IMPORT_MSG) from exc

            kwargs: dict[str, Any] = {"device_map": self.device_map, "dispatch": self.dispatch}
            if self.dtype:
                import torch

                kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._nn_model = VisionLanguageModel(self.model_name, **kwargs)
        return self._nn_model

    def _render_prompt(self, nn_model: Any, text: str, images: list) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        return nn_model.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    def _capture_activations(self, nn_model: Any, prompt: str, images: list) -> Path:
        from psychscanner.nnsight_backend.chat_nnsight import _cache_to_tensor_dict, _resolve_module

        modules = (
            [_resolve_module(nn_model, path) for path in self.capture_modules]
            if self.capture_modules
            else None
        )
        with nn_model.trace(prompt, images=images or None, remote=self.remote) as tracer:
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
        text, images = _extract_text_and_images(messages)
        prompt = self._render_prompt(nn_model, text, images)

        activations_path = self._capture_activations(nn_model, prompt, images)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature
        if stop:
            gen_kwargs["stop_strings"] = stop
            gen_kwargs["tokenizer"] = nn_model.tokenizer

        with nn_model.generate(prompt, images=images or None, remote=self.remote, **gen_kwargs) as tracer:
            out_ids = nn_model.generator.output.save()

        prompt_len = nn_model.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
        text_out = nn_model.tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

        message = AIMessage(
            content=text_out,
            additional_kwargs={
                "activations_path": str(activations_path),
                "capture_modules": self.capture_modules,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "vlm-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "remote": self.remote}


if __name__ == "__main__":
    # ponytail: smoke check only, requires `pip install psychscanner[vlm]`,
    # network access to fetch the model, and a local image; not runnable in
    # CI/sandbox.
    import sys

    from langchain_core.messages import HumanMessage

    if len(sys.argv) < 2:
        click.echo("usage: python chat_vlm.py <image_path> [model] [dtype]")
        sys.exit(1)

    model = ChatVLMModel(
        model=sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2-VL-2B-Instruct",
        dtype=sys.argv[3] if len(sys.argv) > 3 else None,
    )
    result = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the color in this image in one word."},
                    {"type": "image", "path": sys.argv[1]},
                ]
            )
        ]
    )
    assert "activations_path" in result.additional_kwargs
    assert Path(result.additional_kwargs["activations_path"]).exists()
    click.echo(result)
