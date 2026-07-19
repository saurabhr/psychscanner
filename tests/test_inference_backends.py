"""Unit tests for the nnsight/nnterp/vlm chat-model backends.

Follows the smoke-test style of ``test_multimodal.py`` — direct construction,
in-memory assertions, no live model calls. ``torch``/``nnsight``/``nnterp``
are optional deps not installed in this environment, so these tests cover
the pure-Python logic (path resolution, cache flattening, message parsing,
routing) plus the real ImportError contract when the optional deps are
absent, rather than exercising an actual forward pass.
"""
from __future__ import annotations

import base64
import re
import sys
import types
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from psychscanner.memories.base.model_provider import llm_chat_model
from psychscanner.nnsight_backend.chat_nnsight import (
    NNSIGHT_IMPORT_MSG,
    ChatNNsightModel,
    _cache_to_tensor_dict,
    _resolve_module,
)
from psychscanner.nnterp_backend.chat_nnterp import NNTERP_IMPORT_MSG, ChatNNterpModel
from psychscanner.vlm_backend.chat_vlm import VLM_IMPORT_MSG, ChatVLMModel, _extract_text_and_images


# ── _resolve_module (shared by nnsight_backend and vlm_backend) ────────────────

def test_resolve_module_dotted_and_indexed_path():
    root = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp="layer0-mlp"), SimpleNamespace(mlp="layer1-mlp")]
        )
    )

    assert _resolve_module(root, "model.layers.1.mlp") == "layer1-mlp"


def test_resolve_module_plain_attribute_path():
    root = SimpleNamespace(embed_tokens="embedding")

    assert _resolve_module(root, "embed_tokens") == "embedding"


# ── _cache_to_tensor_dict ────────────────────────────────────────────────────

class _FakeTensor:
    def __init__(self, tag):
        self.tag = tag

    def detach(self):
        return self

    def cpu(self):
        return f"cpu:{self.tag}"


def test_cache_to_tensor_dict_reads_output_attr_and_detaches():
    cache = {
        "layers.0": SimpleNamespace(output=_FakeTensor("l0")),
        "layers.1": SimpleNamespace(output=_FakeTensor("l1")),
    }

    result = _cache_to_tensor_dict(cache)

    assert result == {"layers.0": "cpu:l0", "layers.1": "cpu:l1"}


def test_cache_to_tensor_dict_passes_through_raw_non_tensor_values():
    # Entries without an `.output` attr fall back to the raw value; values
    # without `.detach()` are kept as-is instead of erroring.
    cache = {"layers.0": "already-plain-value"}

    result = _cache_to_tensor_dict(cache)

    assert result == {"layers.0": "already-plain-value"}


# ── Optional-dependency guard (real: torch/nnsight/nnterp are not installed) ──

def test_nnsight_model_raises_friendly_import_error_without_optional_deps():
    model = ChatNNsightModel(model="hf-internal-testing/tiny-random-gpt2")

    with pytest.raises(ImportError, match=re.escape(NNSIGHT_IMPORT_MSG)):
        model._get_model()


def test_nnterp_model_raises_friendly_import_error_without_optional_deps():
    model = ChatNNterpModel(model="hf-internal-testing/tiny-random-gpt2")

    with pytest.raises(ImportError, match=re.escape(NNTERP_IMPORT_MSG)):
        model._get_model()


def test_vlm_model_raises_friendly_import_error_without_optional_deps():
    model = ChatVLMModel(model="Qwen/Qwen2-VL-2B-Instruct")

    with pytest.raises(ImportError, match=re.escape(VLM_IMPORT_MSG)):
        model._get_model()


# ── model_provider routing ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("family", "expected_cls"),
    [
        ("nnsight", ChatNNsightModel),
        ("NNSight", ChatNNsightModel),  # family match is case-insensitive
        ("nnterp", ChatNNterpModel),
        ("vlm", ChatVLMModel),
    ],
)
def test_llm_chat_model_routes_new_backends(family, expected_cls):
    chat_model = llm_chat_model("hf-internal-testing/tiny-random-gpt2", family)

    assert isinstance(chat_model, expected_cls)
    assert chat_model.model_name == "hf-internal-testing/tiny-random-gpt2"


def test_llm_chat_model_forwards_extra_parameters_to_backend():
    chat_model = llm_chat_model(
        "hf-internal-testing/tiny-random-gpt2",
        "nnsight",
        parameters={"max_new_tokens": 7, "capture_modules": ["model.layers.0"]},
    )

    assert chat_model.max_new_tokens == 7
    assert chat_model.capture_modules == ["model.layers.0"]


# ── pydantic mutable-default isolation (nnterp's capture_kinds) ────────────────

def test_nnterp_capture_kinds_default_not_shared_across_instances():
    first = ChatNNterpModel(model="m1")
    second = ChatNNterpModel(model="m2")

    first.capture_kinds.append("attentions")

    assert first.capture_kinds == ["layers", "attentions"]
    assert second.capture_kinds == ["layers"]  # unaffected by mutation on `first`


# ── _extract_text_and_images (vlm_backend) ──────────────────────────────────
# PIL is an optional dep not installed in this environment either; stub it
# so the message-parsing logic (independent of any real image decoding) is
# still exercised.

@pytest.fixture
def fake_pil(monkeypatch):
    opened = []

    class _FakeImage:
        def __init__(self, source):
            self.source = source
            self.mode = "RGBA"

        def convert(self, mode):
            self.mode = mode
            return self

    class _FakeImageModule:
        @staticmethod
        def open(source):
            opened.append(source)
            return _FakeImage(source)

    fake_pil_pkg = types.ModuleType("PIL")
    fake_pil_pkg.Image = _FakeImageModule
    monkeypatch.setitem(sys.modules, "PIL", fake_pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_pkg.Image)
    return opened


def test_extract_text_and_images_collects_text_and_path_image(fake_pil, tmp_path):
    img_path = tmp_path / "pic.png"
    img_path.write_bytes(b"\x89PNG")
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this."},
                {"type": "image", "path": str(img_path)},
            ]
        )
    ]

    text, images = _extract_text_and_images(messages)

    assert text == "Describe this."
    assert len(images) == 1
    assert images[0].source == str(img_path)
    assert images[0].mode == "RGB"  # converted


def test_extract_text_and_images_decodes_base64_image_and_preserves_order(fake_pil):
    raw = b"fake-bytes"
    b64 = base64.b64encode(raw).decode()
    messages = [
        HumanMessage(
            content=[
                {"type": "image", "base64": b64},
                {"type": "text", "text": "first"},
                "plain string block",
                {"type": "text", "text": "second"},
            ]
        )
    ]

    text, images = _extract_text_and_images(messages)

    assert text == "first plain string block second"  # joined in list order
    assert len(images) == 1


def test_extract_text_and_images_no_images_returns_empty_list(fake_pil):
    text, images = _extract_text_and_images([HumanMessage(content="just text")])

    assert text == "just text"
    assert images == []
