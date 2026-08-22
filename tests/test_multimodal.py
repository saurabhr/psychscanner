"""Unit tests for multimodal stimulus support and tool-binding scaffolding.

Follows the smoke-test style of ``test_readme_quickstart.py`` — direct
construction, in-memory assertions, no live model calls.
"""
from __future__ import annotations

import base64
import hashlib
import json

from langchain_core.messages import AIMessage, HumanMessage

from psychscanner.datasets.prompts.multimodal import audio_block, file_block, image_block
from psychscanner.datasets.prompts.task_prompts import gen_stimulus_prompt
from psychscanner.feedback import FeedbackBase
from psychscanner.scanner_models.media_store import externalize_media
from psychscanner.scanner_models.psyscan_io import _stimulus_str
from psychscanner.task_runner import TaskRunner


# ── Block builders ────────────────────────────────────────────────────────────

def test_image_block_local_file(tmp_path):
    raw = b"\x89PNG\r\n"
    path = tmp_path / "img.png"
    path.write_bytes(raw)

    block = image_block(path)

    assert block["type"] == "image"
    assert block["mime_type"] == "image/png"
    assert base64.b64decode(block["base64"]) == raw


def test_image_block_url_passthrough():
    block = image_block("https://example.com/pic.jpg")

    assert block == {"type": "image", "url": "https://example.com/pic.jpg"}


def test_audio_block_mime_override(tmp_path):
    path = tmp_path / "clip.bin"
    path.write_bytes(b"RIFF....")

    block = audio_block(path, mime_type="audio/wav")

    assert block["type"] == "audio"
    assert block["mime_type"] == "audio/wav"


def test_file_block_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")

    block = file_block(path)

    assert block["type"] == "file"
    assert block["mime_type"] == "application/pdf"


# ── gen_stimulus_prompt ────────────────────────────────────────────────────────

def test_gen_stimulus_prompt_multimodal_context_absent():
    blocks = [{"type": "image", "base64": "aaaa", "mime_type": "image/png"}, {"type": "text", "text": "q"}]
    trstim = {"stimulus": blocks, "trcode": "feat_1", "context_present": False}

    msg = gen_stimulus_prompt(trstim)

    assert isinstance(msg, HumanMessage)
    assert msg.content == blocks


def test_gen_stimulus_prompt_resolves_json_authored_path_block(tmp_path, monkeypatch):
    """A hand-written JSON task card can use {"path": ...} instead of image_block().

    Relative to cwd, not absolute -- task-card JSON is untrusted input, so
    resolve_path_block() rejects absolute/".."-traversal paths (see
    test_multimodal_path_containment.py). This exercises the still-supported
    relative-path form.
    """
    monkeypatch.chdir(tmp_path)
    raw = b"\x89PNG\r\n"
    (tmp_path / "img.png").write_bytes(raw)
    trstim = {
        "stimulus": [{"type": "image", "path": "img.png"}, {"type": "text", "text": "q"}],
        "trcode": "feat_1",
        "context_present": False,
    }

    msg = gen_stimulus_prompt(trstim)

    assert msg.content[0]["type"] == "image"
    assert "path" not in msg.content[0]
    assert base64.b64decode(msg.content[0]["base64"]) == raw
    assert msg.content[0]["mime_type"] == "image/png"
    assert msg.content[1] == {"type": "text", "text": "q"}


def test_json_authored_path_block_rejects_absolute_path(tmp_path):
    """Regression (code review finding): a task card is untrusted input --
    an absolute path in a "path" block used to be read straight off disk
    and shipped to the model provider (e.g. {"path": "/etc/passwd"})."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak me")
    trstim = {
        "stimulus": [{"type": "file", "path": str(secret)}],
        "trcode": "feat_1",
        "context_present": False,
    }

    try:
        gen_stimulus_prompt(trstim)
        assert False, "expected ValueError for an absolute path"
    except ValueError as exc:
        assert "relative path" in str(exc)


def test_json_authored_path_block_rejects_parent_traversal():
    """Regression (code review finding): "../" escapes were also unguarded."""
    trstim = {
        "stimulus": [{"type": "file", "path": "../../etc/passwd"}],
        "trcode": "feat_1",
        "context_present": False,
    }

    try:
        gen_stimulus_prompt(trstim)
        assert False, "expected ValueError for a '..' path"
    except ValueError as exc:
        assert "relative path" in str(exc)


def test_gen_stimulus_prompt_multimodal_context_present():
    """Previously dead code: context_present=True used to json.dumps-flatten list stimuli."""
    blocks = [{"type": "image", "base64": "aaaa", "mime_type": "image/png"}]
    trstim = {
        "stimulus": blocks,
        "trcode": "conj_1",
        "context_present": True,
        "context_item": "Conjunction search block",
    }

    msg = gen_stimulus_prompt(trstim)

    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, list)
    assert msg.content[0] == {"type": "text", "text": "TRIAL_CONTEXT: Conjunction search block"}
    assert msg.content[1:] == blocks


def test_gen_stimulus_prompt_text_context_present_unchanged():
    """Regression: plain-text stimulus with context still json.dumps-wraps as before."""
    trstim = {
        "stimulus": "APPLE",
        "trcode": "test_1",
        "context_present": True,
        "context_item": "Encoding phase",
    }

    msg = gen_stimulus_prompt(trstim)

    assert isinstance(msg, HumanMessage)
    parsed = json.loads(msg.content)
    assert parsed == {"TRIAL_CONTEXT": "Encoding phase", "TRIAL": "APPLE"}


# ── FeedbackBase.inject_feedback ───────────────────────────────────────────────

class _DummyFeedback(FeedbackBase):
    def on_response(self, trial, response):
        return None


def test_inject_feedback_preserves_multimodal_blocks():
    blocks = [{"type": "image", "base64": "aaaa", "mime_type": "image/png"}]
    input_dict = {"inputs": [HumanMessage(content=blocks)]}

    result = _DummyFeedback().inject_feedback(input_dict, "Good job")

    content = result["inputs"][0].content
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Good job"}
    assert content[1:] == blocks


def test_inject_feedback_text_stimulus_unchanged():
    input_dict = {"inputs": [HumanMessage(content="plain stimulus")]}

    result = _DummyFeedback().inject_feedback(input_dict, "fb text")

    parsed = json.loads(result["inputs"][0].content)
    assert parsed["current_trial"] == {"stimulus": "plain stimulus"}


# ── _stimulus_str (CSV export) ──────────────────────────────────────────────────

def test_stimulus_str_omits_base64_keeps_type_and_mime():
    blocks = [
        {"type": "image", "base64": "aaaa" * 100, "mime_type": "image/png"},
        {"type": "text", "text": "q"},
    ]

    result = _stimulus_str(blocks)

    assert "aaaa" not in result
    assert json.loads(result) == [{"type": "image", "mime_type": "image/png"}, {"type": "text"}]


def test_stimulus_str_handles_mixed_string_and_block_content():
    """Content lists may legitimately mix plain strings with block dicts."""
    stimulus = ["prototype instructions", {"type": "image", "base64": "aaaa", "mime_type": "image/png"}]

    result = _stimulus_str(stimulus)

    assert json.loads(result) == ["prototype instructions", {"type": "image", "mime_type": "image/png"}]


# ── externalize_media ────────────────────────────────────────────────────────

def test_externalize_media_dedups_across_trials(tmp_path):
    raw = b"fake-image-bytes"
    b64 = base64.b64encode(raw).decode()
    trials = [
        {"stimulus": [{"type": "image", "base64": b64, "mime_type": "image/png"}]},
        {"stimulus": [{"type": "image", "base64": b64, "mime_type": "image/png"}]},
    ]
    media_dir = tmp_path / "media"

    result = externalize_media(trials, media_dir)

    digest = hashlib.sha256(raw).hexdigest()
    expected_path = media_dir / f"{digest}.png"
    assert expected_path.read_bytes() == raw
    assert len(list(media_dir.iterdir())) == 1  # written once despite two trials

    for trial in result:
        block = trial["stimulus"][0]
        assert "base64" not in block
        assert block["path"] == str(expected_path)

    # input is not mutated
    assert "base64" in trials[0]["stimulus"][0]


def test_externalize_media_rewrites_message_content(tmp_path):
    raw = b"audio-bytes"
    b64 = base64.b64encode(raw).decode()
    msg = HumanMessage(content=[{"type": "audio", "base64": b64, "mime_type": "audio/wav"}])
    trials = [{"inputs": [msg]}]

    result = externalize_media(trials, tmp_path / "media")

    rewritten = result[0]["inputs"][0]
    assert isinstance(rewritten, HumanMessage)
    assert "base64" not in rewritten.content[0]
    assert "path" in rewritten.content[0]
    assert "base64" in msg.content[0]  # original message unaffected


# ── TaskRunner thread-id assignment (trial-chain worked example) ───────────────

def test_taskrunner_trial_chain_shares_thread_id_per_trcode():
    captured_configs = []

    class _FakeAIApp:
        def invoke(self, input_dict, config=None):
            captured_configs.append(config)
            return {**input_dict, "inputs": [*input_dict["inputs"], AIMessage(content="ok")]}

    class _FakeAgent:
        parser = None
        ai_app = _FakeAIApp()

    tasktrials = {
        "trials": [
            {"trcode": "conj_1", "stimulus": "a", "tasktype": "x", "parser": None, "fb": False},
            {"trcode": "conj_1", "stimulus": "b", "tasktype": "x", "parser": None, "fb": False},
            {"trcode": "conj_2", "stimulus": "c", "tasktype": "x", "parser": None, "fb": False},
        ]
    }
    runner = TaskRunner(
        scanning_agent=_FakeAgent(),
        trace_cfg={"trial": "sess-", "task": "sess-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="trial",
        hmsg="stimulus",
    )

    runner.execute()

    thread_ids = [c["configurable"]["thread_id"] for c in captured_configs]
    assert thread_ids[0] == thread_ids[1] == "sess-conj_1"
    assert thread_ids[2] == "sess-conj_2"
    assert thread_ids[0] != thread_ids[2]
