"""Externalize inline base64 media from trial records before they're persisted.

Every ``.psyscan`` checkpoint re-serializes the full trial list
(:func:`~psychscanner.scanner_models.scanner_model.ScannerModel.model_dump`),
so an inline base64 image/audio stimulus would otherwise be duplicated into
every checkpoint written across every persona/``nsim`` condition that reuses
it. This module writes each unique blob once, content-addressed by hash, and
replaces it with a path reference in the persisted copy only — the messages
already sent to the LLM earlier in the run are unaffected.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "application/pdf": ".pdf",
}


def _guess_ext(mime_type: str | None) -> str:
    if not mime_type:
        return ""
    return _EXT_BY_MIME.get(mime_type) or (mimetypes.guess_extension(mime_type) or "")


def _write_block(block: dict, media_dir: Path) -> dict:
    raw = base64.b64decode(block["base64"])
    digest = hashlib.sha256(raw).hexdigest()
    media_path = media_dir / f"{digest}{_guess_ext(block.get('mime_type'))}"
    if not media_path.exists():
        media_dir.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(raw)
    return {**{k: v for k, v in block.items() if k != "base64"}, "path": str(media_path)}


def _rewrite(value: Any, media_dir: Path) -> Any:
    if isinstance(value, BaseMessage):
        value.content = _rewrite(value.content, media_dir)
        return value
    if isinstance(value, list):
        return [_rewrite(v, media_dir) for v in value]
    if isinstance(value, dict):
        if "base64" in value:
            return _write_block(value, media_dir)
        return {k: _rewrite(v, media_dir) for k, v in value.items()}
    return value


def externalize_media(trials: list[dict], media_dir: Path) -> list[dict]:
    """Return a deep copy of *trials* with inline base64 media externalized.

    Walks every field of every trial dict (``stimulus``, ``inputs``/``hmsg``
    message content, ``pred_dict``, ...); any block dict carrying a
    ``"base64"`` key is written to ``media_dir/<sha256><ext>`` (skipped if
    that file already exists — natural dedup across trials/personas that
    reuse the same stimulus) and rewritten with a ``"path"`` key instead.
    Does not mutate *trials*.
    """
    rewritten = copy.deepcopy(trials)
    return [
        {key: _rewrite(value, media_dir) for key, value in trial.items()}
        for trial in rewritten
    ]
