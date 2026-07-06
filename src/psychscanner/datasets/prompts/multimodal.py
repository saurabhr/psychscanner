"""Standard LangChain content-block builders for multimodal trial stimuli.

Task JSON can't embed binary data directly, so these helpers turn a local
file path or URL into the standard content-block dict that ``gen_stimulus_prompt``
passes straight through as ``HumanMessage`` content
(https://docs.langchain.com/oss/python/langchain/messages). Use a list of
these blocks as a trial's ``"stimulus"`` value.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def _is_url(path_or_url: str | Path) -> bool:
    return isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://"))


def _block(block_type: str, path_or_url: str | Path, mime_type: str | None) -> dict:
    if _is_url(path_or_url):
        block: dict = {"type": block_type, "url": path_or_url}
        if mime_type:
            block["mime_type"] = mime_type
        return block

    path = Path(path_or_url)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = mime_type or mimetypes.guess_type(path.name)[0]
    block = {"type": block_type, "base64": data}
    if mime_type:
        block["mime_type"] = mime_type
    return block


def image_block(path_or_url: str | Path, *, mime_type: str | None = None) -> dict:
    """Build a standard image content block from a local path or URL."""
    return _block("image", path_or_url, mime_type)


def audio_block(path_or_url: str | Path, *, mime_type: str | None = None) -> dict:
    """Build a standard audio content block from a local path or URL."""
    return _block("audio", path_or_url, mime_type)


def file_block(path_or_url: str | Path, *, mime_type: str | None = None) -> dict:
    """Build a standard file content block (e.g. a PDF) from a local path or URL."""
    return _block("file", path_or_url, mime_type)


def resolve_path_block(block: dict) -> dict:
    """Resolve a JSON-authored ``{"type": ..., "path": "local/file.png"}`` block to base64.

    Lets a hand-written JSON task card reference local media by plain file
    path, without a Python call to :func:`image_block`/:func:`audio_block`/
    :func:`file_block`. Blocks without a ``"path"`` key (already ``base64``
    or ``url``) pass through unchanged.
    """
    if "path" not in block:
        return block
    path = Path(block["path"])
    mime_type = block.get("mime_type") or mimetypes.guess_type(path.name)[0]
    resolved = {k: v for k, v in block.items() if k != "path"}
    resolved["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    if mime_type:
        resolved["mime_type"] = mime_type
    return resolved


def website_block(url: str) -> dict:
    """Fetch a webpage and return its visible text as a text-plain document block.

    Requires the ``multimodal`` optional dependency group
    (``pip install psychscanner[multimodal]``).
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "website_block requires the 'multimodal' extra: "
            "pip install psychscanner[multimodal]"
        ) from exc

    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    text = BeautifulSoup(response.text, "html.parser").get_text(separator="\n", strip=True)
    return {"type": "text-plain", "text": text, "mime_type": "text/plain"}
