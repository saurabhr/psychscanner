"""End-to-end test of the README Quick Start example.

The README documents this code path; this test pins it. By default the test runs
against a local Ollama smol model (`smollm2:360m-instruct-fp16`) so it has no
API-key dependency and works fully offline once the model is pulled. The Groq
configuration used during initial development is kept as a commented alternative
for reference.
"""
from __future__ import annotations

import os  # noqa: F401  (kept available for the commented Groq alternative)
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from dotenv import load_dotenv

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

load_dotenv()

# ── Backend selection ────────────────────────────────────────────────────────
# Default: local Ollama smol model. Pull it once with:
#     ollama pull smollm2:360m-instruct-fp16
MODEL_NAME = "smollm2:360m-instruct-fp16"
MODEL_FAMILY = "ollama"

# Alternative: Groq-hosted gpt-oss-120b (requires GROQ_API_KEY in .env)
# MODEL_NAME = "openai/gpt-oss-120b"
# MODEL_FAMILY = "groq"

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_reachable(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {OLLAMA_URL}; live README Quick Start test skipped.",
)

# Alternative gate for the Groq variant:
# requires_groq = pytest.mark.skipif(
#     not os.getenv("GROQ_API_KEY"),
#     reason="GROQ_API_KEY not set; live README Quick Start test skipped.",
# )


def test_readme_quickstart_smoke(tmp_path: Path) -> None:
    """The README Quick Start setup validates without making any API calls.

    ExpCard must validate, the bundled VVIQ-16 task must load, the parser must
    resolve, and the scanner must initialise without error.
    """
    card = ExpCardInit(
        model=MODEL_NAME,
        family=MODEL_FAMILY,
        projectname="readme_quickstart_test",
        proj_dir=tmp_path,
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
        parser=DefaultLiteralVivid15,
    )
    exp = ExpCard(card)
    scanner = ScannerModel(expcard=exp)

    assert exp.parser is DefaultLiteralVivid15
    assert exp.task_data["taskname"] == "vviq16"
    assert len(scanner.scanner_data["system_prompts"]) == 1
    assert len(scanner.scanner_data["task_prompts"]["trials"]) == 16


@requires_ollama
def test_readme_quickstart_live_ollama(tmp_path: Path) -> None:
    """Run the README Quick Start through a local Ollama smol model and validate the CSV export."""
    card = ExpCardInit(
        model=MODEL_NAME,
        family=MODEL_FAMILY,
        parameters={"temperature": 0},
        projectname="readme_quickstart_test",
        proj_dir=tmp_path,
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
        parser=DefaultLiteralVivid15,
    )
    scanner = ScannerModel(expcard=ExpCard(card))
    results = scanner.run()

    assert len(results) == 1
    assert len(results[0]) == 16  # VVIQ-16 has 16 items

    df = to_csv(scanner, path=tmp_path)

    assert len(df) == 16
    assert "resp_Vividness" in df.columns
    assert df["taskname"].unique().to_list() == ["vviq16"]
    assert df["model"].unique().to_list() == [MODEL_NAME]
    assert df["family"].unique().to_list() == [MODEL_FAMILY]
    # Every trial got a parsed integer rating in 1..5
    ratings = [r for r in df["resp_Vividness"].to_list() if r is not None]
    assert ratings, "no Vividness ratings parsed"
    assert all(1 <= int(r) <= 5 for r in ratings)


# ── Groq alternative (commented; uncomment and use @requires_groq to enable) ──
#
# @requires_groq
# def test_readme_quickstart_live_groq(tmp_path: Path) -> None:
#     """Run the README Quick Start through Groq and validate the CSV export."""
#     card = ExpCardInit(
#         model="openai/gpt-oss-120b",
#         family="groq",
#         projectname="readme_quickstart_test",
#         proj_dir=tmp_path,
#         cogtype="no",
#         nsim=1,
#         memory="SingleTurn",
#         parser=DefaultLiteralVivid15,
#     )
#     scanner = ScannerModel(expcard=ExpCard(card))
#     results = scanner.run()
#     assert len(results) == 1
#     assert len(results[0]) == 16
#     df = to_csv(scanner, path=tmp_path)
#     assert df["model"].unique().to_list() == ["openai/gpt-oss-120b"]
#     assert df["family"].unique().to_list() == ["groq"]
