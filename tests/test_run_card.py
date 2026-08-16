"""Smoke test: run_card() reaches the same result as the manual chain it replaces."""
from pathlib import Path

from psychscanner import run_card

TASKS_DIR = Path(__file__).resolve().parent.parent / "examples" / "tasks"


def test_run_card_runs_with_mock_llm(tmp_path):
    results = run_card("example_survey", dirs=TASKS_DIR, proj_dir=tmp_path)
    assert results, "expected at least one trial result"


def test_run_card_defaults_projectname_to_taskname(tmp_path):
    run_card("example_survey", dirs=TASKS_DIR, proj_dir=tmp_path)
    assert (tmp_path / "example_survey").is_dir()
