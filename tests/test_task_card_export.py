"""Unit tests for the .tcard.psyscan / .xcard.psyscan save/export API."""
from __future__ import annotations

import json

import pytest

from psychscanner import (
    ExpCardInit,
    TASK_CARD_EXT,
    EXPERIMENT_CARD_EXT,
    load_experiment_card,
    load_task_card,
    save_experiment_card,
    save_task_card,
)


def _minimal_task(taskname: str = "my_task") -> dict:
    return {
        "tasktype": "sc",
        "taskname": taskname,
        "instructions": {"definition": ["Answer briefly."]},
        "contexts": ["general"],
        "contexts_id": ["Q"],
        "context_present": False,
        "items": {"Q_1": [{"trcode": "Q_1", "stimulus": "2 + 2?"}]},
        "chain_type": "item",
        "parser": None,
    }


def test_save_task_card_round_trips(tmp_path):
    task = _minimal_task()
    path = save_task_card(task, tmp_path / f"my_task{TASK_CARD_EXT}")
    assert path.exists()
    assert load_task_card(path) == task


def test_save_task_card_requires_taskname_and_items():
    with pytest.raises(ValueError, match="taskname"):
        save_task_card({"items": {}}, "x.tcard.psyscan")
    with pytest.raises(ValueError, match="items"):
        save_task_card({"taskname": "x"}, "x.tcard.psyscan")


def test_save_task_card_warns_on_taskname_filename_mismatch(tmp_path):
    task = _minimal_task(taskname="foo")
    with pytest.warns(UserWarning, match="taskname.*filename stem"):
        save_task_card(task, tmp_path / f"bar{TASK_CARD_EXT}")


def test_save_task_card_warns_on_unexpected_extension(tmp_path):
    task = _minimal_task(taskname="my_task")
    with pytest.warns(UserWarning, match="recommended extension"):
        save_task_card(task, tmp_path / "my_task.txt")


def test_save_task_card_no_warning_when_consistent(tmp_path, recwarn):
    task = _minimal_task(taskname="my_task")
    save_task_card(task, tmp_path / f"my_task{TASK_CARD_EXT}")
    assert len(recwarn) == 0


def test_experiment_card_functions_are_aliases():
    from psychscanner.staging.scanner_cards import save_expcard, load_expcard

    assert save_experiment_card is save_expcard
    assert load_experiment_card is load_expcard


def test_save_experiment_card_round_trips(tmp_path):
    card = ExpCardInit(
        model="mock-llm",
        family="mock-llm",
        task_file=_minimal_task(),
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
    )
    path = tmp_path / f"my_experiment{EXPERIMENT_CARD_EXT}"
    save_experiment_card(card, path)
    assert path.exists()

    loaded = load_experiment_card(path, proj_dir=tmp_path / "results")
    assert loaded.model == "mock-llm"
    assert loaded.family == "mock-llm"


if __name__ == "__main__":
    # Lazy self-check, per this project's own convention -- run directly
    # (no pytest) to sanity-check the module imports and basic round trip.
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as d:
        p = save_task_card(_minimal_task(), _Path(d) / f"my_task{TASK_CARD_EXT}")
        assert load_task_card(p) == _minimal_task()
    print("demo() OK")
