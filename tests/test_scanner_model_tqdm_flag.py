"""Regression test: progress_bar=True must actually show the tqdm bar.

ScannerModel.run(progress_bar=...) previously passed that flag straight
through to TaskRunner.execute(disable_tqdm=...) without negating it, so
requesting the bar hid it and vice versa.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from psychscanner import ExpCard, ExpCardInit, ScannerModel
from psychscanner.task_runner import TaskRunner


def _minimal_task() -> dict:
    return {
        "tasktype": "sc",
        "taskname": "tqdm_flag_min",
        "instructions": {"definition": ["Answer briefly."]},
        "contexts": ["general"],
        "contexts_id": ["Q"],
        "context_present": False,
        "items": {"Q_1": [{"trcode": "Q_1", "stimulus": "2 + 2?"}]},
        "chain_type": "item",
        "parser": None,
    }


def _run_and_capture_disable_tqdm(tmp_path: Path, progress_bar: bool) -> bool:
    seen = []
    real_execute = TaskRunner.execute

    def spy_execute(self, *args, disable_tqdm=True, **kwargs):
        seen.append(disable_tqdm)
        return real_execute(self, *args, disable_tqdm=disable_tqdm, **kwargs)

    card = ExpCardInit(
        model="mock-chat-model",
        family="mock-llm",
        task_file=_minimal_task(),
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
        projectname="tqdm_flag_test",
        proj_dir=tmp_path,
    )
    scanner = ScannerModel(expcard=ExpCard(card))
    with patch.object(TaskRunner, "execute", spy_execute):
        scanner.run(progress_bar=progress_bar)
    return seen[0]


def test_progress_bar_true_does_not_disable_tqdm(tmp_path: Path) -> None:
    assert _run_and_capture_disable_tqdm(tmp_path, progress_bar=True) is False


def test_progress_bar_false_disables_tqdm(tmp_path: Path) -> None:
    assert _run_and_capture_disable_tqdm(tmp_path / "b", progress_bar=False) is True


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        assert _run_and_capture_disable_tqdm(base / "a", True) is False
        assert _run_and_capture_disable_tqdm(base / "b", False) is True
    print("demo() OK")
