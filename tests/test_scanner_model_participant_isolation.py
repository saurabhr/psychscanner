"""Regression test: each simulated participant must get its own thread-id
components, not just the "task" one.

Found in review: ScannerModel.run()'s per-participant loop refreshed
trace_cfg["task"] every iteration but left trace_cfg["trial"] (and
["item"]) at the session-wide value set once before the loop. For
chain_type="trial" + Convo memory, TaskRunner builds thread_id from
trace_cfg["trial"] + trcode -- so every participant running the same
trcode shared one real conversation thread with every other participant.
"""
from __future__ import annotations

from unittest.mock import patch

from pathlib import Path

from psychscanner import ExpCard, ExpCardInit, ScannerModel
from psychscanner.task_runner import TaskRunner


def _trial_chain_task() -> dict:
    return {
        "tasktype": "sc",
        "taskname": "trial_chain_min",
        "instructions": {"definition": ["Answer briefly."]},
        "contexts": ["general"],
        "contexts_id": ["Q"],
        "context_present": False,
        "items": {
            "Q_1": [
                {"trcode": "Q_1", "stimulus": "first turn"},
                {"trcode": "Q_1", "stimulus": "second turn"},
            ]
        },
        "chain_type": "trial",
        "parser": None,
    }


def test_each_participant_gets_a_distinct_trial_thread_id(tmp_path: Path) -> None:
    seen_trial_values = []
    real_init = TaskRunner.__init__

    def spy_init(self, *args, trace_cfg=None, **kwargs):
        if trace_cfg is not None:
            seen_trial_values.append(trace_cfg["trial"])
        return real_init(self, *args, trace_cfg=trace_cfg, **kwargs)

    card = ExpCardInit(
        model="mock-chat-model",
        family="mock-llm",
        task_file=_trial_chain_task(),
        cogtype="no",
        nsim=2,
        memory="Convo",
        projectname="participant_isolation_test",
        proj_dir=tmp_path,
    )
    scanner = ScannerModel(expcard=ExpCard(card))

    with patch.object(TaskRunner, "__init__", spy_init):
        scanner.run(progress_bar=False)

    assert len(seen_trial_values) == 2, seen_trial_values
    assert seen_trial_values[0] != seen_trial_values[1], (
        "both simulated participants were assigned the same trace_cfg['trial'], "
        "which means they'd share one real conversation thread for chain_type='trial'"
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_each_participant_gets_a_distinct_trial_thread_id(Path(d))
    print("demo() OK")
