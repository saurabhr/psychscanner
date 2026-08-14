"""Regression test: tunnel_systemtrials() must not crash on an empty tunnel log.

Previously the IndexError guard set last_scan_on_off_events = None but then
unconditionally did last_scan_on_off_events["run_type"], raising TypeError.
Not reachable through the normal run() path today (create_tunnel() always
writes a CRITICAL BEGIN record first), but tunnel_systemtrials() is public
API and should be safe to call directly against a genuinely empty log.
"""
from __future__ import annotations

from pathlib import Path

from psychscanner import ExpCard, ExpCardInit, ScannerModel


class _EmptyTunnel:
    def load_tunnel_logs(self, *args, **kwargs):
        return []


def _minimal_task() -> dict:
    return {
        "tasktype": "sc",
        "taskname": "tunnel_empty_min",
        "instructions": {"definition": ["Answer briefly."]},
        "contexts": ["general"],
        "contexts_id": ["Q"],
        "context_present": False,
        "items": {"Q_1": [{"trcode": "Q_1", "stimulus": "2 + 2?"}]},
        "chain_type": "item",
        "parser": None,
    }


def test_tunnel_systemtrials_returns_none_on_empty_log(tmp_path: Path) -> None:
    card = ExpCardInit(
        model="mock-chat-model",
        family="mock-llm",
        task_file=_minimal_task(),
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
        projectname="tunnel_empty_test",
        proj_dir=tmp_path,
    )
    scanner = ScannerModel(expcard=ExpCard(card))

    resume_idx = scanner.tunnel_systemtrials(tunnel=_EmptyTunnel())

    assert resume_idx is None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_tunnel_systemtrials_returns_none_on_empty_log(Path(d))
    print("demo() OK")
