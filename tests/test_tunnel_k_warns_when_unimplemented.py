"""tunnel_k is documented but was never read anywhere in the run path --
setting it silently had no effect. Rather than implement per-trial
checkpointing (a real architecture change), ScannerModel now warns so a
user configuring it isn't silently misled.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from psychscanner import ExpCard, ExpCardInit, ScannerModel


def _minimal_task() -> dict:
    return {
        "tasktype": "sc",
        "taskname": "tunnel_k_min",
        "instructions": {"definition": ["Answer briefly."]},
        "contexts": ["general"],
        "contexts_id": ["Q"],
        "context_present": False,
        "items": {"Q_1": [{"trcode": "Q_1", "stimulus": "2 + 2?"}]},
        "chain_type": "item",
        "parser": None,
    }


def _card(tmp_path: Path, tunnel_k: int) -> ExpCard:
    card = ExpCardInit(
        model="mock-chat-model",
        family="mock-llm",
        task_file=_minimal_task(),
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
        projectname="tunnel_k_test",
        proj_dir=tmp_path,
        tunnel_k=tunnel_k,
    )
    return ExpCard(card)


def test_default_tunnel_k_does_not_warn(tmp_path: Path, recwarn) -> None:
    ScannerModel(expcard=_card(tmp_path, -1))
    assert not any("tunnel_k" in str(w.message) for w in recwarn)


def test_non_default_tunnel_k_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="tunnel_k"):
        ScannerModel(expcard=_card(tmp_path, 5))


if __name__ == "__main__":
    import tempfile
    import warnings as _warnings

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            ScannerModel(expcard=_card(base / "a", -1))
            assert not w
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            ScannerModel(expcard=_card(base / "b", 5))
            assert w and "tunnel_k" in str(w[0].message)
    print("demo() OK")
