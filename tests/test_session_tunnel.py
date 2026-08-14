"""Regression test: load_tunnel_logs must survive a truncated trailing line.

A run killed mid-write can leave a partial JSONL line in the tunnel log --
exactly the interruption checkpointing exists to survive. Before the fix,
json.loads() on that line raised uncaught, so *every* subsequent resume
attempt (tunnel_status="1") crashed before ever reaching the resume logic.
"""
from __future__ import annotations

import json
from pathlib import Path

from psychscanner.session_tunnel.session_tunnel import SessionTunnel


def _good_line(session_id: str) -> str:
    record = {
        "timestamp": 0.0,
        "level": "CRITICAL",
        "run_type": "BEGIN",
        "session_id": session_id,
        "state": None,
    }
    return json.dumps({"record": {"extra": {"serialized": json.dumps(record)}}})


def test_load_tunnel_logs_skips_truncated_trailing_line(tmp_path: Path) -> None:
    tunnel = SessionTunnel(tunnel_status="1", project_name="proj", tunnel_dir=tmp_path)
    tunnel.tunnel_file.write_text(
        _good_line("s0") + "\n" + _good_line("s1") + "\n" + '{"record": {"extra": {"ser' # truncated
    )

    logs = tunnel.load_tunnel_logs(return_all=False)

    assert len(logs) == 2
    assert [r["session_id"] for r in logs] == ["s0", "s1"]


def test_load_tunnel_logs_skips_blank_lines(tmp_path: Path) -> None:
    tunnel = SessionTunnel(tunnel_status="1", project_name="proj", tunnel_dir=tmp_path)
    tunnel.tunnel_file.write_text(_good_line("s0") + "\n\n")

    logs = tunnel.load_tunnel_logs(return_all=False)
    assert len(logs) == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_load_tunnel_logs_skips_truncated_trailing_line(Path(d))
        test_load_tunnel_logs_skips_blank_lines(Path(d) / "b")
    print("demo() OK")
