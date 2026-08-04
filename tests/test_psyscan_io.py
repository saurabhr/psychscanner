"""Unit tests for psyscan_io row-building, focused on nested-response flattening.

Union-type parsers (e.g. AllResponseRMEI, used by trial-chain tasks) return a
payload nested under one key, e.g. {"response": {"Judgment": "external"}}.
polars/CSV can't hold a nested value in a cell, so _trials_to_rows must
flatten any dict/list resp_* value to a JSON string before it reaches the
DataFrame -- otherwise to_csv raises ComputeError: CSV format does not
support nested data (found when quickstart's trial-chain demo tried to
export it for the first time).
"""
from __future__ import annotations

import io
import json

import polars as pl

from psychscanner.scanner_models.psyscan_io import _rows_to_frame, _trials_to_rows


class _FakeAIMessage:
    def __init__(self, content: str):
        self.content = content


def test_nested_response_is_flattened_to_json_string():
    trials = [{
        "trcode": "imagined_1",
        "pred_resp": _FakeAIMessage("{'response': {'Judgment': 'external'}}"),
    }]

    rows = _trials_to_rows(trials, meta={})

    assert isinstance(rows[0]["resp_response"], str)
    assert json.loads(rows[0]["resp_response"]) == {"Judgment": "external"}


def test_flat_response_is_left_as_a_plain_scalar():
    trials = [{
        "trcode": "imagined_1",
        "pred_resp": _FakeAIMessage("{'Word_2': 'sandpaper', 'Rating': 3.0}"),
    }]

    rows = _trials_to_rows(trials, meta={})

    assert rows[0]["resp_Word_2"] == "sandpaper"
    assert rows[0]["resp_Rating"] == 3.0


def test_nested_response_rows_write_to_csv_without_error():
    """The actual regression: this used to raise polars.exceptions.ComputeError."""
    trials = [
        {"trcode": "imagined_1", "pred_resp": _FakeAIMessage("{'response': {'Word_2': 'sandpaper'}}")},
        {"trcode": "imagined_1", "pred_resp": _FakeAIMessage("{'response': {'Judgment': 'external'}}")},
    ]
    df = _rows_to_frame(_trials_to_rows(trials, meta={}))

    buf = io.StringIO()
    df.write_csv(buf)  # must not raise
    assert "resp_response" in df.columns
    assert df["resp_response"].dtype == pl.Utf8
