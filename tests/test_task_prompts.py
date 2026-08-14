"""Regression tests for gen_trial_promptdata's context/context_item handling.

Covers two real bugs found in review: a trailing comma turning `context`
into a 1-tuple on every trial, and an unconditional contexts_id.index()
lookup crashing any context_present=False task (including the project's
own get_task_template("sc") skeleton, which ships contexts_id="").
"""
from __future__ import annotations

from types import SimpleNamespace

from psychscanner.datasets.prompts.task_prompts import gen_trial_promptdata


def _expcard(task_data: dict) -> SimpleNamespace:
    return SimpleNamespace(task_data=task_data)


def test_context_is_a_plain_string_not_a_tuple():
    task_data = {
        "chain_type": "item",
        "on_file": {
            "taskname": "t",
            "tasktype": "sc",
            "context_present": True,
            "contexts": ["Context A"],
            "contexts_id": ["S1"],
        },
        "items": {"S1_1": [{"trcode": "S1_1", "stimulus": "hello"}]},
    }
    trials = gen_trial_promptdata(_expcard(task_data))["trials"]
    assert trials[0]["context"] == "S1"
    assert trials[0]["context_item"] == "Context A"


def test_context_present_false_does_not_crash_and_yields_empty_context_item():
    # Matches get_task_template("sc")'s shape: context_present="" (falsy),
    # contexts_id="" -- .index() on this would previously raise ValueError.
    task_data = {
        "chain_type": "item",
        "on_file": {
            "taskname": "t",
            "tasktype": "sc",
            "context_present": "",
            "contexts": "",
            "contexts_id": "",
        },
        "items": {"Q_1": [{"trcode": "Q_1", "stimulus": "2 + 2?"}]},
    }
    trials = gen_trial_promptdata(_expcard(task_data))["trials"]
    assert trials[0]["context"] == "Q"
    assert trials[0]["context_item"] == ""


if __name__ == "__main__":
    test_context_is_a_plain_string_not_a_tuple()
    test_context_present_false_does_not_crash_and_yields_empty_context_item()
    print("demo() OK")
