"""Unit tests for per-trial tool-name resolution against the card-level tool pool."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from psychscanner.memories.single_turn_convo import _resolve_trial_tools
from psychscanner.task_runner import TaskRunner


@tool
def image_zoom(region: str) -> str:
    """Return a zoomed-in crop of the display for the named region."""
    return region


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return query


def test_resolve_trial_tools_none_falls_back_to_full_pool():
    resolved = _resolve_trial_tools(None, [image_zoom, web_search])
    assert resolved == [image_zoom, web_search]


def test_resolve_trial_tools_empty_list_opts_out():
    resolved = _resolve_trial_tools([], [image_zoom, web_search])
    assert resolved == []


def test_resolve_trial_tools_selects_named_subset():
    resolved = _resolve_trial_tools(["image_zoom"], [image_zoom, web_search])
    assert resolved == [image_zoom]


def test_resolve_trial_tools_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown tool name"):
        _resolve_trial_tools(["not_a_real_tool"], [image_zoom])


def test_resolve_trial_tools_no_available_pool():
    assert _resolve_trial_tools(None, None) == []
    with pytest.raises(ValueError, match="unknown tool name"):
        _resolve_trial_tools(["image_zoom"], None)


def test_taskrunner_carries_per_trial_tools_into_input_dict():
    """TaskRunner threads the trial JSON's "tools" key through to invoke state."""
    captured_inputs = []

    class _FakeAIApp:
        def invoke(self, input_dict, config=None):
            captured_inputs.append(input_dict)
            return {**input_dict, "inputs": [*input_dict["inputs"], AIMessage(content="ok")]}

    class _FakeAgent:
        parser = None
        ai_app = _FakeAIApp()

    tasktrials = {
        "trials": [
            {"trcode": "t1", "stimulus": "a", "tasktype": "x", "parser": None, "fb": False, "tools": ["image_zoom"]},
            {"trcode": "t2", "stimulus": "b", "tasktype": "x", "parser": None, "fb": False, "tools": []},
            {"trcode": "t3", "stimulus": "c", "tasktype": "x", "parser": None, "fb": False},
        ]
    }
    runner = TaskRunner(
        scanning_agent=_FakeAgent(),
        trace_cfg={"trial": "sess-", "task": "sess-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )

    runner.execute()

    assert [i["tools"] for i in captured_inputs] == [["image_zoom"], [], None]
