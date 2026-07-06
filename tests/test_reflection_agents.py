"""Tests for the Basic Reflection, Reflexion, and LATS agents (reflection-agents blog)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from psychscanner.agents.reflection_agents import (
    make_basic_reflection_agent,
    make_lats_agent,
    make_reflexion_agent,
)
from psychscanner.task_runner import TaskRunner


class _ScriptedModel:
    """Returns each of ``responses`` in order, ignoring the input messages."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=next(self._responses))


class _ScriptedToolCallingModel(_ScriptedModel):
    """``_ScriptedModel`` that returns pre-built messages (already AIMessage,
    possibly with ``tool_calls``) as-is, plus a no-op ``bind_tools``."""

    def invoke(self, messages):
        self.calls += 1
        return next(self._responses)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _run(agent):
    tasktrials = {"trials": [{"trcode": "t1", "stimulus": HumanMessage(content="task"), "tasktype": "x", "parser": None, "fb": False}]}
    runner = TaskRunner(
        scanning_agent=agent,
        trace_cfg={"trial": "tut-", "task": "tut-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )
    return runner.execute()[0]["pred_resp"].content


# ── Basic Reflection ─────────────────────────────────────────────────────────

def test_basic_reflection_loops_until_max_messages_then_returns_last_draft():
    model = _ScriptedModel(["draft one", "needs more detail", "final draft"])
    agent = make_basic_reflection_agent(model, max_messages=4)
    assert _run(agent) == "final draft"
    assert model.calls == 3


# ── Reflexion ────────────────────────────────────────────────────────────────

@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


def test_reflexion_stops_as_soon_as_revise_stops_requesting_tools():
    model = _ScriptedToolCallingModel([])
    model._responses = iter([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "a"}, "id": "1"}]),
        AIMessage(content="final answer using result for a"),
    ])
    agent = make_reflexion_agent(model, [lookup], max_iterations=2)
    assert _run(agent) == "final answer using result for a"
    assert model.calls == 2


def test_reflexion_stops_at_max_iterations_even_if_still_requesting_tools():
    model = _ScriptedToolCallingModel([])
    model._responses = iter([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "a"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "b"}, "id": "2"}]),
        AIMessage(content="best guess: 42", tool_calls=[{"name": "lookup", "args": {"query": "c"}, "id": "3"}]),
    ])
    agent = make_reflexion_agent(model, [lookup], max_iterations=2)
    assert _run(agent) == "best guess: 42"
    assert model.calls == 3


# ── LATS ─────────────────────────────────────────────────────────────────────

def test_lats_picks_highest_scored_candidate():
    model = _ScriptedModel(["answer-good", "answer-bad", "8", "3"])
    agent = make_lats_agent(model, max_iterations=1, branching=2, score_threshold=0.8)
    assert _run(agent) == "answer-good"
    assert model.calls == 4
