"""Tests for the planner/executor/validator agent (arxiv:2310.00194)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from psychscanner.agents.planner_executor_agent import make_planner_executor_agent
from psychscanner.task_runner import TaskRunner


class _ScriptedModel:
    """Returns each of ``responses`` in order, one per ``.invoke`` call."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=next(self._responses))


def _run(agent):
    tasktrials = {"trials": [{"trcode": "t1", "stimulus": HumanMessage(content="2+2?"), "tasktype": "x", "parser": None, "fb": False}]}
    runner = TaskRunner(
        scanning_agent=agent,
        trace_cfg={"trial": "tut-", "task": "tut-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )
    return runner.execute()[0]["pred_resp"].content


def test_accepts_on_first_pass_without_looping():
    model = _ScriptedModel(["1. Add the numbers", "4", "ACCEPT"])
    agent = make_planner_executor_agent(model, max_iterations=2)
    assert _run(agent) == "4"
    assert model.calls == 3


def test_revises_once_then_accepts():
    model = _ScriptedModel(["1. Guess", "5", "REVISE: wrong", "1. Add carefully", "4", "ACCEPT"])
    agent = make_planner_executor_agent(model, max_iterations=2)
    assert _run(agent) == "4"
    assert model.calls == 6


def test_stops_at_max_iterations_even_without_accept():
    model = _ScriptedModel(["1. Guess", "5", "REVISE: wrong", "1. Guess again", "6", "REVISE: still wrong"])
    agent = make_planner_executor_agent(model, max_iterations=2)
    assert _run(agent) == "6"
    assert model.calls == 6
