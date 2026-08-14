"""Tests for the MAP (Modular Agentic Planner) agent, Webb/Mondal/Momennejad 2025."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from psychscanner.agents.map_agent import make_map_agent
from psychscanner.task_runner import TaskRunner


class _RoutedModel:
    """Dispatches on which module's system prompt is calling, like the
    scripted models in test_reflection_agents.py but keyed by module name
    since MAP's modules aren't a strict linear sequence (Search calls Actor
    up to ``branching`` times, each possibly retried by the Monitor)."""

    def __init__(self, *, decompose="subgoal", actor_by_call=None, predictor="outcome",
                 evaluator_by_call=None, orchestrator="YES"):
        self.decompose = decompose
        self.actor_by_call = iter(actor_by_call or [])
        self.predictor = predictor
        self.evaluator_by_call = iter(evaluator_by_call or [])
        self.orchestrator = orchestrator
        self.calls = 0
        self.actor_situations = []

    def invoke(self, messages):
        self.calls += 1
        system = messages[0].content
        if "TaskDecomposer" in system:
            return AIMessage(content=self.decompose)
        if "Actor" in system:
            self.actor_situations.append(messages[1].content)
            return AIMessage(content=next(self.actor_by_call))
        if "Predictor" in system:
            return AIMessage(content=self.predictor)
        if "Evaluator" in system:
            return AIMessage(content=next(self.evaluator_by_call))
        if "Orchestrator" in system:
            return AIMessage(content=self.orchestrator)
        raise AssertionError(f"unexpected system prompt: {system}")


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


def test_map_picks_the_higher_evaluator_scored_action():
    model = _RoutedModel(actor_by_call=["left", "right"], evaluator_by_call=["8", "3"])
    agent = make_map_agent(model, branching=2)
    assert _run(agent) == "left"
    # decompose(1) + actor(2) + predictor(2) + evaluator(2) + orchestrator(1) = 8
    assert model.calls == 8


def test_map_monitor_rejects_malformed_action_and_actor_retries():
    model = _RoutedModel(
        actor_by_call=["left", "not sure", "right"],
        evaluator_by_call=["3", "8"],
    )
    agent = make_map_agent(model, branching=2, max_actor_retries=2)
    assert _run(agent) == "right"


def test_map_loops_back_to_search_when_orchestrator_says_not_done():
    model = _RoutedModel(
        actor_by_call=["left", "right", "left", "right"],
        evaluator_by_call=["8", "3", "8", "3"],
        orchestrator="NO",
    )
    agent = make_map_agent(model, branching=2, max_actions=2)
    assert _run(agent) == "left"
    # decompose(1) + 2 search rounds x (actor2 + predictor2 + evaluator2 = 6) + orchestrator(2)
    assert model.calls == 1 + 2 * 6 + 2
    # Regression: round 2's Actor call must plan against round 1's
    # predicted_state ("outcome", from _RoutedModel's default predictor
    # response), not the original trial stimulus ("task") every round.
    round1_situations = model.actor_situations[:2]
    round2_situations = model.actor_situations[2:]
    assert all("Situation: task" in s for s in round1_situations)
    assert all("Situation: outcome" in s for s in round2_situations), model.actor_situations
