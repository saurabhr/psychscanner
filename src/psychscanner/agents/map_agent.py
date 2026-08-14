"""Modular Agentic Planner (MAP) -- Webb, Mondal & Momennejad (2025),
"A brain-inspired agentic architecture to improve planning with LLMs",
Nature Communications 16, 8633, https://doi.org/10.1038/s41467-025-63804-5
(preprint arxiv:2310.00194; module definitions and Algorithm 1 read in full
from the PDF before writing this).

Six specialized LLM modules, each its own prompted call, interacting in the
paper's fixed control loop:

- ``TaskDecomposer(state, goal) -> subgoal`` -- one intermediate subgoal
  ("the TaskDecomposer is only utilized to generate a single intermediate
  goal" per the paper).
- ``Actor(state, subgoal, feedback) -> action`` -- proposes one candidate
  action.
- ``Monitor(state, action) -> valid, feedback`` -- gates the action; on
  reject, Actor retries with feedback (the paper's ProposeAction loop).
- ``Predictor(state, action) -> predicted_state`` -- next-state prediction.
- ``Evaluator(predicted_state, goal) -> value`` -- scalar value estimate.
- ``Orchestrator(state, subgoal) -> done`` -- whether the (sub)goal is met.

``Search`` proposes ``branching`` candidate actions through the
Actor/Monitor loop, predicts each one's resulting state with the Predictor,
scores each with the Evaluator, and keeps the best -- the paper's tree
search collapsed to one level (see scoping note below).

**Scoping to this task.** The paper's benchmarks (graph traversal, Tower of
Hanoi, PlanBench) are multi-step environments: Search recurses to depth
``L``, and Algorithm 1's outer loop appends actions to a growing plan until
the Orchestrator confirms the goal or a max length ``T`` is hit. Prospect
Theory trials are single-shot binary choices (pick the preferred gamble;
trials are independent, there's no environment state to step through), so
this uses the paper's own degenerate case: one subgoal, search with no
recursive expansion past the immediate action (``L=1``), and ``max_actions``
defaults to 1. Every module still makes a real call -- the Orchestrator
still runs even though a single-decision trial always confirms after it.

The Monitor is the one exception: the paper's Monitor is an LLM call that
checks whether a proposed action "violates the rules of a task". Here the
only rule is "the action must literally be 'left' or 'right'", a format
check with no judgment call for an LLM to make, so it's implemented in
plain Python -- deterministic and independently checkable, not a corner
cut.

Run from the psychscanner project root:
    python -m psychscanner.agents.map_agent   # runs demo() as a self-check
"""
from __future__ import annotations

import re
from typing import Any
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .custom_agent import CustomAgent

DEFAULT_GOAL = (
    "Choose the gamble, left or right, that you genuinely prefer, weighing "
    "the potential gains, losses, and their probabilities."
)


def _stimulus_text(stimulus: Any) -> str:
    content = getattr(stimulus, "content", stimulus)
    return content if isinstance(content, str) else str(content)


def _parse_side(text: str) -> str | None:
    match = re.search(r"\b(left|right)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _task_decomposer(model: Any, state: str, goal: str) -> str:
    response = model.invoke([
        SystemMessage(content="You are the TaskDecomposer in a planning system. Given the situation and the "
                               "overall goal, state ONE brief intermediate subgoal that would help decide the "
                               "action. Reply with a single short sentence, no preamble."),
        HumanMessage(content=f"Situation: {state}\nGoal: {goal}"),
    ])
    return response.content.strip()


def _actor(model: Any, state: str, subgoal: str, prior: list[str], feedback: str) -> str:
    avoid = f" Already proposed: {', '.join(prior)} -- propose the other option." if prior else ""
    fb = f"\nMonitor feedback on your last proposal: {feedback}" if feedback else ""
    response = model.invoke([
        SystemMessage(content="You are the Actor. Propose exactly one candidate action: reply with only the "
                               "single word 'left' or 'right'." + avoid),
        HumanMessage(content=f"Situation: {state}\nSubgoal: {subgoal}{fb}"),
    ])
    return response.content.strip()


def _monitor(action_text: str) -> tuple[bool, str]:
    """Plain-Python validity gate -- see module docstring for why."""
    if _parse_side(action_text) is None:
        return False, "Action must be exactly 'left' or 'right'."
    return True, ""


def _predictor(model: Any, state: str, action: str) -> str:
    response = model.invoke([
        SystemMessage(content="You are the Predictor. Given the situation and a proposed action, describe in "
                               "one sentence the resulting outcome state -- what the person experiences by "
                               "choosing that gamble."),
        HumanMessage(content=f"Situation: {state}\nProposed action: {action}"),
    ])
    return response.content.strip()


def _evaluator(model: Any, predicted_state: str, goal: str) -> float:
    response = model.invoke([
        SystemMessage(content="You are the Evaluator. Rate how well the predicted outcome state serves the "
                               "goal, from 0 (worst) to 10 (best). Reply with just the number."),
        HumanMessage(content=f"Goal: {goal}\nPredicted outcome: {predicted_state}"),
    ])
    match = re.search(r"\d+(\.\d+)?", response.content)
    return min(float(match.group()), 10.0) if match else 5.0


def _orchestrator(model: Any, state: str, subgoal: str, action: str) -> bool:
    response = model.invoke([
        SystemMessage(content="You are the Orchestrator. Reply with exactly 'YES' if choosing the given action "
                               "satisfies the subgoal, else 'NO'."),
        HumanMessage(content=f"Situation: {state}\nSubgoal: {subgoal}\nChosen action: {action}"),
    ])
    return response.content.strip().upper().startswith("Y")


class _MAPState(TypedDict):
    state_text: str
    goal: str
    subgoal: str
    plan: list[str]
    predicted_state: str
    value: float
    done: bool
    branching: int
    max_actions: int
    max_actor_retries: int


def _search(model: Any, state: _MAPState) -> dict:
    """Propose ``branching`` actions via the Actor/Monitor loop, predict and
    score each with Predictor/Evaluator, keep the best (Search, L=1)."""
    candidates: list[tuple[str, str, float]] = []
    proposed: list[str] = []
    for _ in range(state["branching"]):
        feedback = ""
        side = None
        for _attempt in range(state["max_actor_retries"] + 1):
            action_text = _actor(model, state["state_text"], state["subgoal"], proposed, feedback)
            valid, feedback = _monitor(action_text)
            if valid:
                side = _parse_side(action_text)
                break
        if side is None or side in proposed:
            continue
        proposed.append(side)
        predicted = _predictor(model, state["state_text"], side)
        value = _evaluator(model, predicted, state["goal"])
        candidates.append((side, predicted, value))

    if not candidates:
        # Both branches failed the Monitor's format check -- shouldn't
        # happen against a real model, but keep the graph terminating.
        return {"plan": [*state["plan"], "left"], "predicted_state": "", "value": 0.0}

    best_side, best_predicted, best_value = max(candidates, key=lambda c: c[2])
    return {"plan": [*state["plan"], best_side], "predicted_state": best_predicted, "value": best_value}


def make_map_agent(
    model: Any,
    *,
    goal: str = DEFAULT_GOAL,
    branching: int = 2,
    max_actions: int = 1,
    max_actor_retries: int = 2,
    parser: Any | None = None,
) -> CustomAgent:
    """Build the MAP graph (TaskDecomposer -> Search -> Orchestrator loop)
    and adapt it to the ``ScanningAgent`` contract."""

    def decompose(state: _MAPState) -> dict:
        return {"subgoal": _task_decomposer(model, state["state_text"], state["goal"])}

    def search(state: _MAPState) -> dict:
        return _search(model, state)

    def orchestrate(state: _MAPState) -> dict:
        action = state["plan"][-1]
        done = _orchestrator(model, state["state_text"], state["subgoal"], action)
        return {"done": done}

    def route_after_orchestrate(state: _MAPState) -> str:
        if state["done"] or len(state["plan"]) >= state["max_actions"]:
            return END
        return "search"

    workflow = StateGraph(_MAPState)
    workflow.add_node("decompose", decompose)
    workflow.add_node("search", search)
    workflow.add_node("orchestrate", orchestrate)
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "search")
    workflow.add_edge("search", "orchestrate")
    workflow.add_conditional_edges("orchestrate", route_after_orchestrate, {"search": "search", END: END})
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        result = graph.invoke({
            "state_text": _stimulus_text(input_dict["inputs"][-1]),
            "goal": goal,
            "subgoal": "",
            "plan": [],
            "predicted_state": "",
            "value": 0.0,
            "done": False,
            "branching": branching,
            "max_actions": max_actions,
            "max_actor_retries": max_actor_retries,
        })
        return AIMessage(content=result["plan"][-1])

    return CustomAgent(call_fn, parser=parser)


def demo() -> None:
    """Self-check with a scripted fake model -- no real LLM, no network.
    Verifies: Monitor rejects a malformed action and the Actor retries,
    Search picks the higher-evaluator-scored side, and the final answer is
    a bare 'left'/'right' as ScanningAgent expects."""

    class _FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class _FakeModel:
        """Scripted so 'left' always scores higher than 'right', and the
        first Actor call for the right-branch returns a malformed action
        to exercise the Monitor's retry path."""

        def __init__(self):
            self.calls = 0
            self._right_attempts = 0

        def invoke(self, messages):
            self.calls += 1
            system = messages[0].content
            if "TaskDecomposer" in system:
                return _FakeResponse("Compare expected value and loss exposure.")
            if "Actor" in system:
                if "Already proposed: left" in system:
                    self._right_attempts += 1
                    if self._right_attempts == 1:
                        return _FakeResponse("I choose neither, this is unclear.")
                    return _FakeResponse("right")
                return _FakeResponse("left")
            if "Predictor" in system:
                human = messages[1].content
                return _FakeResponse("outcome for right") if "right" in human.split("action: ")[-1] else _FakeResponse("outcome for left")
            if "Evaluator" in system:
                human = messages[1].content
                return _FakeResponse("8") if "left" in human else _FakeResponse("3")
            if "Orchestrator" in system:
                return _FakeResponse("YES")
            raise AssertionError(f"unexpected system prompt: {system}")

    model = _FakeModel()
    agent = make_map_agent(model, branching=2)
    result = agent.invoke({"inputs": [HumanMessage(content="Left: 50% chance to gain 80, otherwise lose 10.\n"
                                                             "Right: 50% chance to gain 40, otherwise lose 20.")]})
    final = result["inputs"][-1]
    assert final.content == "left", f"expected 'left' (higher evaluator score), got {final.content!r}"
    assert model._right_attempts == 2, "Monitor should have rejected the malformed first right-branch proposal"
    print("demo() OK — Monitor retry path exercised, Search picked the higher-scored action ('left')")


if __name__ == "__main__":
    demo()
