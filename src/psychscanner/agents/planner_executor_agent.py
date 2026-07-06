"""Modular planner / executor / validator agent (arxiv:2310.00194).

"Improving Planning with Large Language Models: A Modular Agentic
Architecture" (https://arxiv.org/pdf/2310.00194) separates planning
cognition from execution: a **Planner** produces a high-level step
decomposition rather than solving the task directly, an **Executor**
translates the plan into a concrete answer, and a **Validator** checks the
result and can send the loop back to the planner with feedback — informed by
a rolling **Memory** of prior attempts — instead of one end-to-end
generation. Nothing like this exists in psychscanner today (the built-in
agent is a single LLM call, see ``memories.single_turn_convo``), so this
builds the loop directly on LangGraph, bounded to ``max_iterations`` passes.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .custom_agent import CustomAgent


class _PlannerExecutorState(TypedDict):
    stimulus: Any
    trcode: str
    plan: str
    execution: str
    feedback: str
    memory: list[str]
    iteration: int
    max_iterations: int


def make_planner_executor_agent(
    model: Any, *, max_iterations: int = 2, parser: Any | None = None
) -> CustomAgent:
    """Build the planner/executor/validator graph and adapt it to ``ScanningAgent``."""

    def _stimulus_text(stimulus: Any) -> str:
        content = getattr(stimulus, "content", stimulus)
        return content if isinstance(content, str) else str(content)

    def planner(state: _PlannerExecutorState) -> dict:
        history = "\n".join(state["memory"]) or "(no prior attempts)"
        response = model.invoke([
            SystemMessage(content="You are the Planner. Produce a short, numbered step-by-step "
                                   "plan for answering the task. Do not answer it yourself."),
            HumanMessage(content=f"Task: {_stimulus_text(state['stimulus'])}\n\nPrior attempts:\n{history}"),
        ])
        return {"plan": response.content, "memory": [*state["memory"], f"Plan: {response.content}"]}

    def executor(state: _PlannerExecutorState) -> dict:
        response = model.invoke([
            SystemMessage(content="You are the Executor. Carry out the plan below and give the final answer."),
            HumanMessage(content=f"Task: {_stimulus_text(state['stimulus'])}\n\nPlan:\n{state['plan']}"),
        ])
        return {"execution": response.content, "memory": [*state["memory"], f"Execution: {response.content}"]}

    def validator(state: _PlannerExecutorState) -> dict:
        response = model.invoke([
            SystemMessage(content="You are the Validator. Reply with exactly 'ACCEPT' if the execution "
                                   "correctly answers the task, otherwise reply 'REVISE: <what's wrong>'."),
            HumanMessage(content=f"Task: {_stimulus_text(state['stimulus'])}\n\nExecution:\n{state['execution']}"),
        ])
        feedback = response.content
        return {
            "feedback": feedback,
            "memory": [*state["memory"], f"Validator: {feedback}"],
            "iteration": state["iteration"] + 1,
        }

    def route_after_validator(state: _PlannerExecutorState) -> str:
        accepted = state["feedback"].strip().upper().startswith("ACCEPT")
        return END if accepted or state["iteration"] >= state["max_iterations"] else "planner"

    workflow = StateGraph(_PlannerExecutorState)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("validator", validator)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "validator")
    workflow.add_conditional_edges("validator", route_after_validator, {"planner": "planner", END: END})
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        result = graph.invoke({
            "stimulus": input_dict["inputs"][-1],
            "trcode": input_dict.get("trcode", ""),
            "plan": "",
            "execution": "",
            "feedback": "",
            "memory": [],
            "iteration": 0,
            "max_iterations": max_iterations,
        })
        return AIMessage(content=result["execution"])

    return CustomAgent(call_fn, parser=parser)
