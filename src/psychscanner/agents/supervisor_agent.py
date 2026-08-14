"""Multimodal supervisor -> planner -> worker agent, after Jockey (TwelveLabs + LangGraph).

https://www.langchain.com/blog/jockey-twelvelabs-langgraph describes Jockey, a
conversational video agent built from a **Supervisor** (routes and coordinates),
a **Planner** (breaks complex requests into steps), and specialized **Workers**
(video search / text generation / editing). psychscanner has no video
pipeline, so this generalizes the same three-role shape to whatever content
blocks a trial's stimulus actually carries (image / audio / text, see
``psychscanner.datasets.prompts.multimodal``): a single-block-type stimulus is
routed straight to the matching worker; a mixed-block stimulus goes through
the planner, which orders the workers needed and dispatches each on its slice
of the content, then an aggregate step combines their findings.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .custom_agent import CustomAgent

_ROLE_FOR_BLOCK = {"image": "vision_worker", "audio": "audio_worker", "text": "text_worker"}


def _blocks(stimulus: Any) -> list:
    content = getattr(stimulus, "content", stimulus)
    return content if isinstance(content, list) else [content]


def _block_types(stimulus: Any) -> set[str]:
    types = set()
    for block in _blocks(stimulus):
        block_type = block.get("type") if isinstance(block, dict) else None
        types.add(block_type if block_type in ("image", "audio") else "text")
    # An empty content list (stimulus.content == []) has no blocks to loop
    # over, so this would otherwise return an empty set -- and
    # route_after_supervisor's `(single_type,) = _block_types(...)` crashes
    # unpacking it. Treat "no blocks" the same as "one text block", matching
    # the loop's own default-to-text handling above.
    return types or {"text"}


def _blocks_of_type(stimulus: Any, block_type: str) -> list:
    def matches(block: Any) -> bool:
        this_type = block.get("type") if isinstance(block, dict) else None
        return (this_type if this_type in ("image", "audio") else "text") == block_type

    return [block for block in _blocks(stimulus) if matches(block)]


class _SupervisorState(TypedDict):
    stimulus: Any
    trcode: str
    plan: list[str]
    step_outputs: list[str]


def make_supervisor_agent(model: Any, *, parser: Any | None = None) -> CustomAgent:
    """Build the Jockey-style supervisor/planner/worker graph as a ``ScanningAgent``."""

    def _run_worker(role: str, content: Any) -> str:
        message = HumanMessage(content=content)
        response = model.invoke(
            [SystemMessage(content=f"You are the {role} in a multimodal analysis pipeline. "
                                    "Describe concisely what your slice of the stimulus shows."), message]
        )
        return f"[{role}] {response.content}"

    def supervisor(state: _SupervisorState) -> dict:
        types = _block_types(state["stimulus"])
        return {"plan": [] if len(types) <= 1 else [
            _ROLE_FOR_BLOCK[t] for t in ("image", "audio", "text") if t in types
        ]}

    def route_after_supervisor(state: _SupervisorState) -> str:
        if state["plan"]:
            return "planner"
        (single_type,) = _block_types(state["stimulus"])
        return _ROLE_FOR_BLOCK[single_type]

    def planner(state: _SupervisorState) -> dict:
        # The routing plan is already computed in `supervisor`; this node is
        # the seam where a real planner (an LLM call producing/reordering
        # `state["plan"]`) would slot in for more elaborate multi-step tasks.
        return {}

    def dispatch(state: _SupervisorState) -> dict:
        outputs = [
            _run_worker(role, _blocks_of_type(state["stimulus"], block_type))
            for block_type, role in _ROLE_FOR_BLOCK.items()
            if role in state["plan"]
        ]
        return {"step_outputs": outputs}

    def single_worker(role: str):
        def run(state: _SupervisorState) -> dict:
            return {"step_outputs": [_run_worker(role, state["stimulus"].content
                                                  if hasattr(state["stimulus"], "content")
                                                  else state["stimulus"])]}
        return run

    def aggregate(state: _SupervisorState) -> dict:
        outputs = state.get("step_outputs") or []
        if len(outputs) <= 1:
            content = outputs[0] if outputs else ""
        else:
            response = model.invoke([
                SystemMessage(content="Combine these worker findings into one concise answer."),
                HumanMessage(content="\n".join(outputs)),
            ])
            content = response.content
        return {"step_outputs": [content]}

    workflow = StateGraph(_SupervisorState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("planner", planner)
    workflow.add_node("dispatch", dispatch)
    workflow.add_node("aggregate", aggregate)
    for role in _ROLE_FOR_BLOCK.values():
        workflow.add_node(role, single_worker(role))
        workflow.add_edge(role, "aggregate")

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor", route_after_supervisor, {"planner": "planner", **{r: r for r in _ROLE_FOR_BLOCK.values()}}
    )
    workflow.add_edge("planner", "dispatch")
    workflow.add_edge("dispatch", "aggregate")
    workflow.add_edge("aggregate", END)
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        stimulus = input_dict["inputs"][-1]
        result = graph.invoke({"stimulus": stimulus, "trcode": input_dict.get("trcode", ""), "plan": [], "step_outputs": []})
        return AIMessage(content=result["step_outputs"][0] if result.get("step_outputs") else "")

    return CustomAgent(call_fn, parser=parser)
