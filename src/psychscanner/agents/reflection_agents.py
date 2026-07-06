"""Basic Reflection, Reflexion, and LATS agents, after the LangChain blog post
https://www.langchain.com/blog/reflection-agents.

- **Basic Reflection**: a generate/reflect loop — a generator drafts a
  response, a reflector critiques it as a teacher would, and the generator
  redrafts. Purely self-critique, no external grounding.
- **Reflexion**: draft -> execute_tools -> revise, grounding the critique in
  real tool/search results (citations, missing/superfluous content) instead
  of the model's own opinion. One fixed trajectory — early mistakes cascade.
- **LATS** (Language Agent Tree Search): Monte Carlo tree search over
  candidate trajectories — select (UCT), expand (sample several
  continuations), reflect/evaluate (score each), backpropagate — so it
  explores multiple paths instead of one linear retry chain.

None of these exist in psychscanner today (the built-in agent is one LLM
call per trial, see ``memories.single_turn_convo``), so all three are built
directly on LangGraph and adapted to the ``ScanningAgent`` contract.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .custom_agent import CustomAgent


def _stimulus_text(stimulus: Any) -> str:
    content = getattr(stimulus, "content", stimulus)
    return content if isinstance(content, str) else str(content)


# ── 1. Basic Reflection ──────────────────────────────────────────────────────

class _ReflectionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    max_messages: int


def make_basic_reflection_agent(model: Any, *, max_messages: int = 6, parser: Any | None = None) -> CustomAgent:
    """Generate/reflect loop; the final answer is the last generator turn."""

    def _flip_roles(messages: list[BaseMessage]) -> list[BaseMessage]:
        # The reflector critiques the assistant's own answer, so it's presented
        # to the chat model as if a human wrote it (chat models expect
        # alternating human/ai turns) — same trick the reference tutorial uses.
        flipped = [messages[0]]
        for m in messages[1:]:
            cls = HumanMessage if isinstance(m, AIMessage) else AIMessage
            flipped.append(cls(content=m.content))
        return flipped

    def generate(state: _ReflectionState) -> dict:
        response = model.invoke(
            [SystemMessage(content="Answer the task below, incorporating any teacher critique above.")]
            + state["messages"]
        )
        return {"messages": [response]}

    def reflect(state: _ReflectionState) -> dict:
        response = model.invoke(
            [SystemMessage(content="You are a teacher grading an answer. Critique it specifically: "
                                    "what's missing, wrong, or could be clearer.")]
            + _flip_roles(state["messages"])
        )
        return {"messages": [HumanMessage(content=response.content)]}

    def route_after_generate(state: _ReflectionState) -> str:
        return END if len(state["messages"]) >= state["max_messages"] else "reflect"

    workflow = StateGraph(_ReflectionState)
    workflow.add_node("generate", generate)
    workflow.add_node("reflect", reflect)
    workflow.add_edge(START, "generate")
    workflow.add_conditional_edges("generate", route_after_generate, {"reflect": "reflect", END: END})
    workflow.add_edge("reflect", "generate")
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        result = graph.invoke({
            "messages": [HumanMessage(content=_stimulus_text(input_dict["inputs"][-1]))],
            "max_messages": max_messages,
        })
        last_ai = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        return AIMessage(content=last_ai.content)

    return CustomAgent(call_fn, parser=parser)


# ── 2. Reflexion ──────────────────────────────────────────────────────────────

class _ReflexionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int


def make_reflexion_agent(
    model: Any, tools: Sequence[Callable], *, max_iterations: int = 2, parser: Any | None = None
) -> CustomAgent:
    """Draft -> execute_tools -> revise loop, grounded in real tool calls.

    ``model`` must support ``bind_tools`` (``mock-llm`` does not).
    """
    model_with_tools = model.bind_tools(list(tools))
    tool_node = ToolNode(list(tools))

    def draft(state: _ReflexionState) -> dict:
        response = model_with_tools.invoke(
            [SystemMessage(content="Draft an answer to the task. Call a tool first if it would help "
                                    "ground your answer in facts.")]
            + state["messages"]
        )
        return {"messages": [response]}

    def revise(state: _ReflexionState) -> dict:
        response = model_with_tools.invoke(
            [SystemMessage(content="Revise your answer using the tool results above into a direct final answer "
                                    "(no meta-commentary, no follow-up questions). Call another tool only if you "
                                    "genuinely need to.")]
            + state["messages"]
        )
        return {"messages": [response], "iteration": state["iteration"] + 1}

    def route_after_revise(state: _ReflexionState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            return END
        last = state["messages"][-1]
        return "execute_tools" if getattr(last, "tool_calls", None) else END

    workflow = StateGraph(_ReflexionState)
    workflow.add_node("draft", draft)
    workflow.add_node("execute_tools", tool_node)
    workflow.add_node("revise", revise)
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "execute_tools")
    workflow.add_edge("execute_tools", "revise")
    workflow.add_conditional_edges("revise", route_after_revise, {"execute_tools": "execute_tools", END: END})
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        result = graph.invoke({
            "messages": [HumanMessage(content=_stimulus_text(input_dict["inputs"][-1]))],
            "iteration": 0,
            "max_iterations": max_iterations,
        })
        last_answer = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content)
        return AIMessage(content=last_answer.content)

    return CustomAgent(call_fn, parser=parser)


# ── 3. LATS (Language Agent Tree Search) ─────────────────────────────────────

class _LATSNode:
    """One candidate trajectory in the search tree."""

    def __init__(self, messages: list[BaseMessage], parent: "_LATSNode | None" = None):
        self.messages = messages
        self.parent = parent
        self.children: list[_LATSNode] = []
        self.visits = 0
        self.value = 0.0

    def uct(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits) if self.parent else 0.0
        return exploit + explore

    def best_child(self) -> "_LATSNode":
        return max(self.children, key=lambda n: n.uct())


def _select(root: _LATSNode) -> _LATSNode:
    node = root
    while node.children:
        node = node.best_child()
    return node


def _expand(model: Any, node: _LATSNode, stimulus: str, branching: int) -> list[_LATSNode]:
    children = []
    for _ in range(branching):
        response = model.invoke(
            [SystemMessage(content=f"Task: {stimulus}\nPropose an answer, distinct from any earlier attempts above.")]
            + node.messages
        )
        children.append(_LATSNode([*node.messages, response], parent=node))
    node.children = children
    return children


def _score(model: Any, stimulus: str, node: _LATSNode) -> float:
    response = model.invoke([
        SystemMessage(content="Rate the assistant's last answer from 0 (wrong/useless) to 10 (fully correct "
                               "and complete). Reply with just the number."),
        HumanMessage(content=f"Task: {stimulus}\nAnswer: {node.messages[-1].content}"),
    ])
    match = re.search(r"\d+(\.\d+)?", response.content)
    return min(float(match.group()), 10.0) / 10.0 if match else 0.5


def _backpropagate(node: _LATSNode, score: float) -> None:
    n: _LATSNode | None = node
    while n is not None:
        n.visits += 1
        n.value += score
        n = n.parent


def _best_node(root: _LATSNode) -> _LATSNode | None:
    best, stack = None, list(root.children)
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node.visits == 0:
            continue
        if best is None or (node.value / node.visits) > (best.value / best.visits):
            best = node
    return best


class _LATSState(TypedDict):
    root: _LATSNode
    stimulus: str
    iteration: int
    max_iterations: int
    branching: int
    score_threshold: float


def make_lats_agent(
    model: Any,
    *,
    max_iterations: int = 2,
    branching: int = 2,
    score_threshold: float = 0.8,
    parser: Any | None = None,
) -> CustomAgent:
    """Monte Carlo tree search over candidate answers: select / expand / reflect / backpropagate."""

    def iterate(state: _LATSState) -> dict:
        leaf = _select(state["root"])
        for child in _expand(model, leaf, state["stimulus"], state["branching"]):
            _backpropagate(child, _score(model, state["stimulus"], child))
        return {"iteration": state["iteration"] + 1}

    def route_after_iterate(state: _LATSState) -> str:
        best = _best_node(state["root"])
        if best is not None and (best.value / best.visits) >= state["score_threshold"]:
            return END
        return END if state["iteration"] >= state["max_iterations"] else "iterate"

    workflow = StateGraph(_LATSState)
    workflow.add_node("iterate", iterate)
    workflow.add_edge(START, "iterate")
    workflow.add_conditional_edges("iterate", route_after_iterate, {"iterate": "iterate", END: END})
    graph = workflow.compile()

    def call_fn(input_dict: dict) -> AIMessage:
        stimulus = _stimulus_text(input_dict["inputs"][-1])
        root = _LATSNode([HumanMessage(content=stimulus)])
        result = graph.invoke({
            "root": root, "stimulus": stimulus, "iteration": 0,
            "max_iterations": max_iterations, "branching": branching, "score_threshold": score_threshold,
        })
        best = _best_node(result["root"]) or root
        last_ai = next((m for m in reversed(best.messages) if isinstance(m, AIMessage)), None)
        return AIMessage(content=last_ai.content if last_ai else "")

    return CustomAgent(call_fn, parser=parser)
