from __future__ import annotations

import logging

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict, NotRequired
from typing import Sequence
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from .base.base_agent import AgentInitializer
from psychscanner.parsers import resolve_parser

logger = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True for HTTP-429 / rate-limit errors from any LLM provider."""
    if "RateLimit" in type(exc).__name__:
        return True
    msg = str(exc)
    return "429" in msg or "rate_limit_exceeded" in msg.lower() or "rate limit" in msg.lower()


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential_jitter(initial=2, max=64),
    stop=stop_after_attempt(8),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _invoke_with_retry(runnable, invoke_input: dict):
    return runnable.invoke(invoke_input)


def _resolve_trial_tools(trial_tool_names: list[str] | None, available_tools: list | None) -> list:
    """Resolve a trial's ``"tools"`` name list against the card-level tool pool.

    ``trial_tool_names=None`` (key absent from the trial JSON) falls back to
    the full ``available_tools`` pool. An explicit list — including ``[]`` —
    selects that exact subset by ``BaseTool.name`` (or ``__name__`` for plain
    functions), letting a trial opt out of tools entirely with ``"tools": []``.
    """
    if trial_tool_names is None:
        return available_tools or []

    by_name = {getattr(t, "name", None) or getattr(t, "__name__", None): t for t in (available_tools or [])}
    unknown = [name for name in trial_tool_names if name not in by_name]
    if unknown:
        msg = f"Trial references unknown tool name(s) {unknown}; not in card-level tools {list(by_name)}."
        raise ValueError(msg)
    return [by_name[name] for name in trial_tool_names]


def _make_summary(messages: list, existing_summary: str, model) -> str:
    """Summarize a batch of messages, folding in any existing summary."""
    existing_block = (
        f"Existing summary:\n{existing_summary}\n\n" if existing_summary else ""
    )
    history_text = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in messages
    )
    prompt = (
        f"{existing_block}"
        f"Summarize the following conversation concisely, preserving key facts:\n\n"
        f"{history_text}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


def _trim_history(state, agent_cfg) -> dict:
    """Return state updates for message trimming and optional summarization.

    Returns an empty dict when no trimming is needed.
    Only active when memory_k > 0 and memory_type is Convo.
    """
    memory_k = agent_cfg.memory_k
    summary_k = agent_cfg.summary_k or 0

    if not memory_k or memory_k < 0:
        return {}

    messages = list(state["inputs"])
    if len(messages) <= memory_k:
        return {}

    overflow = messages[:-memory_k]
    updates = {}

    if summary_k > 0:
        if len(overflow) < summary_k:
            # Threshold not reached yet -- per docs/guides/memory_types.md,
            # "summary_k is a threshold, not a batch size: once the overflow
            # reaches summary_k messages, the entire overflow is
            # summarized". Leave overflow in state (don't remove it) so it
            # keeps accumulating toward the threshold on a later call,
            # instead of being silently deleted before ever being folded
            # into a summary.
            return {}
        existing = state.get("summary") or ""
        updates["summary"] = _make_summary(overflow, existing, agent_cfg.modelobject)

    # Remove overflow messages from LangGraph state: unconditionally when
    # summarization is disabled (summary_k=0, plain hard truncation), or
    # once the threshold above was actually met and summarized.
    updates["inputs"] = [RemoveMessage(id=m.id) for m in overflow]
    return updates


def single_turn_convo_node(
    agent_cfg, workflow=None, nodename="sigconvo", compile_graph=True, add_start=True,
):
    prompt = agent_cfg.agent_prompt
    if prompt is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                MessagesPlaceholder(variable_name="inputs"),
            ]
        )

    class State(TypedDict):
        inputs:         Annotated[Sequence[BaseMessage], add_messages]
        system_message: str
        trcode:         str
        parser:         NotRequired[str | None]   # per-trial parser name from task JSON
        tools:          NotRequired[list[str] | None]  # per-trial tool-name subset from task JSON
        summary:        NotRequired[str]           # rolling summary of trimmed messages

    def call_model(state: State):
        # ── 1. Resolve parser for this trial ─────────────────────────────────
        trial_parser_name = state.get("parser")

        if trial_parser_name and trial_parser_name != "0":
            parser_cls = resolve_parser(trial_parser_name)
        elif callable(agent_cfg.parser) and not isinstance(agent_cfg.parser, type):
            result = agent_cfg.parser(state["trcode"])
            parser_cls = resolve_parser(result) if isinstance(result, str) else result
        else:
            parser_cls = agent_cfg.parser

        # ── 2. Trim history and optionally summarize ──────────────────────────
        trim_updates = _trim_history(state, agent_cfg)
        # Apply trimmed message list locally for this invocation
        if "inputs" in trim_updates:
            removed_ids = {u.id for u in trim_updates["inputs"]}
            messages = [m for m in state["inputs"] if m.id not in removed_ids]
        else:
            messages = list(state["inputs"])

        # ── 3. Inject rolling summary into system message ─────────────────────
        system_msg = state["system_message"]
        current_summary = trim_updates.get("summary") or state.get("summary") or ""
        if current_summary:
            system_msg = f"{system_msg}\n\n[Conversation summary: {current_summary}]"

        # ── 4. Build runnable and invoke ──────────────────────────────────────
        invoke_input = {
            "system_message": system_msg,
            "inputs": messages,
        }
        # Tools bind fresh per call, same as with_structured_output below,
        # since one agent_cfg.modelobject is reused across every trial. A
        # trial JSON "tools" key selects a subset of agent_cfg.tools by name
        # (falls back to the full pool when absent); combining tools with a
        # parser is provider-dependent — both ride the tool-calling protocol
        # and most providers only honor one enforced tool_choice at a time.
        model = agent_cfg.modelobject
        trial_tools = _resolve_trial_tools(state.get("tools"), agent_cfg.tools)
        if trial_tools:
            model = model.bind_tools(trial_tools)

        if parser_cls is None:
            runnable = prompt | model
            response = _invoke_with_retry(runnable, invoke_input)
        else:
            runnable = prompt | model.with_structured_output(
                parser_cls,
                include_raw=agent_cfg.parser_raw,
                **agent_cfg.parser_config,
            )
            response = _invoke_with_retry(runnable, invoke_input)
            if agent_cfg.parser_raw:
                response = response["raw"]
            else:
                response = AIMessage(str(response.model_dump()))

        # ── 5. Return response + any trim/summary state updates ───────────────
        # trim_updates["inputs"] (RemoveMessage entries marking overflow for
        # deletion) must ride in the SAME list as the new response — a second
        # "inputs" key here would just overwrite the first in this dict literal.
        return {**trim_updates, "inputs": [*trim_updates.get("inputs", []), response]}

    if workflow is None:
        workflow = StateGraph(state_schema=State)

    workflow.add_node(nodename, call_model)
    if add_start:
        workflow.add_edge(START, nodename)

    if compile_graph:
        if agent_cfg.memory_type == "SingleTurn":
            return workflow.compile()
        elif agent_cfg.memory_type == "Convo":
            return workflow.compile(checkpointer=MemorySaver())

    return workflow
