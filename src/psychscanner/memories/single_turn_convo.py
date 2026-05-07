from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict, NotRequired
from typing import Sequence

from .base.base_agent import AgentInitializer
from psychscanner.parsers import resolve_parser


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

    if summary_k > 0 and len(overflow) >= summary_k:
        existing = state.get("summary") or ""
        updates["summary"] = _make_summary(overflow, existing, agent_cfg.modelobject)

    # Remove overflow messages from LangGraph state regardless of summarization
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
        if parser_cls is None:
            runnable = prompt | agent_cfg.modelobject
            invoke_input = {
                "system_message": system_msg,
                "inputs": messages,
            }
            response = runnable.invoke(invoke_input)
        else:
            runnable = prompt | agent_cfg.modelobject.with_structured_output(
                parser_cls,
                include_raw=agent_cfg.parser_raw,
                **agent_cfg.parser_config,
            )
            invoke_input = {
                "system_message": system_msg,
                "inputs": messages,
            }
            response = runnable.invoke(invoke_input)
            if agent_cfg.parser_raw:
                response = response["raw"]
            else:
                response = AIMessage(str(response.model_dump()))

        # ── 5. Return response + any trim/summary state updates ───────────────
        return {**trim_updates, "inputs": [response]}

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
