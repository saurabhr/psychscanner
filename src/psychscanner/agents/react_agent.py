"""A real tool-calling loop, via LangGraph's prebuilt ReAct agent.

``psychscanner``'s built-in agent (``psychscanner.memories.single_turn_convo``)
binds tools with ``model.bind_tools(...)`` but never executes them or loops on
the result — see the "Tool Binding" section of ``docs/guides/cognitive_tasks.md``.
That's the pattern LangChain's own agent docs describe as *the* definition of an
agent: "a model calling tools in a loop until a given task is complete"
(https://docs.langchain.com/oss/python/langchain/agents). LangGraph already ships
that loop as ``langgraph.prebuilt.create_react_agent``, so this module adapts it
to the ``ScanningAgent`` contract rather than reimplementing it.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain.agents import create_agent

from .custom_agent import CustomAgent


def make_react_agent(
    model: Any,
    tools: Sequence[Callable],
    *,
    system_message: str | None = None,
    parser: Any | None = None,
) -> CustomAgent:
    """Wrap ``langchain.agents.create_agent`` as a ``ScanningAgent``.

    ``model`` is any LangChain chat model that supports ``bind_tools``
    (``mock-llm`` does not — use ``ollama``/``openai``/``anthropic`` etc., see
    ``psychscanner.memories.llm_chat_model``). The tool set is fixed for the
    life of the returned agent: unlike the built-in agent's per-trial
    ``"tools"`` JSON subsetting, ``create_agent`` binds one tool set for every
    trial — build a separate agent per subset if you need that.
    """
    graph = create_agent(model, list(tools), system_prompt=system_message)

    def call_fn(input_dict: dict) -> Any:
        result = graph.invoke({"messages": input_dict["inputs"]})
        return result["messages"][-1]

    return CustomAgent(call_fn, parser=parser)
