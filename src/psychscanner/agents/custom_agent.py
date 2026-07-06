"""Plug a researcher-supplied LLM or VLM into the simulation loop.

``TaskRunner`` never imports the LangGraph agent built by ``ScannerModel`` —
it only calls ``test_agent.ai_app.invoke(input_dict, config=...)`` and reads
``test_agent.parser`` (see ``task_runner.py``). Anything satisfying that
shape can stand in for the built-in LangChain/LangGraph agent, so a custom
model needs no LangGraph knowledge: wrap it in :class:`CustomAgent` and pass
it to ``ScannerModel.run(custom_agent=...)`` or ``TaskRunner(scanning_agent=...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ScanningAgent(Protocol):
    """The contract ``TaskRunner`` expects from any scanning agent."""

    parser: Any

    def invoke(self, input_dict: dict, config: dict | None = None) -> dict: ...


class CustomAgent:
    """Adapt a plain callable to the :class:`ScanningAgent` contract.

    ``call_fn`` receives the per-trial invoke dict (keys: ``inputs``,
    ``system_message``, ``trcode``, ``parser``, ``tools``) and must return
    the trial's prediction as an object with a ``.content`` attribute (e.g.
    ``langchain_core.messages.AIMessage``). Use this to drop in any LLM or
    VLM — a raw provider SDK call, a local model, a REST API — without
    building a LangGraph graph.
    """

    def __init__(self, call_fn: Callable[[dict], Any], parser: Any | None = None):
        self.call_fn = call_fn
        self.parser = parser
        self.ai_app = self

    def invoke(self, input_dict: dict, config: dict | None = None) -> dict:
        response = self.call_fn(input_dict)
        return {**input_dict, "inputs": [*input_dict["inputs"], response]}
