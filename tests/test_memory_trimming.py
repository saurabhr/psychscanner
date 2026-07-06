"""Regression test for Convo memory context trimming (memory_k / summary_k).

Guards against the ``call_model`` return clobbering the RemoveMessage list
computed by ``_trim_history`` with the plain ``"inputs": [response]`` update —
which previously made memory_k a no-op (history grew unbounded).
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from psychscanner.memories.base.mock_llm import ChatMockModel
from psychscanner.memories.single_turn_convo import single_turn_convo_node
from psychscanner.scanner_models.agent_config import AgentConfig


def _build_graph(memory_k, summary_k):
    agent_cfg = AgentConfig(
        modelname="mock-chat-model",
        familyname="mock-llm",
        parameters=None,
        modelobject=ChatMockModel(model="mock-chat-model", repeat_buffer_length=10),
        memory_type="Convo",
        memory_k=memory_k,
        summary_k=summary_k,
        chain_type="task",
        system_msg=None,
        parser=None,
        parser_raw=False,
        parser_config={},
    )
    return single_turn_convo_node(agent_cfg)


def _run_turns(graph, n_turns, thread_id="t1"):
    config = {"configurable": {"thread_id": thread_id}}
    state = None
    for i in range(n_turns):
        state = graph.invoke(
            {
                "inputs": [HumanMessage(content=f"turn {i}")],
                "system_message": "sys",
                "trcode": f"tr{i}",
            },
            config=config,
        )
    return state


def test_memory_k_caps_history_length():
    graph = _build_graph(memory_k=4, summary_k=0)
    state = _run_turns(graph, n_turns=6)
    # Without the fix this grows unbounded (12 messages after 6 turns).
    assert len(state["inputs"]) <= 5


def test_summary_k_populates_rolling_summary_once_overflow_hits_threshold():
    graph = _build_graph(memory_k=4, summary_k=2)
    state = _run_turns(graph, n_turns=6)
    assert state.get("summary")


def test_unlimited_memory_k_grows_history():
    graph = _build_graph(memory_k=-1, summary_k=0)
    state = _run_turns(graph, n_turns=6)
    assert len(state["inputs"]) == 12
