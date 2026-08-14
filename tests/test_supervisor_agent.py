"""Tests for the Jockey-style supervisor/planner/worker agent."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from psychscanner.agents.supervisor_agent import make_supervisor_agent
from psychscanner.task_runner import TaskRunner


class _EchoModel:
    """Returns an AIMessage naming the role (from the system prompt) and content it saw."""

    def invoke(self, messages):
        system_msg, human_msg = messages
        role = "worker"
        for candidate in ("vision_worker", "audio_worker", "text_worker", "pipeline"):
            if candidate in system_msg.content:
                role = candidate
                break
        return AIMessage(content=f"{role}:{human_msg.content}")


def _run(agent, stimulus, trcode="t1"):
    tasktrials = {"trials": [{"trcode": trcode, "stimulus": stimulus, "tasktype": "x", "parser": None, "fb": False}]}
    runner = TaskRunner(
        scanning_agent=agent,
        trace_cfg={"trial": "tut-", "task": "tut-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )
    return runner.execute()[0]["pred_resp"].content


def test_single_block_type_routes_directly_to_one_worker():
    agent = make_supervisor_agent(_EchoModel())
    content = _run(agent, HumanMessage(content=[{"type": "image", "base64": "x"}]))
    assert content.startswith("[vision_worker]") and "vision_worker:" in content


def test_mixed_blocks_route_through_planner_and_aggregate():
    agent = make_supervisor_agent(_EchoModel())
    content = _run(
        agent,
        HumanMessage(content=[{"type": "image", "base64": "x"}, {"type": "text", "text": "describe this"}]),
    )
    assert "vision_worker:" in content and "text_worker:" in content


def test_empty_content_list_does_not_crash_routing():
    """Regression: content=[] made _block_types return an empty set, and
    route_after_supervisor's `(single_type,) = _block_types(...)` crashed
    unpacking it -- ValueError: not enough values to unpack."""
    agent = make_supervisor_agent(_EchoModel())
    content = _run(agent, HumanMessage(content=[]))
    assert content.startswith("[text_worker]")
