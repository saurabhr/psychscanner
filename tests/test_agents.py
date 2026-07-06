"""CustomAgent lets a researcher plug any LLM/VLM callable into TaskRunner."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from psychscanner.agents import CustomAgent
from psychscanner.task_runner import TaskRunner


def test_custom_agent_drives_taskrunner_without_langgraph():
    calls = []

    def my_vlm(input_dict: dict) -> AIMessage:
        calls.append(input_dict["trcode"])
        return AIMessage(content=f"echo:{input_dict['inputs'][-1]}")

    agent = CustomAgent(my_vlm, parser=None)
    tasktrials = {
        "trials": [
            {"trcode": "t1", "stimulus": "hello", "tasktype": "x", "parser": None, "fb": False},
        ]
    }
    runner = TaskRunner(
        scanning_agent=agent,
        trace_cfg={"trial": "sess-", "task": "sess-task"},
        system_message="sys",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )

    results = runner.execute()

    assert calls == ["t1"]
    assert results[0]["pred_resp"].content == "echo:hello"
