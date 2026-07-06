"""End-to-end test for the LangGraph ReAct tool-calling agent adapter."""
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from psychscanner.agents.react_agent import make_react_agent
from psychscanner.task_runner import TaskRunner


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


class _ScriptedToolCallingModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` plus a no-op ``bind_tools`` so it satisfies
    ``create_agent``'s "model must support tool calling" check; the scripted
    ``messages`` iterator drives the actual responses regardless of tools."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _scripted_model() -> GenericFakeChatModel:
    """A fake model that calls ``get_weather`` once, then answers from the result."""
    return _ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}],
                ),
                AIMessage(content="It's sunny in Paris."),
            ]
        )
    )


def test_react_agent_executes_tool_and_returns_final_answer():
    agent = make_react_agent(_scripted_model(), [get_weather])

    tasktrials = {
        "trials": [
            {
                "trcode": "t1",
                "stimulus": HumanMessage(content="What's the weather in Paris?"),
                "tasktype": "x",
                "parser": None,
                "fb": False,
            }
        ]
    }
    runner = TaskRunner(
        scanning_agent=agent,
        trace_cfg={"trial": "tut-", "task": "tut-task"},
        system_message="You are a helpful assistant.",
        tasktrials=tasktrials,
        chain_type="item",
        hmsg="stimulus",
    )
    results = runner.execute()

    assert len(results) == 1
    assert results[0]["pred_resp"].content == "It's sunny in Paris."
