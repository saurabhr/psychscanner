"""Regression tests for conditional intermediate trials in TaskRunner.

Covers: a handler inserting a new trial before the task card's next one,
and the recursive-break guard that resumes the original trial sequence once
a handler keeps proposing the same stimulus too many times in a row.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from psychscanner.feedback import NextTrialBase
from psychscanner.memories.base.mock_llm import ChatMockModel
from psychscanner.memories.base.base_agent import AgentInitializer
from psychscanner.memories.single_turn_convo import single_turn_convo_node
from psychscanner.scanner_models.agent_config import AgentConfig
from psychscanner.task_runner import TaskRunner


def _build_agent():
    agent_cfg = AgentConfig(
        modelname="mock-chat-model",
        familyname="mock-llm",
        parameters=None,
        modelobject=ChatMockModel(model="mock-chat-model", repeat_buffer_length=20),
        memory_type="SingleTurn",
        memory_k=-1,
        summary_k=0,
        chain_type="item",
        system_msg=None,
        parser=None,
        parser_raw=False,
        parser_config={},
    )
    agent = AgentInitializer(agent_cfg=agent_cfg)
    agent.ai_app = single_turn_convo_node(agent_cfg)
    return agent


def _trial(trcode, stimulus):
    return {
        "trcode": trcode,
        "stimulus": stimulus,
        "hmsg": HumanMessage(content=stimulus),
        "tasktype": "survey",
        "context_present": False,
    }


def _run(nt_handler_cls, trials):
    runner = TaskRunner(
        scanning_agent=_build_agent(),
        trace_cfg={"item": "t", "trial": "t", "task": "t", "chain_type": "item"},
        system_message="sys",
        tasktrials={"trials": trials},
        chain_type="item",
        next_trial=True,
        next_trial_fn=nt_handler_cls,
    )
    return runner.execute()


def test_next_trial_inserts_one_intermediate_trial():
    class InsertOnce(NextTrialBase):
        def next_trial(self, trial, response):
            if trial["trcode"] == "t0" and not trial.get("_is_retry"):
                return {"trcode": "t0_retry", "stimulus": "please clarify", "_is_retry": True}
            return None

    trials = [_trial("t0", "first question"), _trial("t1", "second question")]
    recorder = _run(InsertOnce, trials)

    trcodes = [r["trcode"] for r in recorder]
    assert trcodes == ["t0", "t0_retry", "t1"]
    assert [r["is_intermediate"] for r in recorder] == [False, True, False]


def test_next_trial_stops_after_max_repeat_and_resumes_task_card():
    class LoopForever(NextTrialBase):
        max_repeat = 3

        def next_trial(self, trial, response):
            # Always proposes the exact same stimulus - would hang the run
            # without the recursive-break guard.
            return {"trcode": "loop", "stimulus": "same stimulus every time"}

    trials = [_trial("t0", "first question"), _trial("t1", "second question")]
    recorder = _run(LoopForever, trials)

    trcodes = [r["trcode"] for r in recorder]
    # After t0: exactly max_repeat=3 identical "loop" trials, then the guard
    # breaks the chain and t1 (the task card's next trial) resumes - the
    # handler fires again after t1 and repeats the same pattern.
    assert trcodes == ["t0", "loop", "loop", "loop", "t1", "loop", "loop", "loop"]
    assert recorder[4]["is_intermediate"] is False  # t1 resumed, not another "loop"
