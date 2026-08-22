"""Regression test for the checkpoint-persistence bug found in code review:
TrialSimulationModel didn't declare fb_response/is_intermediate/parser/tools,
so Pydantic v2's default extra="ignore" silently dropped them from every
persisted .psyscan checkpoint -- the feedback signal this package's
feedback-scored tasks depend on was never actually reaching disk.

Exercises the real path: TaskRunner.execute() with a feedback handler
produces trial_response dicts -> TaskSimulationModel.model_dump_json(),
the exact serialization ScannerModel uses to write checkpoints.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from psychscanner.feedback import FeedbackBase
from psychscanner.memories.base.mock_llm import ChatMockModel
from psychscanner.memories.base.base_agent import AgentInitializer
from psychscanner.memories.single_turn_convo import single_turn_convo_node
from psychscanner.scanner_models.agent_config import AgentConfig
from psychscanner.simulation_model.simulation_model import TaskSimulationModel
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
        # gen_trial_promptdata() (task_prompts.py) normally adds these to
        # every trial before TaskRunner sees it; this test drives TaskRunner
        # directly (same shortcut test_conditional_next_trial.py takes), so
        # they're supplied by hand to satisfy TrialSimulationModel's schema.
        "trid": trcode,
        "context": None,
        "context_item": "",
        "taskname": "demo",
    }


def test_feedback_and_intermediate_fields_survive_checkpoint_serialization():
    class RewardFeedback(FeedbackBase):
        def on_response(self, trial, response):
            return "Reward: +1.0"

    trials = [_trial("t0", "first"), _trial("t1", "second")]
    runner = TaskRunner(
        scanning_agent=_build_agent(),
        trace_cfg={"item": "t", "trial": "t", "task": "t", "chain_type": "item"},
        system_message="sys",
        tasktrials={"trials": trials},
        chain_type="item",
        feedback=True,
        feedback_fn=RewardFeedback,
    )
    recorder = runner.execute()

    # Sanity: the in-memory recorder has the fields (this part never broke).
    assert recorder[0]["fb_response"] == "Reward: +1.0"
    assert recorder[0]["is_intermediate"] is False

    # ScannerModel.run() merges these three keys onto every trial dict
    # before model_dump (scanner_model.py) -- replicate that here since
    # this test drives TaskRunner directly.
    recorder = [
        {**r, "system_message_idx": 0, "system_template": "sys", "tunnel_id": "t"}
        for r in recorder
    ]

    # The actual bug: did these fields survive the checkpoint round-trip?
    dumped = json.loads(TaskSimulationModel(taskdata=recorder).model_dump_json())
    row = dumped["taskdata"][0]
    assert row["fb_response"] == "Reward: +1.0"
    assert row["is_intermediate"] is False
    assert "parser" in row  # was previously dropped entirely (extra="ignore")
    assert "tools" in row
