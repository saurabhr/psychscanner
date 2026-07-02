from __future__ import annotations

import ast
import json
from typing import Callable

from tqdm import tqdm
from langchain_core.messages import HumanMessage
import click


def _parse_response(pred_resp, parser_status: str) -> dict:
    """Convert an AIMessage into a plain dict for feedback handlers.

    For structured output, `pred_resp.content` is the string repr of a
    pydantic `.model_dump()`, so we use ast.literal_eval.  Unstructured
    responses are wrapped in {"content": ...}.
    """
    content = pred_resp.content
    if parser_status == "1":
        try:
            return ast.literal_eval(content)
        except Exception:
            pass
    return {"content": content}


class TaskRunner:
    def __init__(
        self,
        scanning_agent: object | None = None,
        trace_cfg: dict | None = None,
        system_message: str = None,
        tasktrials: list = None,
        chain_type: str = None,
        tunnel=None,
        hmsg="hmsg",
        feedback: bool = False,
        feedback_fn: Callable | None = None,
    ) -> None:
        self.test_agent = scanning_agent
        self.trace_cfg = trace_cfg
        self.tunnel = tunnel
        self.system_message = system_message
        self.tasktrials = tasktrials

        self.chain_type = chain_type
        if self.chain_type is None:
            self.chain_type = trace_cfg["chain_type"]

        self.stimulus_key = hmsg
        self.task_recorder = []

        self.trial_response = None
        self.input_dict = {}
        self.pred_dict = {}
        self.trial_prompt = None
        self.tr_idx = None
        self.parser_status = "0"

        # Normalize legacy "0"/"1" strings to bool
        if isinstance(feedback, str):
            feedback = feedback == "1"
        self.feedback: bool = bool(feedback)
        self.feedback_fn = feedback_fn
        self.fb_response: str | None = None

    def execute(
        self,
        test_agent: object = None,
        tasktrials: dict | None = None,
        disable_tqdm: bool = True,
    ) -> list:
        """Execute all task trials and return a list of per-trial result dicts.

        Parameters
        ----------
        test_agent:
            AI agent; defaults to the instance's scanning_agent.
        tasktrials:
            Task trial data; defaults to the instance's tasktrials.
        disable_tqdm:
            When True the tqdm bar is hidden.

        Returns
        -------
        list
            One dict per trial containing inputs, prompt, prediction, and
            optional feedback.
        """
        click.echo("----<>---- task running")
        if test_agent is None:
            test_agent = self.test_agent
        if tasktrials is None:
            tasktrials = self.tasktrials

        trial_prompts = tasktrials["trials"]

        # Instantiate the feedback handler once for the whole simulation so that
        # cross-trial state (e.g. word lists) accumulates correctly.
        fb_handler = None
        if self.feedback and self.feedback_fn is not None:
            fb_handler = self.feedback_fn()

        self.fb_response = None

        for self.tr_idx, self.trial_prompt in tqdm(
            enumerate(trial_prompts), disable=disable_tqdm
        ):
            if self.trial_prompt["tasktype"] == "episodic_system":
                if isinstance(self.trial_prompt["system_message"], dict):
                    self.system_message = json.dumps(
                        self.trial_prompt["system_message"]
                    )
                else:
                    self.system_message = (
                        self.system_message + "\n" + self.trial_prompt["system_message"]
                    )

            # Per-trial parser: trial JSON overrides card-level parser
            trial_parser = self.trial_prompt.get("parser") or self.test_agent.parser
            self.parser_status = "0" if trial_parser is None else "1"

            self.input_dict = {
                "inputs": [self.trial_prompt[self.stimulus_key]],
                "system_message": self.system_message,
                "trcode": self.trial_prompt["trcode"],
                "parser": self.trial_prompt.get("parser"),  # str or None
                "tools": self.trial_prompt.get("tools"),  # list[str] names selecting from agent_cfg.tools, or None
            }

            # Inject previous trial's feedback when enabled and available.
            # The per-trial "fb" key (default True when absent) opts in/out.
            trial_wants_fb = self.trial_prompt.get("fb", True)
            if fb_handler is not None and self.fb_response is not None and trial_wants_fb:
                self.input_dict = fb_handler.inject_feedback(
                    self.input_dict, self.fb_response
                )

            # ── invoke the agent ──────────────────────────────────────────
            # "item" chain type also needs a thread_id when the graph uses
            # MemorySaver (Convo memory).  Use the task-scoped id so that
            # conversation accumulates across all trials in one simulation run.
            # Passing thread_id to a SingleTurn graph (no checkpointer) is
            # silently ignored by LangGraph.
            thread_id = None
            if self.chain_type == "trial":
                thread_id = self.trace_cfg["trial"] + self.trial_prompt["trcode"]
            elif self.chain_type in ["task", "item"]:
                thread_id = self.trace_cfg["task"]

            if thread_id is not None:
                config = {"configurable": {"thread_id": thread_id}}
                self.pred_dict = test_agent.ai_app.invoke(self.input_dict, config=config)
            else:
                self.pred_dict = test_agent.ai_app.invoke(self.input_dict)

            pred_resp = self.pred_dict["inputs"][-1]

            # ── generate feedback for next trial ──────────────────────────
            self.fb_response = None
            if fb_handler is not None and trial_wants_fb:
                parsed_resp = _parse_response(pred_resp, self.parser_status)
                self.fb_response = fb_handler.on_response(self.trial_prompt, parsed_resp)

            self.trial_response = {
                "trial_idx": self.tr_idx,
                **self.input_dict,
                **self.trial_prompt,
                "pred_resp": pred_resp,
                "pred_dict": self.pred_dict,
                "trace_id": thread_id,
                "chain_type": self.chain_type,
                "system_message": self.system_message,
                "fb_response": self.fb_response,
            }
            self.task_recorder.append(self.trial_response)

        return self.task_recorder
