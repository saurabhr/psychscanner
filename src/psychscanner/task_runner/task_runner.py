from __future__ import annotations

import ast
import json
from typing import Callable

from tqdm import tqdm
from langchain_core.messages import HumanMessage
import click

from psychscanner.datasets.prompts.task_prompts import gen_stimulus_prompt


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
        next_trial: bool = False,
        next_trial_fn: Callable | None = None,
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
        self.fb_handler = None
        self.fb_response: str | None = None

        if isinstance(next_trial, str):
            next_trial = next_trial == "1"
        self.next_trial: bool = bool(next_trial)
        self.next_trial_fn = next_trial_fn
        self.nt_handler = None

    def _run_single(self, trial_dict: dict, tr_idx, *, is_intermediate: bool = False):
        """Invoke the agent on one trial dict and record the result.

        Shared by the main task-card loop and by conditional intermediate
        trials inserted via ``next_trial_fn`` - both need the same episodic
        system-message handling, feedback injection, thread_id selection,
        and result bookkeeping.

        Returns
        -------
        tuple
            ``(pred_resp, parsed_resp)`` - the raw AIMessage and its parsed dict.
        """
        self.trial_prompt = trial_dict
        self.tr_idx = tr_idx

        if trial_dict.get("tasktype") == "episodic_system" and "system_message" in trial_dict:
            if isinstance(trial_dict["system_message"], dict):
                self.system_message = json.dumps(trial_dict["system_message"])
            else:
                self.system_message = self.system_message + "\n" + trial_dict["system_message"]

        # Per-trial parser: trial JSON overrides card-level parser
        trial_parser = trial_dict.get("parser") or self.test_agent.parser
        self.parser_status = "0" if trial_parser is None else "1"

        self.input_dict = {
            "inputs": [trial_dict[self.stimulus_key]],
            "system_message": self.system_message,
            "trcode": trial_dict["trcode"],
            "parser": trial_dict.get("parser"),  # str or None
            "tools": trial_dict.get("tools"),  # list[str] names selecting from agent_cfg.tools, or None
        }

        # Inject previous trial's feedback when enabled and available.
        # The per-trial "fb" key (default True when absent) opts in/out.
        trial_wants_fb = trial_dict.get("fb", True)
        if self.fb_handler is not None and self.fb_response is not None and trial_wants_fb:
            self.input_dict = self.fb_handler.inject_feedback(self.input_dict, self.fb_response)

        # ── invoke the agent ──────────────────────────────────────────
        # "item" chain type also needs a thread_id when the graph uses
        # MemorySaver (Convo memory).  Use the task-scoped id so that
        # conversation accumulates across all trials in one simulation run.
        # Passing thread_id to a SingleTurn graph (no checkpointer) is
        # silently ignored by LangGraph.
        thread_id = None
        if self.chain_type == "trial":
            thread_id = self.trace_cfg["trial"] + trial_dict["trcode"]
        elif self.chain_type in ["task", "item"]:
            thread_id = self.trace_cfg["task"]

        if thread_id is not None:
            config = {"configurable": {"thread_id": thread_id}}
            self.pred_dict = self.test_agent.ai_app.invoke(self.input_dict, config=config)
        else:
            self.pred_dict = self.test_agent.ai_app.invoke(self.input_dict)

        pred_resp = self.pred_dict["inputs"][-1]
        parsed_resp = _parse_response(pred_resp, self.parser_status)

        # ── generate feedback for next trial ──────────────────────────
        self.fb_response = None
        if self.fb_handler is not None and trial_wants_fb:
            self.fb_response = self.fb_handler.on_response(trial_dict, parsed_resp)

        self.trial_response = {
            "trial_idx": tr_idx,
            "is_intermediate": is_intermediate,
            **self.input_dict,
            **trial_dict,
            "pred_resp": pred_resp,
            "pred_dict": self.pred_dict,
            "trace_id": thread_id,
            "chain_type": self.chain_type,
            "system_message": self.system_message,
            "fb_response": self.fb_response,
        }
        self.task_recorder.append(self.trial_response)

        return pred_resp, parsed_resp

    def _materialize_intermediate(self, raw_trial: dict, template_trial: dict) -> dict:
        """Turn a handler-authored trial dict (task-JSON shape) into an
        executable one carrying a pre-built ``hmsg`` HumanMessage, the same
        shape every trial in ``tasktrials["trials"]`` already has.

        Fields not supplied by the handler (context flags, tasktype) fall
        back to the trial it's inserted after, so a handler only has to
        specify what's actually changing (typically ``trcode``/``stimulus``).
        """
        trial = {
            "context_present": template_trial.get("context_present", False),
            "context_item": template_trial.get("context_item"),
            "tasktype": template_trial.get("tasktype"),
            "taskname": template_trial.get("taskname"),
            "context": template_trial.get("context"),
            "trid": template_trial.get("trid"),
            **raw_trial,
        }
        trial[self.stimulus_key] = gen_stimulus_prompt(trial)
        return trial

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
            optional feedback. Trials inserted by ``next_trial_fn`` carry
            ``is_intermediate=True``.
        """
        click.echo("----<>---- task running")
        if test_agent is None:
            test_agent = self.test_agent
        self.test_agent = test_agent
        if tasktrials is None:
            tasktrials = self.tasktrials

        trial_prompts = tasktrials["trials"]

        # Instantiate the feedback/next-trial handlers once for the whole
        # simulation so that cross-trial state (e.g. word lists) accumulates
        # correctly.
        self.fb_handler = None
        if self.feedback and self.feedback_fn is not None:
            self.fb_handler = self.feedback_fn()

        self.nt_handler = None
        if self.next_trial and self.next_trial_fn is not None:
            self.nt_handler = self.next_trial_fn()

        self.fb_response = None

        # A running execution-order counter, not the task-card index: it
        # stays a plain int (required by TrialSimulationModel.trial_idx)
        # even once intermediate trials are interleaved with the original
        # sequence, and is the same as the task-card index whenever
        # next_trial_fn never fires.
        exec_idx = 0

        for trial_prompt in tqdm(trial_prompts, disable=disable_tqdm):
            pred_resp, parsed_resp = self._run_single(trial_prompt, exec_idx)
            exec_idx += 1
            current_trial = trial_prompt

            # ── conditional intermediate trials ─────────────────────────
            # A handler may keep inserting new trials before the task
            # card's next one (e.g. adaptive/staircase designs). Guard
            # against a handler that keeps proposing the same stimulus by
            # breaking the chain once it repeats max_repeat times in a row,
            # and resuming the original trial sequence from the task card.
            if self.nt_handler is not None:
                last_stim_key = None
                repeat_streak = 0
                while True:
                    new_trial = self.nt_handler.next_trial(current_trial, parsed_resp)
                    if new_trial is None:
                        break

                    stim_key = json.dumps(new_trial.get("stimulus"), sort_keys=True, default=str)
                    if stim_key == last_stim_key:
                        repeat_streak += 1
                    else:
                        repeat_streak = 1
                        last_stim_key = stim_key
                    if repeat_streak > self.nt_handler.max_repeat:
                        break

                    exec_trial = self._materialize_intermediate(new_trial, current_trial)
                    pred_resp, parsed_resp = self._run_single(
                        exec_trial, exec_idx, is_intermediate=True
                    )
                    exec_idx += 1
                    current_trial = exec_trial

        return self.task_recorder
