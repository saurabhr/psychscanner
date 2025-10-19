from __future__ import annotations
from tqdm import tqdm
from langchain_core.messages import HumanMessage
import click
from typing import Optional
import json

class TaskRunner:
    def __init__(
        self,
        scanning_agent: object | None = None,
        trace_cfg: dict | None = None,
        system_message: str = None,
        tasktrials: list = None,
        chain_type: str = None,
        tunnel = None,
        hmsg="hmsg",
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
        self.input_dict = {} # trial input
        self.pred_dict = {} # prediction input
        self.trial_prompt = None
        self.tr_idx = None
        self.tr_ai_resp_value = None
        if self.test_agent.parser == "0":
            self.parser_status = "0"
        else:
            self.parser_status = "1"

        self.stim_collector = []

    def execute(
        self,
        test_agent: object = None,
        tasktrials: dict | None = None,
        disable_tqdm: bool = True,
    ) -> list:  # noqa: FBT001, FBT002
        """Executes the task trials and records the results.

        Parameters:
        ----------
        ai_agent : object, optional
            The AI agent used for predictions. Defaults to the instance's ai_agent.
        tasktrials : dict, optional
            A dictionary containing task trial data. Defaults to the instance's tasktrials.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to True.

        Returns:
        -------
        list
            A list of dictionaries containing the results of each trial.
        """
        click.echo("----<>---- task running")
        if test_agent is None:
            test_agent = self.test_agent

        if tasktrials is None:
            tasktrials = self.tasktrials

        trial_prompts = tasktrials["trials"]

        for self.tr_idx, self.trial_prompt in tqdm(
            enumerate(trial_prompts), disable=disable_tqdm):
            if self.trial_prompt["tasktype"] == "episodic_system":
                if isinstance(self.trial_prompt["system_message"], dict):
                    esystem_message = json.dumps(self.trial_prompt["system_message"])
                    self.system_message = self.system_message + "\n" + esystem_message
                else:
                    self.system_message = self.system_message + "\n" + self.trial_prompt["system_message"]
            self.input_dict = {
                "inputs": [self.trial_prompt[self.stimulus_key]],
                "system_message": self.system_message,
                "trcode": self.trial_prompt["trcode"],
            }
            
            thread_id = None
            if self.chain_type == "trial":
                thread_id = self.trace_cfg["trial"] + self.trial_prompt["trcode"]
            elif self.chain_type == "task":
                thread_id = self.trace_cfg["task"]
            elif self.chain_type == "item":
                thread_id = None

            if self.chain_type in ["trial","task"]:
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                    }
                }
                self.pred_dict = self.test_agent.ai_app.invoke(self.input_dict,config=config)
            else:
                self.pred_dict = self.test_agent.ai_app.invoke(
                    self.input_dict
                )

            pred_resp = self.pred_dict["inputs"][-1]
            click.echo(f"--<input_dict>-- {self.input_dict}")
            click.echo(f"---<pred_dict>-- {pred_resp}")
            click.echo(f"---<chain_type>-- {self.chain_type}")
            click.echo(f"---<thread_id>-- {thread_id}")

            self.trial_response = {
                "trial_idx": self.tr_idx,
                **self.input_dict,
                **self.trial_prompt,
                "pred_resp": pred_resp,
                "pred_dict": self.pred_dict,
                "trace_id": thread_id,
                "chain_type": self.chain_type,
                "system_message": self.system_message,
            }
            self.task_recorder.append(self.trial_response)

        return self.task_recorder
