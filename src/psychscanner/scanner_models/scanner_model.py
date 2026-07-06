"""Scanner model sets up scanning for the task.

This module defines the ScannerModel class, which processes experimental card data
and generates system and trial prompt data for tasks.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Callable
from pathlib import Path
import click
from psychscanner.simulation_model import (
    SimulationModel,
    TaskSimulationModel,
    TrialInfoModel,
    PredSimulationModel,
    InputSimulationModel,
)
from ..memories import llm_chat_model, AgentInitializer
from ..memories.single_turn_convo import single_turn_convo_node
from ..staging import factory_settings
from .agent_config import AgentConfig
from .scanner_data import scanner_data
from .media_store import externalize_media
from psychscanner.task_runner import TaskRunner
DATETIME_FMT = '%Y%m%dT%H-%M-%S'  # YYYY-MM-DD_HH-MM-SS
class ScannerModel:
    def __init__(self,expcard):
        self.expcard = expcard
        self.projectname = self.expcard.card_in.projectname
        self.proj_dir = self.expcard.card_in.proj_dir
        self.data_root_dir = self.expcard.data_root_dir

        self.scanner_data = scanner_data(self.expcard)

        self.agent_config = AgentConfig(
            modelname=self.expcard.card_in.model,
            familyname=self.expcard.card_in.family,
            parameters=self.expcard.card_in.parameters,
            modelobject=llm_chat_model(
                model=self.expcard.card_in.model,
                family=self.expcard.card_in.family,
                parameters=self.expcard.card_in.parameters,
            ),
            memory_type=self.expcard.card_in.memory,
            memory_k=self.expcard.card_in.memory_k,
            summary_k=self.expcard.card_in.summary_k,
            chain_type=self.expcard.task_data["chain_type"],
            system_msg=None,
            parser=self.expcard.parser,
            parser_raw=self.expcard.card_in.parser_raw,
            parser_config=self.expcard.card_in.parser_config,
            tools=self.expcard.tools,
        )

        self.tqdm_progress_flag = expcard.card_in.enabletqdm
        self.tunnel_status = self.expcard.card_in.tunnel_status
        self.tunnel = expcard.session_tunnel
        self.tunnel_data = None
        self.continuing_scan = False
        if self.tunnel_status == "0":
            self.continuing_scan = False
        elif self.tunnel_status == "1":
            self.continuing_scan = True
        self.current_scanner_data = None

        self.feedback = expcard.card_in.feedback

    def tunnel_systemtrials(self, tunnel: Any | None = None) -> int | None:
        """Process tunnel data and determine the resume index.

        Parameters:
        ----------
        tunnel : Optional[Any], optional
            The tunnel object to load logs from. If None, the instance's tunnel is used.

        Returns"
        -------
        Optional[int]
            The index to resume scanning from, or None if no resume index is found.

        Raises"
        ------
        ValueError
            If the session has already ended or if the tunnel data does not match the scanner system prompts data.
        """
        if tunnel is None:
            tunnel = self.tunnel

        self.tunnel_data = tunnel.load_tunnel_logs()

        scan_on_off_events = [i for i in self.tunnel_data if i["level"] == "CRITICAL"]
        scan_info = [i for i in self.tunnel_data if i["level"] == "INFO"]

        try:
            last_scan_on_off_events = scan_on_off_events[-1]
            # check_end = True if i[""]

        except IndexError:
            last_scan_on_off_events = None
        if last_scan_on_off_events["run_type"] == "END":
            msg = "Session already has ended. Delete old files to run."
            raise ValueError(msg)

        idx = None  # Initialize resume_idx with a default value

        try:
            last_scan_info = scan_info[-1]

        except IndexError:
            last_scan_info = None

        if last_scan_info is not None:
            state_completed = last_scan_info["state"]["len_completed_system_prompts"]
            # state_all = last_scan_info["state"]["all_system_msgs_data"]

            # if state_all != self.system_prompt_data:
            #    msg = "Session tunnel system intializiation data does not match scanner system prompts data."
            #    raise ValueError(msg)

            ######### RESUME IDX SCANNER
            idx = state_completed # this is equal to the resuming index

        return idx

    def model_dump(
        self,
        data: object | None = None,
        sim_idx: int | str = "curr_scan",
        data_root_dir: Path | None = None,
        save_str: str = "iterations",
        data_type:str="session"
    ) -> None:
        """Dump the current scanner data to a file.

        Parameters:
        ----------
        data : object, optional
            The data to be dumped. If None, the current scanner data is used.
        sim_idx : int or str, optional
            The simulation index or identifier, by default "curr_scan".
        data_root_dir : Path, optional
            The root directory where the data will be saved. If None, the instance's data root directory is used.
        save_str : str, optional
            A string to append to the filename, by default "iterations".

        Returns:
        -------
        None
        """
        if data is None:
            data = self.current_scanner_data
            save_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if data_root_dir is None:
            data_root_dir = self.data_root_dir
        if not data_root_dir.exists():
            data_root_dir.mkdir(parents=True, exist_ok=True)

        media_dir = data_root_dir / "media"
        if data_type == "session":
            data = [externalize_media(task_trials, media_dir) for task_trials in data]
            data_string = SimulationModel(simdata=data).model_dump_json(indent=4)
        elif data_type == "task":
            data = externalize_media(data, media_dir)
            data_string = TaskSimulationModel(taskdata=data).model_dump_json(indent=4)
        else:
            msg = f"Invalid data_type: {data_type}. Expected 'session' or 'task'."
            raise ValueError(msg)

        with (data_root_dir / f"{sim_idx}-{save_str}.psyscan").open("w") as handle:
            handle.write(data_string)


    def agent(self, agent_cfg=None, memory=None, custom_agent=None):
        """Return the scanning agent, or a researcher-supplied one if given.

        ``custom_agent`` bypasses the built-in LangChain/LangGraph pipeline
        entirely; it only needs to satisfy the ``ScanningAgent`` contract
        (``.ai_app.invoke(...)`` + ``.parser``) — see
        ``psychscanner.agents.CustomAgent``.
        """
        if custom_agent is not None:
            return custom_agent

        if agent_cfg is None:
            agent_cfg = self.agent_config

        if memory is None:
            memory = self.agent_config.memory_type

        if memory in ["SingleTurn", "Convo"]:
            agent = AgentInitializer(agent_cfg=agent_cfg)
            agent.ai_app = single_turn_convo_node(agent_cfg)
        else:
            click.echo("memory type not supported, can be only SingleTurn, or Convo.")

        return agent


    def run(
        self,
        progress_bar: bool = False,
        feedback: Any | None = None,
        feedback_fn: Callable | None = None,
        save_str: str | None = None,
        tunnel: Any | None = None,
        custom_agent: Any | None = None,
    ):

        session_id = (
            datetime.now().strftime(DATETIME_FMT)
            + self.expcard.card_in.projectname
            + self.expcard.card_in.memory
            + self.expcard.task_data["taskname"]
        )
        trace_cfg = {
            "item": session_id,
            "trial": session_id,
            "task": session_id,
        }

        if feedback is None:
            feedback = self.feedback

        if feedback_fn is None:
            if feedback:
                feedback_fn = self.expcard.card_in.feedback_fn

        # self.tunnel_data
        if not progress_bar:  # if false then overwritten by tqdm_progress_flag (see __init__ above) based on expcard.
            progress_bar = bool(self.tqdm_progress_flag)
        if tunnel is not None:
            self.tunnel = tunnel
        tunnel_id = None
        self.tunnel.create_tunnel()

        resume_idx = None  # Initialize resume_idx with a default value
        if self.continuing_scan:
            resume_idx = self.tunnel_systemtrials()

        scanning_system_data = self.scanner_data["system_prompts"]
        trace_cfg["chain_type"] = self.scanner_data["chain_type"]
        scanning_task_data = self.scanner_data["task_prompts"]
        scan_agent = self.agent(custom_agent=custom_agent)
        self.current_scanner_data = []
        system_data_completed = []
        MAX_TEST_SESSION_TUNNEL = 1 #5  # for testing session tunnel otherwise -9999
        click.echo(f"TOTAL RUNS: {len(scanning_system_data)}\tRESUME IDX: {resume_idx}")

        for sys_sim_i,sys_msg_i in enumerate(scanning_system_data):


            if resume_idx is not None and sys_sim_i < resume_idx:
                system_data_completed.append(sys_msg_i)
                continue

            trace_cfg["task"] = session_id + str(sys_sim_i)
            tunnel_id = f"sidx-[{sys_sim_i}]-model-{self.expcard.card_in.model}-family-{self.expcard.card_in.family}-memory-{self.expcard.card_in.memory}-{self.projectname}"

            scan_sys_i = TaskRunner(
                scanning_agent=scan_agent,
                trace_cfg=trace_cfg,
                system_message=sys_msg_i,
                tasktrials=scanning_task_data,
                feedback=feedback,
                feedback_fn=feedback_fn,
            )
            scan_i_data = scan_sys_i.execute(disable_tqdm=progress_bar)
            scan_i_data = [
                {
                    **i,
                    "system_message_idx": sys_sim_i,
                    "system_template": self.scanner_data["system_template"],
                    "tunnel_id": tunnel_id
                }
                for i in scan_i_data
            ]
            self.current_scanner_data.append(scan_i_data)
            if save_str is None:
                save_str = datetime.now().strftime("%Y%m%d_%H%M%S")


            self.model_dump(
                sim_idx=sys_sim_i,
                data=self.current_scanner_data[-1],
                save_str=save_str,
                data_type="task",
            )

            system_data_completed.append(sys_msg_i)
            self.tunnel.scan_checkpoint(
                session_id=tunnel_id,
                run_type=f"SCAN-SystemMessage-IDX-[{sys_sim_i}]",
                state={
                    # "all_system_msgs_data": systemtrials,
                    # "completed_system_prompts": system_data_completed,
                    "len_completed_system_prompts": len(system_data_completed)
                },
            )
            click.echo(f"----<scanned runs>---- i = {sys_sim_i}")
            # if len(system_data_completed) == MAX_TEST_SESSION_TUNNEL:
            #    click.echo("----<breaking scanning>----")
            #    break  # for testing session tunnel
            if (sys_sim_i + 1) == len(scanning_system_data):
                self.tunnel.end_checkpoint()

        # self.current_scanner_data = SimulationModel(simdata=self.current_scanner_data)

        return self.current_scanner_data




