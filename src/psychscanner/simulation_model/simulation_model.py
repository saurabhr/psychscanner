"""SIMULATION VALIDATION MODEL.

This module defines data models for simulation tasks and trials.

It includes models for representing input messages, predictions, trial information,
and task-level simulation data using Pydantic for validation and type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel


from langchain_core.messages import AIMessage, HumanMessage

class RawPredMsgModel(BaseModel):
    raw: AIMessage
    parsed: BaseModel
    parsing_error: Any|None
class PredSimulationModel(BaseModel):
    ai_msg: Optional[Union[RawPredMsgModel,Any]] = None
    input_stim: list[HumanMessage]
    ai_memory: list | None#list[AIMessage, HumanMessage] | None
    trace_config: str


class InputSimulationModel(BaseModel):
    input_stim: list[HumanMessage]
    system_message: str


class TrialInfoModel(BaseModel):
    trcode: str
    trial: str
    stim: str
    taskname: str
    tasktype: str
    context: str
    context_item: str
    tridx: str
    context_id: int|str
    idx: int|str
    instructions: str
    rating: Any | None = None
    response_postfix_prompt: str|None = None
    query_prefix: str|None = None
    fb: bool|None = None
    hmsg: HumanMessage


class TrialSimulationModel(BaseModel):
    trial_idx: int
    inputs: list[Any]
    system_message: str
    stimulus: str | dict | list
    trcode: str | None
    taskname: str
    tasktype: str
    context_present: bool
    context: Any
    context_item: str
    trid: Any
    chain_type: str
    hmsg: Any
    pred_resp: Any
    pred_dict: Any
    trace_id: str|None = None
    system_message_idx: int
    system_template: str
    tunnel_id: str

class TaskSimulationModel(BaseModel):
    taskdata: list[TrialSimulationModel]



class SimulationModel(BaseModel):
    simdata: list[TaskSimulationModel]


"""

# survey
"trial": item + " \n" + item_reponse_template_postfix,
"hmsg"
# Cog task
"trial": "{ 'trial': '"
    + trials_i["stim"]
    + "'}\n"
+ on_file_task["response_template_postfix"][task_i],
# cog task sub instructions:
"trial": on_file_task[task_i]["instruct"],
"trial": "{ 'trial': '"
                    + trials_i["stim"]
                    + "'}\n"
                    + on_file_task["response_template_postfix"][task_i],
"""
