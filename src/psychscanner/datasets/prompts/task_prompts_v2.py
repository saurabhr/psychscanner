"""This module provides functions for generating prompts.

It genrates system messages, trial prompts,and handling task-related data for experiments.
It includes utilities for creating human and AI messages based on experiment card data and task configurations.
"""

from __future__ import annotations

from typing import Any
import json
import click
from langchain_core.messages import AIMessage, HumanMessage
import numpy as np
from itertools import product
import copy

def create_symsg_data_prompt(template, inst: str, persona_role: str) -> dict:


    sys_msg = template
    if "{system_persona}" in template:
        sys_msg = sys_msg.replace("{system_persona}", persona_role)

    if "{instructions}" in template:
        sys_msg = sys_msg.replace("{instructions}", inst)

    symsg_data = {
        "symessage": sys_msg,
        "system_persona": persona_role,
        "instruction": inst,
        "template": template
    }

    return symsg_data


def all_system_msg_prompts(
    expcard: object,
    template: str,
    task_instructions: str | None = None,
    sys_msg_type: str = "custom",
) -> dict:
    """Generate system message prompts based on experiment card data.

    Parameters:
    ----------
    expcard : object
        The experiment card containing population, persona, and task data.
    template : str
        The template string for formatting system messages.
    task_instructions : str | None, optional
        List of task instructions (default is None, which uses expcard data).
    sys_msg_type

    Returns:
    -------
    list[dict]
        A list of dictionaries containing system message prompts.
    """
    all_persona_items = None
    all_persona_items_joined = None
    task_instructions = None
    if sys_msg_type == "custom":
        all_persona_items_grouped = []
        for persona_level in expcard.persona_filedata:
            if "persona_statements" in persona_level:
                persona_level_items = persona_level["persona_statements"]
                all_persona_items_grouped.append(persona_level_items)
            else:
                all_persona_items_grouped.append(list(persona_level.values()))

        all_persona_items = list(product(*all_persona_items_grouped))
        all_persona_items_joined = ["\nYour life and personality is described as follows:\n".join(persona) for persona in all_persona_items]

    else:
        all_persona_items_joined = [""]*expcard.card_in.NSIM

    if expcard.task_filedata["trial_chain"] == "instructions":
        task_instructions = expcard.task_filedata["chain_instructions"]["definition"]
        task_instructions = task_instructions.append("For each task, you will be given a set of instructions to follow.")
        task_instructions = task_instructions.append(
            "You will be presented with the instruction for consecutive tasks in a sequence after each response made by you. Perform the task as per the given instructions."
        )
        task_instructions = "\n\n".join(task_instructions)
    else:
        task_instructions = json.dumps(expcard.task_filedata["instructions"])


    return [
        create_symsg_data_prompt(template, task_instructions, persona_role)
        for persona_role in all_persona_items_joined
    ]


def gen_symsg_promptdata(expcard: Any) -> dict:
    """Generate system message prompt data based on the experiment card.

    Parameters:
    ----------
    expcard : object
        The experiment card containing population, persona, and task data.


    Returns:
    -------
    dict
        A dictionary containing the system message template and inputs.
    """
    sys_msg_template = None  # Initialize with a default value
    #symsg_inputs = {}  # Initialize with a default value
    #persona_filedata = expcard.persona_filedata

    # population = expcard.population_data["pop_roles"]
    # persona = expcard.persona_data["persona_roles"]

    task = expcard.task_filedata
    sys_msg_type = None

    if expcard.card_in.NOCOG == True:
        click.echo(
            f"-----<no cog>----- {expcard.card_in.NOCOG}",
        )
        sys_msg_template = {
            "TASK CONTEXT": "{instructions}",
        }
        sys_msg_template = json.dumps(sys_msg_template, indent=4)

        sys_msg_type = "nocog"
    elif expcard.card_in.NOCOG == "assistant":
        sys_msg_template = {
                "TASK CONTEXT": "{instructions}",
            }
        sys_msg_template = json.dumps(sys_msg_template, indent=4)

        sys_msg_template = (
            "You are a helpful assistant. Perform the task as per the instructions described below.\n\n"
            + sys_msg_template
        )
        sys_msg_type = "assistant"

    elif task["tasktype"] in ["survey", "sc"]:
        sys_msg_template = {
            "You are a helpful assistant with the following individual characteristics": "{system_persona} \n\nPerform the task as per the instructions described below.",
            "TASK CONTEXT": "{instructions}"
        }
        sys_msg_type = "custom"
        sys_msg_template = json.dumps(sys_msg_template, indent=4)
    system_prompts = all_system_msg_prompts(
        expcard, sys_msg_template, sys_msg_type=sys_msg_type
    )

    return {
        "system_template": sys_msg_template,
        "system_prompts": system_prompts,
        "trial_chain": task["trial_chain"]
    }

def gen_trial_prompt(
    trial_data: dict, *, addcontext: bool = False, returnai: bool = False
) -> HumanMessage | AIMessage:
    """Generate a trial prompt.

    Parameters:
    ----------
    trial : dict
        The trial data containing task type, context, and trial information.
    addcontext : bool, optional
        Whether to add context to the prompt (default is False).
    returnai : bool, optional
        Whether to return an AIMessage instead of a HumanMessage (default is False).

    Returns:
    -------
    HumanMessage | AIMessage
        The generated trial prompt message.
    """
    trial_dict = {}

    if trial_data.get("trcode") == "task_instruction":
        trial_dict["NEXT TASK INSTRUCTION FOR TRIALS"] = trial_data["stim"]

    else:

        if addcontext:
            trial_dict["TRIAL_CONTEXT"] = trial_data["context_item"]

        trial_dict["TRIAL"] = trial_data["stim"]

        if trial_data["tasktype"] in ["task", "cogtask", "taskop"]:
            trial_dict["TRIAL"] = eval(trial_data["stim"])["stim"]


    message_content = json.dumps(trial_dict, allow_nan=True, indent=4)

    if trial_data["trial_chain"] == "instructions":
        chain_instructions = trial_data["instructions"]["instructions"][
            "TASK_INSTRUCTIONS"
        ]
        prompt_chain_instructions = []
        for i,j in chain_instructions.items():
            prompt_chain_instructions.append(j.dumps({i:j}))

        message_content = prompt_chain_instructions[0] +"\n\n\n\n"+ [message_content]
        all_trial_i_prompts = [message_content] + prompt_chain_instructions[1:]

        if returnai:
            trial_data["hmsg"] = [AIMessage(i) for i in all_trial_i_prompts]
        else:
            trial_data["hmsg"] = [HumanMessage(i) for i in all_trial_i_prompts]

    else:

        trial_data["trial"] = message_content
        if returnai:
            trial_data["hmsg"] = AIMessage(message_content)
        else:
            trial_data["hmsg"] = HumanMessage(message_content)

    return trial_data



def get_surveytrials_with_human_msg(
    scantask_i: dict,
    expcard=None,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None,
):
    trial_items = scantask_i["items"]
    scantask_i[keyname] = []

    trial_chain = scantask_i.get("trial_chain", None)

    for i, (item_code, item) in enumerate(trial_items.items()):
        trial_i = {
            "trcode": item_code,
            "stim": item,
            "taskname": scantask_i["taskname"],
            "tasktype": scantask_i["tasktype"],
            "context": item_code.split("_")[0],
            "context_item": scantask_i["contexts"][
                scantask_i["contexts_id"].index(item_code.split("_")[0])
            ],
            "tridx": item_code.split("_")[1],
            "context_id": scantask_i["contexts_id"].index(item_code.split("_")[0]),
            "idx": i + 1,
            "trial_chain": trial_chain,
            "fb": False,
            "instructions": scantask_i["instructions"],

        }

        if addcontext is None:
            addcontext = scantask_i["context_present"]

        trial_i = gen_trial_prompt(trial_i, addcontext=addcontext)
        scantask_i[keyname].append(trial_i)

    return scantask_i

def gen_sc_trial_prompt(trial_data, addcontext=False,returnai=False):
    # add pre fix and post fix components later

    trial_dict = trial_data["stim"]
    message_content = json.dumps(trial_dict, allow_nan=True, indent=4)

    if trial_data["trial_chain"] == "instructions":
        chain_instructions = trial_data["instructions"]["instructions"][
            "TASK_INSTRUCTIONS"
        ]
        prompt_chain_instructions = []
        for i, j in chain_instructions.items():
            prompt_chain_instructions.append(j.dumps({i: j}))

        message_content = prompt_chain_instructions[0] + "\n\n\n\n" + [message_content]
        all_trial_i_prompts = [message_content] + prompt_chain_instructions[1:]

        if returnai:
            trial_data["hmsg"] = [AIMessage(i) for i in all_trial_i_prompts]
        else:
            trial_data["hmsg"] = [HumanMessage(i) for i in all_trial_i_prompts]

    else:
        trial_data["trial"] = message_content
        if returnai:
            trial_data["hmsg"] = AIMessage(message_content)
        else:
            trial_data["hmsg"] = HumanMessage(message_content)
    return trial_data

def get_sc_task_with_hmsg(
    scantask_i: dict,
    expcard,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None,
):
    trial_items = scantask_i["items"]
    scantask_i[keyname] = []
    trial_chain = scantask_i.get("trial_chain", None)

    for i, (item_code, item) in enumerate(trial_items.items()):
        trial_i = {
            "trcode": item_code,
            "stim": json.dumps(item),
            "taskname": scantask_i["taskname"],
            "tasktype": scantask_i["tasktype"],
            "context": item_code.split("_")[0],
            "context_item": scantask_i["contexts"][
                scantask_i["contexts_id"].index(item_code.split("_")[0])
            ],
            "tridx": item_code.split("_")[1],
            "context_id": scantask_i["contexts_id"].index(item_code.split("_")[0]),
            "idx": i + 1,
            "trial_chain": trial_chain,
            "instructions": scantask_i["instructions"],
        }
        if addcontext is None:
            addcontext = scantask_i["context_present"]

        trial_i = gen_sc_trial_prompt(
            trial_data = trial_i, addcontext=addcontext
        )  # gen_trial_prompt(trial_i, addcontext=addcontext)

        scantask_i[keyname].append(trial_i)
    return scantask_i


def gen_trial_promptdata(expcard: Any) -> dict:
    """Generate trial prompt data based on the experiment card.

    Parameters:
    ----------
    expcard : Any
        The experiment card containing task data and context information.

    Returns:
    -------
    dict
        A dictionary containing trial prompt data for the given task type.
    """
    task_data = expcard.task_filedata
    exp_survey_task = {}
    if task_data["tasktype"] == "survey":
        exp_survey_task = get_surveytrials_with_human_msg(
            scantask_i=task_data, expcard=expcard
        )
    if task_data["tasktype"] == "sc":
        exp_survey_task = get_sc_task_with_hmsg(
            scantask_i = task_data,
            expcard = expcard
        )

    return exp_survey_task
