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

from .multimodal import resolve_path_block


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
        "template": template,
    }

    return symsg_data


def get_persona_statements(persona_levels:list[dict])->list:

    all_persona = []
    all_p_levels = []
    for p_level in persona_levels:
        if "persona_statements" in p_level:
            all_p_levels += [p_level["persona_statements"]]
        else:
            all_p_levels += [list(p_level.values())]
    all_persona = list(product(*all_p_levels))
    return [" ".join(i) for i in all_persona]

def all_system_msg_prompts(template, instructions, persona_statements):

    all_sys_msgs = []

    for persona in persona_statements:
        sys_msg=template
        if "{system_persona}" in template:
            sys_msg = sys_msg.replace("{system_persona}", persona)

        if "{instructions}" in template:
            sys_msg = sys_msg.replace("{instructions}", instructions)

        all_sys_msgs.append(copy.deepcopy(sys_msg))

    return all_sys_msgs


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

    sys_template_type = expcard.card_in.cogtype
    persona_data = expcard.persona_data
    task = expcard.task_data
    nsim = expcard.card_in.nsim
    chain_type = task["chain_type"]
    persona_statements = None

    instructions = task["on_file"]["instructions"]
    instructions = json.dumps(instructions)
    if sys_template_type == "no":
        sys_msg_template = {
            "TASK CONTEXT": "{instructions}",
        }
        sys_msg_template = json.dumps(sys_msg_template, indent=4)
        persona_statements = [""] * nsim
    elif sys_template_type == "assistant":
        sys_msg_template = {
            "TASK CONTEXT": "{instructions}",
        }
        sys_msg_template = json.dumps(sys_msg_template, indent=4)

        sys_msg_template = (
            "You are a helpful assistant. Perform the task as per the instructions described below.\n\n"
            + sys_msg_template
        )
        persona_statements = [""] * nsim

    elif sys_template_type == "custom":
        sys_msg_template = {
            "You are a helpful assistant with the following individual characteristics": "{system_persona} \n\nPerform the task as per the instructions described below.",
            "TASK CONTEXT": "{instructions}",
        }
        sys_msg_template = json.dumps(sys_msg_template, indent=4)
        persona_statements = get_persona_statements(persona_data)

    system_prompts = all_system_msg_prompts(sys_msg_template,instructions,persona_statements)

    return {
        "system_template": sys_msg_template,
        "system_prompts": system_prompts,
        "chain_type": chain_type,
    }

def gen_stimulus_prompt(
    trstim
):
    stimulus = trstim["stimulus"]
    if isinstance(stimulus, list):
        # Multimodal content: list of standard LangChain content blocks
        # (e.g. {"type": "image", ...} + {"type": "text", ...}). Reachable
        # regardless of context_present/task_instruction so multimodal
        # stimuli work for every task shape, not just context-free ones.
        # A block may carry "path" instead of "base64"/"url" when authored
        # directly in JSON; resolve those here rather than requiring Python.
        content = [
            resolve_path_block(block) if isinstance(block, dict) else block
            for block in stimulus
        ]
        if trstim.get("context_present") and trstim.get("context_item"):
            content = [
                {"type": "text", "text": f"TRIAL_CONTEXT: {trstim['context_item']}"}
            ] + content
        return HumanMessage(content=content)

    trial_dict = {}
    if trstim.get("trcode") == "task_instruction":
        trial_dict["NEXT TASK INSTRUCTION FOR TRIALS"] = stimulus

    elif trstim["context_present"]:
        trial_dict["TRIAL_CONTEXT"] = trstim["context_item"]

        trial_dict["TRIAL"] = stimulus

    else:
        trial_dict = stimulus

    return HumanMessage(json.dumps(trial_dict, indent = 4))

def get_human_feedback_prompt(
    fbmsg: str, context_prefix: str = "", context_postfix: str = ""
) -> HumanMessage:
    """Generate a human feedback prompt message.

    Parameters:
    ----------
    fbmsg : str
        The feedback message content.
    context_prefix : str, optional
        The prefix to add to the feedback message (default is an empty string).
    context_postfix : str, optional
        The postfix to add to the feedback message (default is an empty string).

    Returns:
    -------
    HumanMessage
        The generated human feedback message.
    """
    message_content = context_prefix + fbmsg + context_postfix
    return HumanMessage(message_content)

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
    task_data = expcard.task_data
    exp_trials_raw = task_data["items"]
    trials = []
    for trid, tr in exp_trials_raw.items():
        for trstim in tr:
            trstim["taskname"] = task_data["on_file"]["taskname"]
            trstim["tasktype"] = task_data["on_file"]["tasktype"]
            trstim["context_present"] = task_data["on_file"]["context_present"]
            trstim["context"] = trstim["trcode"].split("_")[0],
            trstim["context_item"] = task_data["on_file"]["contexts"][
                task_data["on_file"]["contexts_id"].index(trstim["trcode"].split("_")[0])
            ]
            trstim["trid"] = trid
            trstim["chain_type"] = task_data["chain_type"]
            trstim["hmsg"] = gen_stimulus_prompt(trstim)
            trials.append(copy.deepcopy(trstim))

    return {
        "trials": trials,
        "chain_type": task_data["chain_type"],
    }
