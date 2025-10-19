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

def create_symsg_data_prompt(expcard,template, inst: str, pop_state: str, persona_role: str) -> dict:
    symsg_data = None

    if isinstance(pop_state, list):
        pop_state = "\n\t".join(pop_state)

    if "rating_inst" in expcard.task_data:
        symsg_data = {
            "symessage": template.format(
                population=pop_state,
                persona=persona_role,
                instructions=inst,
                rating_inst=expcard.task_data["rating_inst"],
            ),
            "population": pop_state,
            "persona": persona_role,
            "instruction": inst,
            "rating_inst": expcard.task_data["rating_inst"],
        }
    else:
        symsg_data = {
        "symessage": template.format(
            population=pop_state, persona=persona_role, instructions=inst
        ),
        "instruction": inst,
        "population": pop_state,
        "persona": persona_role,
    }

    if persona_role == "":
        template = "\t{population}\n\nCONTEXT:\n\t{instructions}\n"

        symsg_data = {
            "symessage": template.format(population=pop_state, instructions=inst),
            "instruction": inst,
            "population": pop_state,
            "persona": persona_role,
        }

    return symsg_data

def all_system_msg_prompts(
    expcard: object,
    template: str,
    populations: list | None = None,
    personas: list | None = None,
    task_instructions: str | None = None,
) -> list[dict]:
    """Generate system message prompts based on experiment card data.

    Parameters:
    ----------
    expcard : object
        The experiment card containing population, persona, and task data.
    template : str
        The template string for formatting system messages.
    populations : list | None, optional
        List of population roles (default is None, which uses expcard data).
    personas : list | None, optional
        List of persona roles (default is None, which uses expcard data).
    task_instructions : str | None, optional
        List of task instructions (default is None, which uses expcard data).

    Returns:
    -------
    list[dict]
        A list of dictionaries containing system message prompts.
    """
    populations = populations or expcard.population_data["pop_roles"]

    personas = personas or expcard.persona_data["persona_roles"]
    task_instructions = task_instructions or expcard.task_data["instructions"]

    populations = [populations] if not isinstance(populations, list) else populations
    personas = [personas] if not isinstance(personas, list) else personas
    task_instructions = (
        [task_instructions]
        if not isinstance(task_instructions, list)
        else task_instructions
    )



    return [
        create_symsg_data_prompt(expcard,template,inst, pop_state, persona_role)
        for inst in task_instructions
        for pop_state in populations
        for persona_role in personas
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
    symsg_inputs = {}  # Initialize with a default value

    population = expcard.population_data["pop_roles"]
    persona = expcard.persona_data["persona_roles"]
    task = expcard.task_data

    if task["tasktype"] == "survey":
        sys_msg_template = (
            "YOU HAVE THE FOLLOWING COGNITIVE AND PERSONALITY CHARACTERISTICS:\n"
            "\tYour cognitive ability is described as follows:\n"
            "\t{population}\n"
            "\tYour first-person description of persona when you describe yourself:\n"
            "\t'{persona}'\n"
            "TASK CONTEXT:\n"
            "\t{instructions}\n"
            "RATING SCALE:\n"
            "\t{rating_inst}\n"
        )
        click.echo(
            f"-----<no cog>----- {expcard.card_in.NOCOG}",
        )

        if expcard.card_in.NOCOG:

            sys_msg_template = (
                "GUIDELINES:\n\t{instructions}\nRATING SCALE:\n\t{rating_inst}\n"
            )

        symsg_inputs = {
            "population": population,
            "persona": persona,
            "instructions": task["instructions"],
            "rating_inst": task["rating_inst"],
        }

    if task["tasktype"] in ["task", "cogtask","taskop"]:
        sys_msg_template = (
            "YOU HAVE THE FOLLOWING COGNITIVE AND PERSONALITY CHARACTERISTICS:\n"
            "\tYour cognitive ability is described as follows:\n"
            "\t{population}\n"
            "\tYour first-person description of persona when you describe yourself:\n"
            "\t'{persona}'\n"
            "\nCONTEXT:\n"
            "\t{instructions}\n"
        )
        symsg_inputs = {
            "population": population,
            "persona": persona,
            "instructions": task["instructions"],
        }


        if expcard.card_in.NOCOG:
            click.echo(f"-----<>----- {expcard.card_in.NOCOG}")

            sys_msg_template = "GUIDELINES:\n\t{instructions}"

    if task["tasktype"] == "sc":
        sys_msg_template = "{instructions}"


    system_prompts = all_system_msg_prompts(expcard, sys_msg_template)

    return {
        "system_template": sys_msg_template,
        "sprompt_inputs": symsg_inputs,
        "system_prompts": system_prompts,
    }


########
# Functions for setting up trial level data
#######
def insert_at_location(dictionary, key, value, index):
    items = list(dictionary.items())
    items.insert(index, (key, value))
    return dict(items)

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
        #print(trial_data.get("tridx"))
        #if trial_data.get("tridx") != "0":
        #    trial_dict = insert_at_location(trial_dict, "PREVIOUS TRIAL FEEDBACK", "Not Available", 0)
    else:
        if "response_postfix_prompt" in trial_data:
            trial_dict["TRIAL INSTRUCTION"] = trial_data["response_postfix_prompt"]
        elif "query_prefix" in trial_data:
            trial_dict["TRIAL INSTRUCTION"] = trial_data["query_prefix"]

        if addcontext:
            trial_dict["TRIAL CONTEXT"] = trial_data["context_item"]

        trial_dict["TRIAL"] = trial_data["stim"]

        if trial_data["tasktype"] in ["task","cogtask","taskop"]:

            trial_dict["TRIAL"] = eval(trial_data["stim"])["stim"]

    #if trial_data["fb"]:
        #if trial_data["trcode"] != "task_instruction":
        #    trial_dict = insert_at_location(trial_dict, "PREVIOUS TRIAL FEEDBACK", "Not Available", 0)

    message_content = json.dumps(trial_dict, allow_nan=True, indent=6)
    trial_data["trial"] = message_content
    if returnai:
        trial_data["hmsg"] = AIMessage(message_content)
    else:
        trial_data["hmsg"] = HumanMessage(message_content)

    return trial_data


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


def get_surveytrials_with_human_msg(
    scantask_i: dict,
    expcard=None,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None,
):
    on_file_task = scantask_i["on_file"]
    trial_items = on_file_task["items"]
    item_reponse_template_postfix = on_file_task["item_reponse_template_postfix"]
    scantask_i[keyname] = []

    for i, (item_code, item) in enumerate(trial_items.items()):
        trial_i = {
            "trcode": item_code,
            "stim": item,
            "taskname": on_file_task["taskname"],
            "tasktype": scantask_i["tasktype"],
            "context": item_code.split("_")[0],
            "context_item": on_file_task["contexts"][
                on_file_task["contexts_id"].index(item_code.split("_")[0])
            ],
            "tridx": item_code.split("_")[1],
            "context_id": on_file_task["contexts_id"].index(item_code.split("_")[0]),
            "idx": i + 1,
            "instructions": scantask_i["instructions"],
            "ratings": scantask_i["rating_inst"],
            "response_postfix_prompt": item_reponse_template_postfix,
            "fb": False,
        }

        if addcontext is None:
            addcontext = expcard.card_in.task_context

        trial_i = gen_trial_prompt(trial_i, addcontext=addcontext)

        scantask_i[keyname].append(trial_i)

    return scantask_i


def get_taskoptrialswith_human_msg(
    scantask_i: dict,
    expcard,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None,
):
    on_file_task = scantask_i["on_file"]
    scantask_i[keyname] = []
    total_trials = 0
    if "transitions" in on_file_task:
        for task_i in on_file_task["transitions"]:
            for idx, trials_i in enumerate(on_file_task[task_i]["trials"]):
                trial_ii = {
                    "trcode": f"{task_i}_{idx + 1}",
                    "stim": str(trials_i),
                    "taskname": on_file_task["taskname"],
                    "tasktype": scantask_i["tasktype"],
                    "context": task_i,
                    "context_item": task_i,
                    "tridx": str(total_trials),
                    "context_id": task_i,
                    "idx": task_i + "_" + str(idx) + "_" + str(total_trials),
                    "instructions": scantask_i["instructions"],
                    "ratings": "",
                    "query_prefix": on_file_task["query_prefix"][task_i],
                    "fb": True,
                }
                total_trials += 1
                if addcontext is None:
                    addcontext = expcard.card_in.task_context
                trial_ii = gen_trial_prompt(trial_ii, addcontext=addcontext)
                scantask_i[keyname].append(trial_ii)
    return scantask_i

def get_tasktrials_with_human_msg(
    scantask_i: dict,
    expcard,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None,
):

    on_file_task = scantask_i["on_file"]
    scantask_i[keyname] = []
    total_trials = 0
    if "transitions" in on_file_task:
        for task_i in on_file_task["transitions"]:
            if "instruct" in on_file_task[task_i]:
                trial_i = {
                    "trcode": "task_instruction",
                    "stim": on_file_task[task_i]["instruct"],
                    "taskname": on_file_task["taskname"],
                    "tasktype": scantask_i["tasktype"],
                    "context": task_i,
                    "context_item": on_file_task[task_i]["instruct"],
                    "tridx": str(total_trials),
                    "context_id": task_i,
                    "idx": task_i + "_" + str(total_trials),
                    "instructions": scantask_i["instructions"],
                    "ratings": "",
                    "response_postfix_prompt": on_file_task["response_template_postfix"][
                        task_i
                    ],
                    "fb": False,
                }


                total_trials += 1

                trial_i = gen_trial_prompt(
                        trial_i, addcontext=False
                    )
                scantask_i[keyname].append(trial_i)
            for idx, trials_i in enumerate(on_file_task[task_i]["trials"]):
                trial_ii = {
                    "trcode": f"{task_i}_{idx + 1}",
                    "stim": str(trials_i),
                    "taskname": on_file_task["taskname"],
                    "tasktype": scantask_i["tasktype"],
                    "context": task_i,
                    "context_item": task_i,
                    "tridx": str(total_trials),
                    "context_id": task_i,
                    "idx": task_i + "_" + str(idx) + "_" + str(total_trials),
                    "instructions": scantask_i["instructions"],
                    "ratings": "",
                    "response_postfix_prompt": on_file_task[
                        "response_template_postfix"
                    ][task_i],
                    "fb": True,
                }
                total_trials += 1
                if addcontext is None:
                    addcontext = expcard.card_in.task_context
                trial_ii = gen_trial_prompt(
                    trial_ii, addcontext=addcontext
                )
                scantask_i[keyname].append(trial_ii)
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
    task_data = expcard.task_data
    exp_survey_task = {}
    if task_data["tasktype"] == "survey":
        exp_survey_task = get_surveytrials_with_human_msg(
            scantask_i=task_data, expcard=expcard
        )
    if task_data["tasktype"] in ["task", "cogtask"]:
        exp_survey_task = get_tasktrials_with_human_msg(
            scantask_i=task_data, expcard=expcard
        )

    if task_data["tasktype"] == "taskop":
        exp_survey_task = get_taskoptrialswith_human_msg(
            scantask_i=task_data, expcard=expcard
        )

    if task_data["tasktype"] == "sc":
        exp_survey_task = get_sc_task_with_hmsg(
            scantask_i = task_data,
            expcard = expcard
        )
    return exp_survey_task



def get_sc_task_with_hmsg(scantask_i: dict,
    expcard,
    keyname: str = "trial_prompts",
    addcontext: bool | None = None):

    on_file_task = scantask_i["on_file"]
    trial_items = on_file_task["items"]
    scantask_i[keyname] = []

    for i, (item_code, item) in enumerate(trial_items.items()):
        trial_i = {
            "trcode": item_code,
            "stim": json.dumps(item),
            "taskname": on_file_task["taskname"],
            "tasktype": scantask_i["tasktype"],
            "context": item_code.split("_")[0],
            "context_item": on_file_task["contexts"][
                on_file_task["contexts_id"].index(item_code.split("_")[0])
            ],
            "tridx": item_code.split("_")[1],
            "context_id": on_file_task["contexts_id"].index(item_code.split("_")[0]),
            "idx": i + 1,
            "instructions": json.dumps(scantask_i["instructions"]),
        }
        if addcontext is None:
            addcontext = expcard.card_in.task_context

        trial_i = gen_sc_trial_prompt(
            trial_i, addcontext=addcontext
        )#gen_trial_prompt(trial_i, addcontext=addcontext)

        scantask_i[keyname].append(trial_i)

    return scantask_i

def gen_sc_trial_prompt(trial_data,addcontext=False):
    # add pre fix and post fix components later
    trial_dict = trial_data["stim"]
    message_content = json.dumps(trial_dict, allow_nan=True, indent=2)
    trial_data["trial"] = message_content
    trial_data["hmsg"] = HumanMessage(message_content)
    return trial_data
