import json
from pathlib import Path
import click

DEFAULT_DATA_PATH = Path(__file__).parent


def open_json(file_dir: str) -> dict:
    """Load and return JSON data from the specified file path.

    Parameters:
    ----------
    persona_dir : str
        The file path to the JSON file.

    Returns:
    -------
    dict
        The JSON data loaded as a dictionary.
    """
    with Path(file_dir).open() as f:
        return json.load(f)


def get_persona_data(expcard):

    if expcard.persona_files is not None:
        persona_files = expcard.persona_files
        if not isinstance(persona_files, list):
            persona_files = [persona_files]
        return [open_json(file) for file in persona_files]
    return None

def get_task_data(expcard) -> dict:
    """Retrieve task data based on the provided experiment card.

    Parameters:
    ----------
    expcard : object
        An object containing experiment details, including task information.

    Returns:
    -------
    dict
        A dictionary containing task details such as task type and number of trials.
    """
    taskinfo = expcard.task_file
    task_initial = open_json(taskinfo) if isinstance(taskinfo, str) or isinstance(taskinfo, Path) else taskinfo

    task = {}
    task["on_file"] = task_initial

    task["tasktype"] = task_initial["tasktype"]
    task["taskname"] = task_initial["taskname"]

    task["context_present"] = task_initial["context_present"]
    if expcard.task_context is not None:
        task["context_present"] = expcard.task_context
    task["parser"] = task_initial["parser"]
    task["chain_type"] = task_initial.get("chain_type", None)
    if expcard.chain_type is not None:
        task["chain_type"] = expcard.chain_type
    task["items"] = task_initial.get("items", None)
    task["trial_parsers"] = task_initial.get("trial_parsers",None)  # trial_parsers
    return task
