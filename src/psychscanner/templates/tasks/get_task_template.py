import json
from pathlib import Path
import click

def get_task_template(ttype=None):

    tasktypes = ["sc"]

    ttemplate = {
        "tasktype": "sc",
        "taskname": "",
        "instructions": "",
        "contexts": "",
        "contexts_id": "",
        "context_present": "",
        "items": {},
        "chain_type": "",
        "trial_parsers": [], # in order of the subtasks
        "parser":"",
        "postfix":"",
        "prefix":""

    }

    if ttype is None:
        return ttemplate

    if ttype in tasktypes:
        with Path("sc.json").open(encoding="utf-8") as f:
            ttemplate = json.load(f)
    else:
        click.echo("Task type not found in the template collection. Returning default task template.")
    return ttemplate
