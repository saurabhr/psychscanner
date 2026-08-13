"""Main CLI for psychscanner."""
from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from . import ExpCard, ExpCardInit, ScannerModel, __version__, to_csv


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True}
)
@click.option(
    "-m",
    "--model",
    type=str,
    help="Use the specified model.",
)
@click.option(
    "-f",
    "--family",
    type=str,
    help="Use the specified family.",

)
@click.option(
    "-p",
    "--parameters",
    type=dict,
    help="Additional parameters for the model.",
    default=None,
)
@click.option(
    "-mem",
    "--memory",
    type=click.Choice(["SingleTurn","Convo"]),
    default="SingleTurn",
    help="Memory type to use.",
)
@click.option(
    "-memk",
    "--memory_k",
    type=int,
    default=-1,
    help="Memory K, number of past interactions to use.",
)
@click.option(
    "-pers",
    "--persona_files",
    type=list[click.Path(exists=False)],
    default=None,
    help="Persona files to use.",
)
@click.option(
    "-t",
    "--task_file",
    type=click.Path(exists=False),
    default=None,
    help="Task file to use.",
)
@click.option(
    "-tc",
    "--task_context",
    type=click.Choice([True, False, None]),
    default=None,
    help="Task context to use.",
)
@click.option(
    "-tus",
    "--tunnel_status",
    default="0",
    type=click.Choice(["0", "1"]),
    help="False/0 (inactive status) or True/1 (active status recommended for final run after testing to save space). To deal with wall time limitation and continue executing from the last iteration. For testing purpose pass inactive.",
)
@click.option(
    "-tuk",
    "--tunnel_k",
    default=-1,
    type=int,
    help="To control after how many trials data is saved. When = 0 then after all trials a simulation is saved. Use this with care.",
)
@click.option(
    "-projname",
    "--projectname",
    default="DEFAULTPROJ",
    type=str,
    help="Project name, could be used to save data in the experiment.",
)
@click.option(
    "-tg",
    "--tags",
    default=[],
    type=list[str],
    help="Tags for added information, could be used to save data in the experiment. Part of only experiment card. Can be used when saving data.",
)
@click.option(
    "-pa",
    "--parser",
    type=str,
    default="0",  # callable should be specified in the function
    help="If other than '0'.Should be defined in the script in the staging.",
)
@click.option(
    "-praw",
    "--parser_raw",
    type=bool,
    default=False,  # callable should be specified in the function
    help="True or false to return the dict with AIMessage with raw output.",
)
@click.option(
    "-pcon",
    "--parser_config",
    type=dict,
    default=None,  # callable should be specified in the function
    help="Dict for parser configration. Default is method=json_schema",
)
@click.option(
    "-pd",
    "--proj_dir",
    type=click.Path(exists=False),
    default=Path.home()
    / "psychscanner",  # callable should be specified in the function
    help="proj_dir",
)
@click.option(
    "-le",
    "--login_env",
    default=None,
    help="File path needed for authentication when using proprietory models. Should be a .env file. For more refer to: https://github.com/theskumar/python-dotenv . Should be kept in .gitignore.",
)
@click.option(
    "-tq",
    "--enabletqdm",
    default=False,
    is_flag=True,
    help="Enable tqdm progress bar.",
)
@click.version_option(__version__, "-v", "--version")
def cli(model: str,
        family: str,
        parameters: dict|None,
        memory: str,
        memory_k: int,
        persona_files: list[click.Path],
        task_file: click.Path|None,
        task_context: bool|None,
        tunnel_status: str,
        tunnel_k: int,
        projectname: str,
        tags: list[str],
        parser: str,
        parser_raw: bool,
        parser_config: dict|None,
        proj_dir: click.Path,
        login_env: click.Path|None,
        enabletqdm: bool) -> None:
    """Run a psychscanner experiment.

    A tool to bridge natural psychology with the artificial.
    """
    if login_env is not None:
        load_dotenv(login_env)

    card_kwargs = {
        "memory": memory,
        "memory_k": memory_k,
        "persona_files": persona_files,
        "task_context": task_context,
        "tunnel_status": tunnel_status,
        "tunnel_k": tunnel_k,
        "projectname": projectname,
        "tags": tags,
        "parser": parser,
        "parser_raw": parser_raw,
        "parser_config": parser_config,
        "proj_dir": Path(proj_dir),
        "enabletqdm": enabletqdm,
    }
    if model is not None:
        card_kwargs["model"] = model
    if family is not None:
        card_kwargs["family"] = family
    if parameters is not None:
        card_kwargs["parameters"] = parameters
    if task_file is not None:
        card_kwargs["task_file"] = task_file

    # No persona files -> nothing to drive "custom" personas from, so run
    # a single no-persona simulation instead (mirrors the README quickstart).
    if persona_files:
        card_kwargs["cogtype"] = "custom"
    else:
        card_kwargs["cogtype"] = "no"
        card_kwargs["nsim"] = 1

    card = ExpCardInit(**card_kwargs)
    scanner = ScannerModel(expcard=ExpCard(card))
    results = scanner.run(progress_bar=enabletqdm)
    df = to_csv(scanner, path=card.proj_dir)
    click.echo(
        f"Ran {len(results)} result batch(es); saved {len(df)} row(s) to {card.proj_dir}"
    )
