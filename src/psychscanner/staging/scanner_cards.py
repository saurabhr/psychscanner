"""Scanner Cards.

Prepares and indexs diffrent information as hash tables.

Classes for ExpCard, ModelCard, DataCard and their validation.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any, Literal, Callable

import click
from pydantic import BaseModel, Field, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict

from psychscanner import datasets
from psychscanner.datasets import load_datasets
from psychscanner.session_tunnel import SessionTunnel
from psychscanner.datasets.prompts.parser import *

class Settings(BaseSettings):
    """Settings for configuring the application.

    Attributes:
    ----------
    model_config : SettingsConfigDict
        Configuration for environment file and its encoding.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",case_sensitive=True)

class ExpCardInit(BaseModel):
    """Experiment Card for Psychscanner."""

    model: str = Field(
        default="mock-chat-model",
        description="Name of the model to be used. Default is 'mockchatmodel'. Other than default, model and family name should be correctly provided.",
    )
    family: str = Field(
        default="mock-llm",
        description="Name of the family of the model. Default is 'mockllm'. Other than default, modelname and family name should be correctly provided.",
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Parameters for the model. Default is an empty dictionary. Model defined parametes passed as a dictinory of key value pair should be correctly provided. To look at available model parametes look at the model documentation.",
    )
    memory: Literal["SingleTurn", "Convo"] = Field(
        default="SingleTurn",
        description="Memory function to use. Default is single turn chat which is stateless and does not account for past interactions. Every interaction is independent of previous chat. For interaction based memories use other otions.",
    )
    memory_k: int | None = Field(
        default=-1,
        description="Memory k. Default is -1 the memory models run with default settings. In Convo, k stroes last k interactions, in summary k only keeps summary of last k interactions. In ConvoSummaryK, k is the number of recent interactions kept as conversation while the past interactions are summarized",
    )

    persona_files: list[FilePath] | None = Field(
        default=None,
        description="Path to the .json persona related files. If None value is updated to DEFAULT_PER in person_roles as list of strings by ExpCard class. If not default, path to the persona file should be correctly provided. File Should be formated in .json file with 'persona_statements' as key and list of string values. List values from the file stored in key: persona_roles.",
    )
    task_file: FilePath | None = Field(
        default=Path(datasets.__file__).parent/"default_survey.json",
        description="Task to run in the scanner. JSON file format with a psychscanner task structure for survey or cognitive tasks. By default runs a VVIQ quesstionnaire with 16 items. On an item the AI responds with a rating value. The VVIQ-16-items suvey can be found as DEFAULT SURVEY in datasets/datasets.py",
    )
    task_context: Literal[True, False,None] = Field(
        default=None,
        description="For survey task to run in the experiment, context should be in the trial as key. Example VVIQ survey has 8 contexts with 4 items for each. When true, formats each items as context: <item>  situation: <item>.  In cognitive task it needs to be explicitly supplied in the trial items. In other task it should be provided as part of stimulus. Optional, functional for survey/questionnaire intialized in json. These context are to group survey items.",
    )
    tunnel_status: Literal["0", "1"] = Field(
        default="0",
        description="Tunnel status. Default is '0'. If not default, should be correctly provided. 0 for no tunnel and 1 for tunnel. Tunnel is a function to monitor the storage and create checkpoints by creting files in the current working directory. If tunnel is off then the session tunnel related files are stored as DEFAULT_<timestamp>.log in subpackare datasets/no_tunnel_runs/<session tunnel files>.",
    )
    tunnel_k: int | None = Field(
        default=-1,
        description="Tunnel k. Default is -1. If not default, should be correctly provided. Tunnel k is the number of checkpoints to be created after every k trials. When -1 ckeckpoints created after all the trials in the simulation. If k is more than the trials the k is set to -1. If tunnel is off then the session tunnel related files are stored as DEFAULT_<timestamp>.log in subpackare datasets/no_tunnel_runs/<session tunnel files>.",
    )
    projectname: str | None = Field(
        default="DEFAULTPROJ",
        description="Name of the project. Default is 'DEFAULTPROJ'. If not default, should be correctly provided. Project name is used to create a folder in the current working directory to store the session files. files are saved in submodules in datasets/defult_project/<projectname timestamped>. If a folder location is given then the data is saved there and if the folder does not exist then it is created. If the folder location is not given then the data is saved in the current working directory by creading a folder with the project name.",
    )
    tags: list[str] | None = Field(
        default=[],
        description="Tags for the project. Default is an empty list. If not default, should be correctly provided. Tags are used to create a folder in the current working directory to store the session files. files are saved in submodules in datasets/defult_project/<projectname timestamped>. If a folder location is given then the data is saved there and if the folder does not exist then it is created. If the folder location is not given then the data is saved in the current working directory by creading a folder with the project name.",
    )
    parser: str = Field(
        default = "0",
        description = "A callalable pydantic object as string or '0' for no parser. Should be defined in the script in the staging.",
    )
    parser_raw: bool = Field(
        default=False,
        description="Returns Raw Dict with original ai message as one of the key.")
    parser_config: dict|None = Field(
        default=None,
        description="Dict for parser configration. Default is method=json_schema.",
    )
    proj_dir: Path | None = Field(
        default=Path.home() / "psychscanner",
        description="Project directory for saving files.",
    )
    login_env: type[Settings] | None = Field(
        default=None,
        description="path to .env file used to authenticate a chat model from the provider. For more refer to: https://github.com/theskumar/python-dotenv . Should be kept in .gitignore.",
    )
    enabletqdm: Literal[False, True] = Field(
        default=False,
        description="Enable tqdm progress bar for simulations.",
    )

    trial_parsers: list[Any]|None = Field(default=None,
                                          description="Used when there is 'trial' chain type and if present in the task json")
    persona_data: list | None = Field(
        default=None,
        description="Persona data initialized after the reading the --persona .json file.",
    )
    task_data: dict | None = Field(
        default=None,
        description="Task data initialized after the reading the --task .json file.",
    )
    session_tunnel: object | None = Field(
        default=None,
        description="Session tunnel object based on tunnel_status parameter",
    )
    cogtype: Literal["assistant", "custom", "no"] = Field(
        default="custom",
        description="If passed as True then the cognitive statements in the prompt are ignored.",
    )
    nsim: int|None = Field(
        default = None,
        description= "Number of simulations when no persona roles are to be used by using NOCOG option."
    )
    chain_type: Literal["item","trial","task"]|None = Field(
        default=None,
        description="'item' is for when only one stimulus is in the trial. 'trial' is for when there is multiple stimulus in a trial. 'task' is for previous trial memory. if given the overrides the 'chain_type' parameter in the task json file otherwise used from the json file."    )

    feedback: Literal["0", "1"] = Field(
        default="0",
        description="Feedback on trials. Trial prompt will have a key called fb as true/false on every trial trompt, if fb key not present in trial prompt dictionary and given 1 for fbactive than the user settings is overwritten to no feedback. if activefb = 1 and fb key in true then the fb is should be explicilty coded or else no feedback is given. Feedback function is stored in key injectfun",
    )  # defult is False, else

    feedback_fn: Callable | None = Field(
        default=None,
        description="Should be provided when feedback is '1' otherwise resets Feedback to '0'. Feedback function to use. Default is None. If not default, should be correctly provided. Feedback function is used to provide feedback on the trials. The feedback function should be a callable object that takes the trial data as input and returns the feedback as output. The feedback function should be defined in the script in the staging.",
    )  # defult is False, else


class ExpCard:
    """Dynamic class for creating an experiment card.

    This class is used to create an experiment card with various parameters
    and settings for a PsychScanner experiment.

    Attributes:
    ----------
    exp_card : ExpCardInit
        An instance of ExpCardInit containing the experiment card details.
    exp_card_dict : dict
        A dictionary representation of the experiment card.

    Methods:
    -------
    __init__(**kwargs)
        Initializes the experiment card with the provided parameters.
    """

    def __init__(self, cls: type[ExpCardInit] | None = None, **kwargs) -> None:
        """Initialize an experiment card.

        Parameters:
        ----------
        cls : type[ExpCardInit], optional
            The class type for the experiment card initialization. Defaults to ExpCardInit.
        **kwargs : dict
            Additional keyword arguments to initialize the experiment card. Look at Experiment Card Init docimentation for data fields.
        """
        if cls is not None:
            self.cls = cls
        else:
            self.cls = ExpCardInit()
            click.echo("No input provided. Using default values.")

        if kwargs:
            self.input = kwargs
            self.card_in = ExpCardInit(**kwargs)
        else:
            self.card_in = self.cls


        if self.card_in.cogtype == "custom":
            self.persona_data = load_datasets.get_persona_data(self.card_in)
        elif self.card_in.cogtype in {"assistant", "no"}:
            self.persona_data = None
            if self.card_in.nsim is None:
                self.card_in.nsim = 1

        self.task_data = load_datasets.get_task_data(self.card_in)
        self.data_root_dir = (
            self.card_in.proj_dir
            / self.card_in.projectname
            / self.task_data["taskname"]
            / f"{self.card_in.family}_{self.card_in.model}_{self.card_in.memory}"
        )
        if not self.data_root_dir.exists():
            self.data_root_dir.mkdir(parents=True, exist_ok=True)
        self.session_tunnel = SessionTunnel(
            tunnel_status=self.card_in.tunnel_status, project_name=self.card_in.projectname,tunnel_dir=self.data_root_dir
        )

        if self.card_in.parser_config is None:
            self.card_in.parser_config = {"method": "json_schema"}

        if self.card_in.parser == "0":
            self.parser = self.card_in.parser

        elif self.card_in.parser == "1":
            self.parser = self.task_data["parser"]
            if isinstance(self.card_in.parser, str):
                self.parser = eval(self.parser) # try to find its alternative.  # noqa: S307
                if not issubclass(self.parser, BaseModel):
                    msg = "Not valid parser provided."
                    raise TypeError(msg)
        elif self.card_in.parser == "dynamic":
            self.parser = "dynamic"
        else:
            self.parser = self.card_in.parser
            if not issubclass(self.parser, BaseModel):
                msg = "Not valid parser provided."
                raise TypeError(msg)

        click.echo("----<PROJECT AND DATA ROOT DIRECTORY>----")
        click.echo(f"\tProject root dir: {self.card_in.proj_dir}")
        click.echo(f"\tSimulation data root dir: {self.data_root_dir}")
        click.echo("----<>----")
