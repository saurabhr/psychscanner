"""Scanner Cards.

Prepares and indexs diffrent information as hash tables.

Classes for ExpCard, ModelCard, DataCard and their validation.
"""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any, Literal, Callable

import click
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from psychscanner import datasets
from psychscanner.datasets import load_datasets
from psychscanner.session_tunnel import SessionTunnel
from psychscanner.parsers import get_parser, resolve_parser, PARSER_REGISTRY

# Sentinel — distinguishes "caller did not pass" from "caller passed None"
_UNSET: object = object()

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(
        default="mock-chat-model",
        description="Name of the model to be used. Default is 'mockchatmodel'. Other than default, model and family name should be correctly provided.",
    )
    family: str = Field(
        default="mock-llm",
        description="Name of the family of the model. Default is 'mockllm'. Other than default, modelname and family name should be correctly provided.",
    )
    parameters: dict | None = Field(
        default=None,
        description="Parameters for the model. Default is an empty dictionary. Model defined parametes passed as a dictinory of key value pair should be correctly provided. To look at available model parametes look at the model documentation.",
    )
    memory: Literal["SingleTurn", "Convo"] = Field(
        default="SingleTurn",
        description="Memory function to use. Default is single turn chat which is stateless and does not account for past interactions. Every interaction is independent of previous chat. For interaction based memories use other otions.",
    )
    memory_k: int | None = Field(
        default=-1,
        description=(
            "Max number of messages kept in full in the conversation context. "
            "Default -1 keeps unlimited history. "
            "When set to N, only the last N messages are passed to the model."
        ),
    )
    summary_k: int | None = Field(
        default=0,
        description=(
            "Summarization batch size. Only active in Convo memory mode with memory_k set. "
            "0 = no summarization, older messages beyond memory_k are simply dropped. "
            "N = when overflow messages reach N, they are summarized into a rolling summary "
            "that is prepended to the system message so context is preserved."
        ),
    )

    persona_files: list[FilePath] | None = Field(
        default=None,
        description="Path to the .json persona related files. If None value is updated to DEFAULT_PER in person_roles as list of strings by ExpCard class. If not default, path to the persona file should be correctly provided. File Should be formated in .json file with 'persona_statements' as key and list of string values. List values from the file stored in key: persona_roles.",
    )
    task_file: dict | FilePath | None = Field(
        default=Path(datasets.__file__).parent/"prompts"/"defaults"/"default_survey.json",
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
    parser: str | type[BaseModel] | Callable | None = Field(
        default=None,
        description=(
            "Structured output parser. Accepts: "
            "None or '0' (no parser); "
            "'1' (resolve by name from task JSON 'parser' field); "
            "a registered parser name string e.g. 'DefaultLiteralVivid15'; "
            "a BaseModel subclass passed directly; "
            "a callable for per-trial dispatch e.g. "
            "lambda trcode: ParserA if 'test' in trcode else ParserB."
        ),
    )
    parser_raw: bool = Field(
        default=False,
        description="Returns Raw Dict with original ai message as one of the key.")
    parser_config: dict|None = Field(
        default=None,
        description="Dict for parser configration. Default is method=json_schema.",
    )
    tools: list[Any] | None = Field(
        default=None,
        description=(
            "LangChain tools to bind to the chat model for every trial. "
            "Accepts @tool-decorated callables or BaseTool instances. "
            "Bound via model.bind_tools(tools) per invocation. Combining "
            "tools with a structured-output parser on the same trial is "
            "provider-dependent (both use the tool-calling protocol)."
        ),
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

    feedback: bool = Field(
        default=False,
        description=(
            "Enable trial-level feedback. Set True (or '1' for backward compatibility) to activate. "
            "When enabled, feedback_fn must also be provided. "
            "Each trial may include an 'fb' key (True/False) to opt individual trials in or out of feedback."
        ),
    )

    @field_validator("feedback", mode="before")
    @classmethod
    def _coerce_feedback(cls, v):
        if v in ("0", False, 0, None, ""):
            return False
        if v in ("1", True, 1):
            return True
        raise ValueError(f"feedback must be a bool or '0'/'1', got {v!r}")

    feedback_fn: Callable | None = Field(
        default=None,
        description=(
            "A FeedbackBase subclass (the class itself, not an instance). "
            "Required when feedback=True. psychscanner instantiates it once per "
            "participant simulation so cross-trial state can live safely in self. "
            "Must implement on_response(trial, response) -> str | None."
        ),
    )

    next_trial: bool = Field(
        default=False,
        description=(
            "Enable conditional intermediate trials. Set True (or '1' for backward "
            "compatibility) to activate. When enabled, next_trial_fn must also be provided. "
            "After each trial, the handler may insert a new trial before the task card's "
            "next one, e.g. for adaptive/staircase designs."
        ),
    )

    @field_validator("next_trial", mode="before")
    @classmethod
    def _coerce_next_trial(cls, v):
        if v in ("0", False, 0, None, ""):
            return False
        if v in ("1", True, 1):
            return True
        raise ValueError(f"next_trial must be a bool or '0'/'1', got {v!r}")

    next_trial_fn: Callable | None = Field(
        default=None,
        description=(
            "A NextTrialBase subclass (the class itself, not an instance). "
            "Required when next_trial=True. psychscanner instantiates it once per "
            "participant simulation so cross-trial state can live safely in self. "
            "Must implement next_trial(trial, response) -> dict | None."
        ),
    )


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

        try:
            self.parser = resolve_parser(
                self.card_in.parser,
                task_parser_name=self.task_data.get("parser"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

        self.tools = self.card_in.tools

        if self.card_in.feedback and self.card_in.feedback_fn is None:
            raise ValueError(
                "feedback=True requires feedback_fn to be set. "
                "Provide a FeedbackBase subclass (not an instance) as feedback_fn."
            )

        if self.card_in.next_trial and self.card_in.next_trial_fn is None:
            raise ValueError(
                "next_trial=True requires next_trial_fn to be set. "
                "Provide a NextTrialBase subclass (not an instance) as next_trial_fn."
            )

        click.echo("----<PROJECT AND DATA ROOT DIRECTORY>----")
        click.echo(f"\tProject root dir: {self.card_in.proj_dir}")
        click.echo(f"\tSimulation data root dir: {self.data_root_dir}")
        click.echo("----<>----")


# ── Scalar fields that round-trip cleanly through JSON ───────────────────────
_SCALAR_FIELDS: tuple[str, ...] = (
    "model", "family", "parameters", "memory", "memory_k", "summary_k",
    "task_context", "tunnel_status", "tunnel_k", "projectname", "tags",
    "parser_raw", "parser_config", "cogtype", "nsim", "chain_type",
    "feedback", "next_trial", "enabletqdm",
)


def save_expcard(
    card_in: ExpCardInit,
    path: str | Path | None = None,
) -> dict:
    """Serialize an ``ExpCardInit`` to a portable, JSON-safe dict.

    The returned dict (and the file written when *path* is given) contains
    everything needed to reproduce the experiment on a different machine:

    - **task_file** and **persona_files** are embedded as inline JSON, so the
      recipient does not need local copies of those files.
    - Registered parsers are stored by name string (re-imported via the
      registry on load).
    - Custom parser classes and callable dispatch functions are stored as
      ``module + qualname`` so they can be re-imported if the module is
      installed on the recipient's machine.
    - ``feedback_fn`` and ``next_trial_fn`` are stored the same way.

    Machine-specific fields (``proj_dir``, ``login_env``) and runtime-only
    fields (``session_tunnel``, ``persona_data``, ``task_data``,
    ``trial_parsers``) are intentionally omitted.

    Parameters
    ----------
    card_in:
        The experiment card to serialize.
    path:
        Optional file path. When given, the dict is also written as JSON.

    Returns
    -------
    dict
        Portable representation. Pass to :func:`load_expcard` to reconstruct.

    Examples
    --------
    >>> from psychscanner import ExpCardInit, save_expcard, load_expcard
    >>> card = ExpCardInit(model="smollm2:360m-instruct-fp16", family="ollama")
    >>> save_expcard(card, "my_experiment.json")
    >>> card2 = load_expcard("my_experiment.json")
    """
    from datetime import datetime, timezone
    import psychscanner as _psy

    d: dict[str, Any] = {
        "_psychscanner_version": _psy.__version__,
        "_saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── scalar / simple fields ────────────────────────────────────────────────
    for field in _SCALAR_FIELDS:
        d[field] = getattr(card_in, field)

    # ── task_file — embed inline ──────────────────────────────────────────────
    tf = card_in.task_file
    if tf is None:
        d["task_file"] = None
    elif isinstance(tf, (str, Path)):
        d["task_file"] = json.loads(Path(tf).read_text())
    else:
        d["task_file"] = tf  # already a dict

    # ── persona_files — embed inline ──────────────────────────────────────────
    pf = card_in.persona_files
    if pf is None:
        d["_persona_embedded"] = None
    else:
        pf_list = pf if isinstance(pf, list) else [pf]
        d["_persona_embedded"] = [
            json.loads(Path(p).read_text()) for p in pf_list
        ]

    # ── parser ────────────────────────────────────────────────────────────────
    parser = card_in.parser
    if parser is None or isinstance(parser, str):
        # None / "0" / "1" / any registered name string
        d["parser"] = parser
    elif isinstance(parser, type) and issubclass(parser, BaseModel):
        # Registered class → name string; custom class → module+qualname dict
        if parser.__name__ in PARSER_REGISTRY:
            d["parser"] = parser.__name__
        else:
            d["parser"] = {
                "__class__": {
                    "module": parser.__module__,
                    "qualname": parser.__qualname__,
                }
            }
    elif callable(parser):
        d["parser"] = {
            "__callable__": {
                "module": getattr(parser, "__module__", None),
                "qualname": getattr(parser, "__qualname__", None),
            }
        }
    else:
        d["parser"] = None

    # ── feedback_fn ───────────────────────────────────────────────────────────
    fb_fn = card_in.feedback_fn
    if fb_fn is None:
        d["feedback_fn"] = None
    else:
        d["feedback_fn"] = {
            "module": getattr(fb_fn, "__module__", None),
            "qualname": getattr(fb_fn, "__qualname__", None),
        }

    # ── next_trial_fn ─────────────────────────────────────────────────────────
    nt_fn = card_in.next_trial_fn
    if nt_fn is None:
        d["next_trial_fn"] = None
    else:
        d["next_trial_fn"] = {
            "module": getattr(nt_fn, "__module__", None),
            "qualname": getattr(nt_fn, "__qualname__", None),
        }

    # ── tools ─────────────────────────────────────────────────────────────────
    # @tool-decorated callables are BaseTool instances wrapping the original
    # function in `.func`; plain functions passed directly have __module__/
    # __qualname__ on themselves. Either way we recover the importable source.
    tools = card_in.tools
    if not tools:
        d["tools"] = None
    else:
        d["tools"] = [
            {
                "module": getattr(getattr(t, "func", t), "__module__", None),
                "qualname": getattr(getattr(t, "func", t), "__qualname__", None),
            }
            for t in tools
        ]

    if path is not None:
        Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))

    return d


def load_expcard(
    source: str | Path | dict,
    *,
    proj_dir: str | Path | None = None,
    parser: type[BaseModel] | Callable | None = _UNSET,
    feedback_fn: Callable | None = _UNSET,
    next_trial_fn: Callable | None = _UNSET,
    tools: list[Any] | None = _UNSET,
) -> ExpCardInit:
    """Reconstruct an ``ExpCardInit`` from a dict or JSON file created by
    :func:`save_expcard`.

    Parameters
    ----------
    source:
        File path (str or Path) or the dict returned by :func:`save_expcard`.
    proj_dir:
        Output root directory override.  If *None*, defaults to
        ``~/psychscanner``.  The original ``proj_dir`` is not stored in the
        portable file because it is machine-specific.
    parser:
        Override the saved parser.  Required when the original parser was a
        lambda or a custom class that is not importable on the current machine.
    feedback_fn:
        Override the saved feedback handler class.  Required when the original
        handler is not importable on the current machine.
    next_trial_fn:
        Override the saved conditional-next-trial handler class.  Required
        when the original handler is not importable on the current machine.
    tools:
        Override the saved tools list.  Required when a tool is a lambda,
        closure, or a stateful ``BaseTool`` instance that is not importable
        on the current machine.

    Returns
    -------
    ExpCardInit
        Ready to pass directly to ``ExpCard``.

    Raises
    ------
    ImportError
        If a custom parser class or feedback handler cannot be re-imported and
        no override was supplied.

    Examples
    --------
    >>> card = load_expcard("my_experiment.json")
    >>> card = load_expcard("my_experiment.json", proj_dir="/data/results")
    >>> card = load_expcard("my_experiment.json", parser=my_dispatch_fn)
    """
    import importlib

    if isinstance(source, (str, Path)):
        d: dict = json.loads(Path(source).read_text())
    else:
        d = dict(source)

    kwargs: dict[str, Any] = {f: d[f] for f in _SCALAR_FIELDS if f in d}

    # ── proj_dir ──────────────────────────────────────────────────────────────
    kwargs["proj_dir"] = Path(proj_dir) if proj_dir is not None else Path.home() / "psychscanner"

    # ── task_file — dict accepted directly by get_task_data ──────────────────
    task_file = d.get("task_file")
    if task_file is not None:
        kwargs["task_file"] = task_file

    # ── persona_files — write embedded dicts to a persistent sub-directory ───
    persona_embedded = d.get("_persona_embedded")
    if persona_embedded is not None:
        portables_dir = kwargs["proj_dir"] / ".portable_personas"
        portables_dir.mkdir(parents=True, exist_ok=True)
        # Use the original save timestamp so repeated loads reuse the same files
        ts = d.get("_saved_at", "").replace(":", "-")[:19]
        persona_paths: list[Path] = []
        for i, pdata in enumerate(persona_embedded):
            p = portables_dir / f"persona_{ts}_{i}.json"
            p.write_text(json.dumps(pdata, indent=2, ensure_ascii=False))
            persona_paths.append(p)
        kwargs["persona_files"] = persona_paths

    # ── parser ────────────────────────────────────────────────────────────────
    if parser is not _UNSET:
        kwargs["parser"] = parser
    else:
        saved = d.get("parser")
        if saved is None or isinstance(saved, str):
            # None / "0" / "1" / registered name — resolve_parser handles it
            kwargs["parser"] = saved
        elif isinstance(saved, dict):
            key = "__class__" if "__class__" in saved else "__callable__"
            info = saved[key]
            kind = "parser class" if key == "__class__" else "parser callable"
            try:
                mod = importlib.import_module(info["module"])
                obj: Any = mod
                for part in info["qualname"].split("."):
                    obj = getattr(obj, part)
                kwargs["parser"] = obj
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    f"Cannot import {kind} '{info['module']}.{info['qualname']}': {exc}.\n"
                    f"Pass it explicitly: load_expcard(..., parser=<your_{kind}>)"
                ) from exc

    # ── feedback_fn ───────────────────────────────────────────────────────────
    if feedback_fn is not _UNSET:
        kwargs["feedback_fn"] = feedback_fn
    else:
        saved_fn = d.get("feedback_fn")
        if saved_fn is not None:
            try:
                mod = importlib.import_module(saved_fn["module"])
                obj = mod
                for part in saved_fn["qualname"].split("."):
                    obj = getattr(obj, part)
                kwargs["feedback_fn"] = obj
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    f"Cannot import feedback handler "
                    f"'{saved_fn['module']}.{saved_fn['qualname']}': {exc}.\n"
                    f"Pass it explicitly: load_expcard(..., feedback_fn=MyHandler)"
                ) from exc

    # ── next_trial_fn ────────────────────────────────────────────────────────
    if next_trial_fn is not _UNSET:
        kwargs["next_trial_fn"] = next_trial_fn
    else:
        saved_nt_fn = d.get("next_trial_fn")
        if saved_nt_fn is not None:
            try:
                mod = importlib.import_module(saved_nt_fn["module"])
                obj = mod
                for part in saved_nt_fn["qualname"].split("."):
                    obj = getattr(obj, part)
                kwargs["next_trial_fn"] = obj
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    f"Cannot import next-trial handler "
                    f"'{saved_nt_fn['module']}.{saved_nt_fn['qualname']}': {exc}.\n"
                    f"Pass it explicitly: load_expcard(..., next_trial_fn=MyHandler)"
                ) from exc

    # ── tools ─────────────────────────────────────────────────────────────────
    if tools is not _UNSET:
        kwargs["tools"] = tools
    else:
        saved_tools = d.get("tools")
        if saved_tools is not None:
            resolved_tools = []
            for saved_tool in saved_tools:
                try:
                    mod = importlib.import_module(saved_tool["module"])
                    obj = mod
                    for part in saved_tool["qualname"].split("."):
                        obj = getattr(obj, part)
                    resolved_tools.append(obj)
                except (ImportError, AttributeError) as exc:
                    raise ImportError(
                        f"Cannot import tool "
                        f"'{saved_tool['module']}.{saved_tool['qualname']}': {exc}.\n"
                        f"Pass it explicitly: load_expcard(..., tools=[...])"
                    ) from exc
            kwargs["tools"] = resolved_tools

    return ExpCardInit(**kwargs)
