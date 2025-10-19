"""This module provides a default factory for creating instances of the PsychScanner class.

It includes:
- A DEFAULT_FACTORY dictionary with default parameters for PsychScanner instances.
- A DefaultFactory class to manage and retrieve these default settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pathlib import Path
# from .. import memories  # noqa: TID252


DEFAULT_POP = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
)
DEFAULT_PER = ""
DEFAULT_PROJ_NAME = "DEFAULTPROJ"
DEFAULT_MODEL_NAME = "mock-chat-model"
DEFAULT_FAMILY_NAME = "mock-llm"
DEFAULT_MEMORY = "SingleTurn"

DEFAULT_CARD = []
DEFAULT_AGENT_CONFIG = []


DEFAULT_FACTORY = {
    "EXP_CARD_INIT": {
        "modelname": DEFAULT_MODEL_NAME,
        "familyname": DEFAULT_FAMILY_NAME,
        "parameters": {},
        "persona": DEFAULT_PER,
        "population": DEFAULT_POP,
        "task": "0",
        "task_context": None,
        "memory": DEFAULT_MEMORY,
        "feedback": "",
        "tunnel_status": "0",
        "tunnel_k": 0,
        "projectname": DEFAULT_PROJ_NAME,
        "tags": [],
        "parser": None,
        "dotenv": None,
        "enabletqdm": False,
    }
}

"""
    EXP_CARD_IN_CLI["modelname"] = modelname
    EXP_CARD_IN_CLI["familyname"] = familyname
    EXP_CARD_IN_CLI["parameters"] = parameters
    EXP_CARD_IN_CLI["persona"] = persona
    EXP_CARD_IN_CLI["population"] = population
    EXP_CARD_IN_CLI["task"] = task
    EXP_CARD_IN_CLI["task_context"] = task_context
    EXP_CARD_IN_CLI["memory"] = memory
    EXP_CARD_IN_CLI["memory_k"] = memory_k
    EXP_CARD_IN_CLI["reflection"] = reflection
    EXP_CARD_IN_CLI["feedback"] = feedback
    EXP_CARD_IN_CLI["tunnel_status"] = tunnel_status
    EXP_CARD_IN_CLI["tunnel_k"] = tunnel_k
    EXP_CARD_IN_CLI["projectname"] = projectname
    EXP_CARD_IN_CLI["tags"] = tags
    EXP_CARD_IN_CLI["parser"] = parser
    EXP_CARD_IN_CLI["parser_raw"] = parser_raw
    EXP_CARD_IN_CLI["login_env"] = login_env
    EXP_CARD_IN_CLI["enabletqdm"] = enabletqdm
    EXP_CARD_IN_CLI["proj_dir"] = proj_dir

"""

PSCAN_OLLAMA_DEFAULT_FACTORY = {**DEFAULT_FACTORY}
PSCAN_OLLAMA_DEFAULT_FACTORY["EXP_CARD_INIT"]["familyname"] = "ollama"
PSCAN_OLLAMA_DEFAULT_FACTORY["EXP_CARD_INIT"]["modelname"] = "llama2"

DEFAULT_SURVEY = {}
DEFAULT_TASK = {}

class DefaultSurveyParser(BaseModel):
    """Give response on a rating scale for the given item."""

    rating: list[int] = Field(
        ...,
        description="Give response by providing the vivdness rating as the integer based on the given vividness rating scale in the range of 1 to 5.",
    )


class DefaultParser(BaseModel):
    """Give response on a rating scale for the given item."""

    response: str = Field(
        ...,
        description="Give response as instructed in the task context or instructions.",
    )


class DefaultFactory:
    """This class provides a default factory for creating instances of the PsychScanner class.

    It contains a dictionary with default parameters for the PsychScanner instance.
    """

    def __init__(self, new_factory: dict | None = None) -> None:
        """Initialize the DefaultFactory with default parameters."""
        self._default_factory = DEFAULT_FACTORY
        if new_factory is not None:
            self._default_factory = new_factory

    def get_factory_settings(self) -> dict:
        """Retrieve the default factory settings.

        Returns: Factory settings of scanner cards in the form of a dictionary.
        -------
        dict
            A dictionary containing the default factory settings.
        """
        return self._default_factory
