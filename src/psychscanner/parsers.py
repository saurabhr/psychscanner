"""Top-level namespace for all bundled parser classes.

Provides a single import path for the Pydantic parser classes that ship with
psychscanner, plus a registry for name-based lookup.

Examples
--------
>>> from psychscanner.parsers import DefaultLiteralVivid15
>>> from psychscanner.parsers import list_parsers, get_parser
>>> list_parsers()                         # ['AllResponseRMEI', ...]
>>> get_parser("DefaultLiteralVivid15")    # <class '...DefaultLiteralVivid15'>
"""
from __future__ import annotations

import inspect
from typing import Type

from pydantic import BaseModel

from psychscanner.datasets.prompts import parser_tasks as _parser_tasks_mod
from psychscanner.datasets.prompts import parser_general as _parser_general_mod


def _collect(mod) -> dict[str, Type[BaseModel]]:
    out: dict[str, Type[BaseModel]] = {}
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj is BaseModel:
            continue
        if not issubclass(obj, BaseModel):
            continue
        if obj.__module__ != mod.__name__:
            continue
        out[name] = obj
    return out


PARSER_REGISTRY: dict[str, Type[BaseModel]] = {
    **_collect(_parser_tasks_mod),
    **_collect(_parser_general_mod),
}


def list_parsers() -> list[str]:
    """Return a sorted list of all bundled parser class names."""
    return sorted(PARSER_REGISTRY)


def get_parser(name: str) -> Type[BaseModel]:
    """Look up a parser class by name.

    Raises
    ------
    KeyError
        If `name` is not a registered parser. The error message lists all
        available parser names to aid discovery.
    """
    if name not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY))
        raise KeyError(
            f"Parser {name!r} not found. Available parsers: {available}"
        )
    return PARSER_REGISTRY[name]


globals().update(PARSER_REGISTRY)

__all__ = [
    "PARSER_REGISTRY",
    "list_parsers",
    "get_parser",
    *sorted(PARSER_REGISTRY),
]
