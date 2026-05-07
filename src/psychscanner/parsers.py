"""Top-level namespace for all bundled parser classes.

Provides a single import path for the Pydantic parser classes that ship with
psychscanner, plus a registry for name-based lookup and a universal resolver.

Examples
--------
>>> from psychscanner.parsers import DefaultLiteralVivid15
>>> from psychscanner.parsers import list_parsers, get_parser, resolve_parser
>>> list_parsers()                              # ['AllResponseRMEI', ...]
>>> get_parser("DefaultLiteralVivid15")         # <class '...DefaultLiteralVivid15'>
>>> resolve_parser("DefaultLiteralVivid15")     # <class '...DefaultLiteralVivid15'>
>>> resolve_parser(DefaultLiteralVivid15)       # <class '...DefaultLiteralVivid15'>
>>> resolve_parser(None)                        # None  (no parser)
"""
from __future__ import annotations

import inspect
from typing import Callable, Type

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


def resolve_parser(
    value: str | Type[BaseModel] | Callable | None,
    task_parser_name: str | None = None,
) -> Type[BaseModel] | Callable | None:
    """Resolve any parser specification to a concrete form.

    Parameters
    ----------
    value:
        ``None`` or ``"0"``      → ``None`` (no structured output)
        ``"1"``                  → registry lookup using *task_parser_name*
        any other string         → registry lookup by name
        a ``BaseModel`` subclass → returned as-is
        a callable (not a class) → returned as-is (Form B dispatch function)
    task_parser_name:
        Used only when *value* is ``"1"``. Typically the ``parser`` field
        from the task JSON file.

    Raises
    ------
    KeyError
        If a name-based lookup fails (delegates to :func:`get_parser`).
    TypeError
        If *value* is not a recognised type.
    ValueError
        If ``value="1"`` but *task_parser_name* is empty or ``None``.
    """
    if value is None or value == "0":
        return None

    # Form B: user-supplied dispatch callable (not a class)
    if callable(value) and not isinstance(value, type):
        return value

    if isinstance(value, str):
        if value == "1":
            if not task_parser_name:
                raise ValueError(
                    "parser='1' requires the task JSON to have a non-empty "
                    "'parser' field naming a registered parser class."
                )
            return get_parser(task_parser_name)
        # direct name lookup — e.g. resolve_parser("DefaultLiteralVivid15")
        return get_parser(value)

    if isinstance(value, type) and issubclass(value, BaseModel):
        return value

    raise TypeError(
        f"parser must be None, '0', '1', a registered parser name string, "
        f"a BaseModel subclass, or a callable — got {type(value)!r}"
    )


globals().update(PARSER_REGISTRY)

__all__ = [
    "PARSER_REGISTRY",
    "list_parsers",
    "get_parser",
    "resolve_parser",
    *sorted(PARSER_REGISTRY),
]
