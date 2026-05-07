"""Backward-compatibility shim for ``parser.py``.

The parser classes that used to be defined here now live in
:mod:`psychscanner.datasets.prompts.parser_tasks` (vividness + reality-
monitoring) and :mod:`psychscanner.datasets.prompts.parser_general`
(``TwoResponses`` is the only one that moved across the split).

This module re-exports every name that was previously importable as
``from psychscanner.datasets.prompts.parser import <Name>`` so existing
user code keeps working unchanged. New code should import from the
themed modules directly, or — best of all — from the top-level
:mod:`psychscanner.parsers` namespace.
"""
from __future__ import annotations

# All vividness + RM classes (originally defined here).
from .parser_tasks import (  # noqa: F401
    AllResponseRMEI,
    AllResponseRMEIN,
    AllResponseRMIE,
    AllResponseRMIEN,
    Confidence16,
    DefaultLiteralVivid010,
    DefaultLiteralVivid15,
    DefaultLiteralVivid15Pol,
    JudgmentEI,
    JudgmentEIN,
    JudgmentIE,
    JudgmentIEN,
    RelatednessRating,
    Response_part_1_rm,
    Response_part_2_rm,
    Response_part_2_rmrevo,
    ResponseRmStEI,
    ResponseRmStEIN,
    ResponseRmStIE,
    ResponseRmStIEN,
    Word2,
)

# `TwoResponses` originally lived here; it now lives in ``parser_general``.
from .parser_general import TwoResponses  # noqa: F401

__all__ = [
    "AllResponseRMEI",
    "AllResponseRMEIN",
    "AllResponseRMIE",
    "AllResponseRMIEN",
    "Confidence16",
    "DefaultLiteralVivid010",
    "DefaultLiteralVivid15",
    "DefaultLiteralVivid15Pol",
    "JudgmentEI",
    "JudgmentEIN",
    "JudgmentIE",
    "JudgmentIEN",
    "RelatednessRating",
    "Response_part_1_rm",
    "Response_part_2_rm",
    "Response_part_2_rmrevo",
    "ResponseRmStEI",
    "ResponseRmStEIN",
    "ResponseRmStIE",
    "ResponseRmStIEN",
    "TwoResponses",
    "Word2",
]
