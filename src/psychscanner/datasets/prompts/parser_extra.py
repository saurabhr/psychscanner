"""Backward-compatibility shim for ``parser_extra.py``.

The parser classes that used to be defined here are now split between
:mod:`psychscanner.datasets.prompts.parser_general` (paradigm-agnostic
shapes — Likert, generic response/rating, word classification, readiness,
``TaskResponse``) and :mod:`psychscanner.datasets.prompts.parser_tasks`
(reality-monitoring task chain pieces, ``Source``, ``DefaultRmChoiceConf16``,
``ResponseRmScSt``, ``DefaultRMEncodingPhase``).

This module re-exports every name that was previously importable as
``from psychscanner.datasets.prompts.parser_extra import <Name>`` so existing
user code keeps working unchanged. New code should import from the
themed modules directly, or — best of all — from the top-level
:mod:`psychscanner.parsers` namespace.
"""
from __future__ import annotations

# Paradigm-agnostic classes that originally lived here.
from .parser_general import (  # noqa: F401
    DefaultLiteralAgree,
    DefaultParser,
    DefaultRatingParser,
    DefaultResponseRating,
    DefaultResponseRatingConvo,
    DefaultWordCaseNonWord,
    DefaultWordCaseNonWordConf16,
    Ready,
    SimpleResponseRating,
    TaskReadyConfidence,
    TaskResponse,
    TwoResponses,
    WordNonWord,
)

# Reality-monitoring classes that originally lived here, now in ``parser_tasks``.
from .parser_tasks import (  # noqa: F401
    DefaultRMEncodingPhase,
    DefaultRmChoiceConf16,
    ResponseRmScSt,
    Source,
    Task_1_ResponseRate,
    Task_2_ResponseRate,
    Task_3_ResponseRate,
)

__all__ = [
    "DefaultLiteralAgree",
    "DefaultParser",
    "DefaultRMEncodingPhase",
    "DefaultRatingParser",
    "DefaultResponseRating",
    "DefaultResponseRatingConvo",
    "DefaultRmChoiceConf16",
    "DefaultWordCaseNonWord",
    "DefaultWordCaseNonWordConf16",
    "Ready",
    "ResponseRmScSt",
    "SimpleResponseRating",
    "Source",
    "Task_1_ResponseRate",
    "Task_2_ResponseRate",
    "Task_3_ResponseRate",
    "TaskReadyConfidence",
    "TaskResponse",
    "TwoResponses",
    "WordNonWord",
]
