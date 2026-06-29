"""General-purpose parser classes — paradigm-agnostic.

This module groups Pydantic parser classes that are not tied to a specific
experimental paradigm:

  * Generic Likert / agreement ratings
  * Generic ``response`` + ``rating`` shapes
  * Word classification (Upper-case / Lower-case / Non-word)
  * Task readiness markers
  * Multi-phase task chain union (``TaskResponse``)

For vividness and reality-monitoring (RM) parsers, see
:mod:`psychscanner.datasets.prompts.parser_tasks`.

The old module name ``psychscanner.datasets.prompts.parser_extra`` continues to
work as a backward-compat re-export shim — every name previously importable
from there is still available there.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, confloat
from typing import Literal, Union


# =============================================================================
# Generic Likert / agreement
# =============================================================================

class DefaultLiteralAgree(BaseModel):
    """Give response on a Likert Scale of range 1 to 5 for the given item."""

    rating: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Agreement Rating.\n \
            Rate agreement on the scale of 1 to 5, where:\n \
            '5' to indicate that you absolutely agree that the statement describes you;\n \
            '1' to indicate that you totally disagree with the statement\n \
            '3' if you not sure, but always to make a choice.\n \
            Always give a single integer rating value ranging from 1 to 5 on the given item.",
    )


# =============================================================================
# Paired-associate learning
# =============================================================================

class PairedAssociateRecall(BaseModel):
    """Recall response for paired-associate learning — recalled word and confidence."""

    recalled_word: str = Field(
        ...,
        description="The single word you believe was paired with the probe word during the study phase.",
    )
    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description=(
            "Confidence in your recall.\n"
            "  1 = Not at all confident\n"
            "  2 = Slightly confident\n"
            "  3 = Somewhat confident\n"
            "  4 = Moderately confident\n"
            "  5 = Very confident\n"
            "  6 = Completely confident"
        ),
    )


# =============================================================================
# Serial probe (behavioral profiling task)
# =============================================================================

class SerialProbeResponse(BaseModel):
    """Recall response for a serial position probe — recalled number and confidence."""

    recalled_number: int = Field(
        ...,
        description="The number you believe was paired with the probe word during the study phase.",
    )
    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description=(
            "Confidence in your recall.\n"
            "  1 = Not at all confident\n"
            "  2 = Slightly confident\n"
            "  3 = Somewhat confident\n"
            "  4 = Moderately confident\n"
            "  5 = Very confident\n"
            "  6 = Completely confident"
        ),
    )


# =============================================================================
# Generic response + rating shapes
# =============================================================================

class DefaultParser(BaseModel):
    """
    give response on a rating scale for the given item.
    """

    response: list[str | int] = Field(
        ...,
        description="Conclude your response by providing the RESPONSE VALUE , enclosed in square brackets.\nOnly the number should be enclosed in square brackets.\nWrap the output in `json` tags, for example: .\nYou must always return valid JSON fenced by a markdown code block. Do not return any additional text.",
    )


class DefaultRatingParser(BaseModel):
    """Give response on a rating scale for the given item."""

    response: int | list[int] = Field(
        ...,
        description="Conclude your response by providing the RATING VALUE, enclosed in square brackets.\nYou must always return valid JSON fenced by a markdown code block. Do not return any additional text. Report only single rating value.",
    )


class DefaultResponseRating(BaseModel):
    """Give response on a rating scale for the given item."""

    response: str = Field(
        ...,
        description="Provide response based on the TRIAL INSTRUCTION or when given NEXT TASK INSTRUCTION FOR TRIALS give response as 'READY'.",
    )
    rating: float = Field(
        ...,
        description="Provide rating based on the  TRIAL INSTRUCTION or when given NEXT TASK INSTRUCTION FOR TRIALS give rating about the prospective confidence on how good you will do in upcoming trials on the scale of 0 (not at all confident) to 6 (highly confident). For rating on other task trials follow the TASK INSTRUCTIONS.",
    )


class DefaultResponseRatingConvo(BaseModel):
    response: str = Field(
        ...,
        description="If asked to give second word as response then return the second word  as Response as instructed.\n \
            'Different RESPONSE VALUES in [ ]:' \
            ['Upper-Case Word.'] = The word is an upper-case word;\n \
            ['Lower-Case Word.'] = The word is a lower-case word;\n \
            ['Non-Word.'] = The word is a non-word, that is not an English word nor a meaningful word.\n \
            ['Second Word Imagined (Internally generated).'] = The second word was imagined internally;\n \
            ['Second Word Perceived (Externally generated).'] = The second word was perceived externally;\n \
            ['New word.'] = The word was a new word that was not perceived or imagined.\n \
            ['Ready.'] = Respond ready when instructed;\n \
            ['Not ready'] = Respond not ready when instructed;\n \
            Always give a single response value on the given item.",
    )
    rating: float = Field(..., description="Give the rating value on a rating scale.")


class SimpleResponseRating(BaseModel):
    response: str = Field(
        ...,
        description="answer 'response' based on the trial prompt. 'response' is a single word based on trial type in word dynamics trial or a single option if the Trial is about Lexical quality or about expeirence quality about the given Trial prompt from the given list options depending on the trial.",
    )
    rating: float = Field(
        ...,
        description="Give rating as instructed in trial instructions for the rating within the rating scale.",
    )


class TwoResponses(BaseModel):
    Response_1: str = Field(..., description="Response to first component of the task based on the instructions.")
    Response_2: str = Field(..., description="Response to second component of the task based on the instructions.")


# =============================================================================
# Word classification (Upper-case / Lower-case / Non-word)
# =============================================================================

class DefaultWordCaseNonWord(BaseModel):
    """Model for response and confidence rating in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response to the question about the second word perceived.
    confidence : Literal
        The confidence rating on a scale from 1 to 6.
    """

    response: Literal["Lower-case Word", "Upper-case Word", "Non-Word"] = Field(
        ...,
        description="Response weather item is Lower-case Word, Upper-case Word or Non-Word?'\
            'Lower-case Word' = The item is a lower-case word;\n \
            'Upper-case Word' = The item is an upper-case word;\n \
            'Non-Word' = The item is a non-word, that is not an english word nor a meaning full word.\
            Always give a single response value on the given item.",
    )


class DefaultWordCaseNonWordConf16(BaseModel):
    """Model for response and confidence rating in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response to the question about the second word perceived.
    confidence : Literal
        The confidence rating on a scale from 1 to 6.
    """

    response: Literal["Lower-case Word", "Upper-case Word", "Non-Word"] = Field(
        ...,
        description="Response weather item is Lower-case Word, Upper-case Word or Non-Word?'\
            'Lower-case Word' = The item is a lower-case word;\n \
            'Upper-case Word' = The item is an upper-case word;\n \
            'Non-Word' = The item is a non-word, that is not an english word nor a meaning full word.\
            Always give a single response value on the given item.",
    )

    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence Rating Scale.\n \
            '6' = Highly confident;\n\
            and \
            '1' = Not at all confident.\n\
            Always give a single integer rating value ranging from 1 to 6 on the given item.",
    )


class WordNonWord(BaseModel):
    """
    Model for the 'Word' or 'Non-Word' response in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response indicating whether the second word is a word or a non-word.
    """

    response: Literal["Upper-Case Word", "Lower-Case Word", "Non-Word"] = Field(
        ...,
        description="Response indicating whether the second word is a word or a non-word.\n \
            'Upper-Case Word' = The word is an upper-case word;\n \
            'Lower-Case Word' = The word is a lower-case word;\n \
            'Non-Word' = The word is a non-word, that is not an English word nor a meaningful word.\n \
            Always give a single response value on the given item.",
    )


# =============================================================================
# Task readiness
# =============================================================================

class Ready(BaseModel):
    """Model for the 'Ready' response in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response indicating readiness to proceed.
    """

    response: Literal["Ready"] = Field(
        ...,
        description="Response indicating readiness to proceed.\n \
            Always give a single response value on the given item.",
    )


class TaskReadyConfidence(BaseModel):
    """Respond Ready and Rate the prospective task confidence on the scale of 1 to 6"""

    response: Literal["READY", "NO STOP"] = Field(
        ..., description="Give Response: READY to start the trials."
    )
    rating: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="How confident are you to accurately perform the prospective trials. Rate on the scale of 1 (not at all confident) to 6 (highly confident).",
    )


# =============================================================================
# Multi-phase task chain — union over all phase responses
# =============================================================================

class TaskResponse(BaseModel):
    ANSWER: Union[
        TaskReadyConfidence,
        Task_1_ResponseRate,
        Task_2_ResponseRate,
        Task_3_ResponseRate,
    ]


# =============================================================================
# Cross-module references for `TaskResponse` Union resolution
# =============================================================================
# `TaskResponse.ANSWER` references ``Task_1/2/3_ResponseRate`` which live in
# ``parser_tasks``. Imported here (after class definitions) to avoid a
# circular-import deadlock. With ``from __future__ import annotations`` the
# Union annotation is stored as a string and only resolved when
# ``model_rebuild()`` is called below.

from .parser_tasks import (  # noqa: E402
    Task_1_ResponseRate,
    Task_2_ResponseRate,
    Task_3_ResponseRate,
)

TaskResponse.model_rebuild()


# =============================================================================
# Cross-module re-exports
# =============================================================================
# These classes are RM-specific and live in ``parser_tasks``. Re-imported here
# for the convenience of code that finds general parsers in this module.

from .parser_tasks import (  # noqa: F401, E402
    DefaultRMEncodingPhase,
    DefaultRmChoiceConf16,
    ResponseRmScSt,
    Source,
)
