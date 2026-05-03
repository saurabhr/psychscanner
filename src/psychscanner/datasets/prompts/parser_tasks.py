"""Task-specific parser classes — vividness and reality-monitoring.

This module groups Pydantic parser classes that target specific experimental
paradigms:

  * vividness rating surveys (VVIQ-style)
  * the reality-monitoring (RM) word-pair task family

For paradigm-agnostic parsers (Likert agreement, generic response/rating,
word-classification, readiness markers, multi-phase task chain union) see
:mod:`psychscanner.datasets.prompts.parser_general`.

The old module name ``psychscanner.datasets.prompts.parser`` continues to work
as a backward-compat re-export shim — every name previously importable from
there is still available there.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, confloat

import json

from pydantic import BaseModel, Field
from typing import Literal, Union, Any


# =============================================================================
# Vividness rating parsers
# =============================================================================

class DefaultLiteralVivid010(BaseModel):
    """Give response on a Vividness Rating Scale of range 0 to 10 for the given item."""

    Vividness: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = Field(
        ...,
        description="Vividness Rating Scale.      \n \
            Vividness rating scale ranges from '0' (no image at all) to '10' (image as clear and vivid as real life).      \n\
            Always give a single rating value ranging from 0 to 10 on the given item.      ",
    )
class DefaultLiteralVivid15(BaseModel):
    """Give response on a Vividness Rating Scale of range 1 to 5 for the given item."""
    Vividness: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Vividness Rating Scale.      \n \
            '5' = Perfectly clear and as vivid as normal vision;      \n \
            '4' = Clear and reasonably vivid;      \n \
            '3' = Moderately clear and vivid;      \n \
            '2'  = Vague and dim;      \n \
            '1' = No image at all, you only “know” that you are thinking of the object.      \n \
            Always give a single integer rating value ranging from 1 to 5 on the given item.",
    )
class DefaultLiteralVivid15Pol(BaseModel):
    """Give response on a Vividness Rating Scale of range 1 to 5 for the given item."""

    Vividness: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="SKALA OCENIANIA: Przywołany przez dany element obrazu może być:      \n \
            '1' = Brak obrazu, „wiesz” tylko, że myślisz o jakimś obiekcie;      \n \
            '2' = Mglisty i przyciemniony;      \n \
            '3' = Umiarkowanie jasny i wyraźny;      \n \
            '4' = Jasny i dostatecznie wyraźny;      \n \
            '5' = Całkowicie jasny i wyraźny jak realny obraz.      \n \
            Zawsze podawaj pojedynczą wartość oceny całkowitej z zakresu od 1 do 5 dla danego elementu.",
    )


# =============================================================================
# Reality monitoring — encoding-phase parser
# =============================================================================

class DefaultRMEncodingPhase(BaseModel):
    second_word: str = Field(
        ...,
        description="In each trial, you will either Perceive (see) the 2nd word or Imagine\
        the 2nd word for the given 1st word.\
        you will be asked to type in the perceived or imagined SECOND word.",
    )

    Relatedness: float = Field(
        ...,
        description="Rate the relatedness of the first and second word.\n\
            Rate the relatedness of the words using the rating scale from 0% (not at all related) to 100% (highly related). The relatedness can be based on whether the 1st and 2nd words are: Phonetically (sound) related, Semantically (meaning) related, Can be grouped together in a common category.",
    )


# =============================================================================
# Reality monitoring — standalone source judgment
# =============================================================================

class Source(BaseModel):
    """Model for the 'Source' response in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response indicating the source of the second word.
    """

    source: Literal[
        "Second word Imagined (Internally generated)",
        "Second word Perceived (Externally generated)",
        "New word",
    ] = Field(
        ...,
        description="Response indicating the source of the second word.\n \
            'Second Word Imagined (Internally generated)' = The second word was imagined internally;\n \
            'Second Word Perceived (Externally generated)' = The second word was perceived externally;\n \
            'New word' = The second word was a new word that was not perceived or imagined.\n \
            Always give a single response value on the given item.",
    )


# =============================================================================
# Reality monitoring — single-chat full responses
# =============================================================================

class DefaultRmChoiceConf16(BaseModel):
    """Model for response and confidence rating in the Reality Monitoring Task.

    Attributes:
    ----------
    response : Literal
        The response to the question about the second word perceived.
    confidence : Literal
        The confidence rating on a scale from 1 to 6.
    """

    response: Literal[
        "Perceived 2nd Word (externally generated)",
        "Imagined 2nd Word (internally generated)",
        "New Word",
    ] = Field(
        ...,
        description="Response to the question: 'What was the second word you perceived?'\n \
                            '2nd Word Perceived (externally generated)' = The second word was perceived externally;\n \
                            '2nd Word Imagined (internally generated)' = The second word was imagined internally;\n \
                            'New Word' = The second word was a new word that was not perceived or imagined.\n \
                            Always give a single response value on the given item.",
    )

    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence Rating Scale.\n \
            '6' = Very confident;\n \
            '5' = Confident;\n \
            '4' = Somewhat confident;\n \
            '3' = Not very confident;\n \
            '2' = Not at all confident;\n \
            '1' = .\n \
            Always give a single integer rating value ranging from 1 to 6 on the given item.",
    )


class ResponseRmScSt(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Relatedness rating between the values of 'word_1' and 'word_2'.",
    )

    Judgment: Literal["internal", "external"] = Field(
        ..., description="Judgment about the type of generation of 'word_2'."
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type of 'word_2.",
    )


class ResponseRmStIE(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relatedness rating between the values of 'word_1' and 'word_2'.",
    )

    Judgment: Literal["internal", "external"] = Field(
        ...,
        description="Judgment about the type of generation of second word in word-pair.",
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type.",
    )


class ResponseRmStEI(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relatedness rating between the values of 'word_1' and 'word_2'.",
    )

    Judgment: Literal["external", "internal"] = Field(
        ...,
        description="Judgment about the type of generation of second word in word-pair.",
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type of 'word_2.",
    )


class ResponseRmStIEN(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description=json.dumps(
            {
                "definition": "Relatedness rating between the values of 'word_1' and 'word_2'.",
                "rating_scale": [
                    "Use any value in the range of 0 to 100 in relatedness percentage rating scale.",
                    "**0** percent signifies the words are **not at all related**.",
                    "**100** percent signifies the words **very highly related**.",
                    "Intermediate values between 0 and 100 denote intermediate values.",
                    "Relateness value is **stricity defined in the **percentage range of 0: not related at all, **TO** 100: very highly related**.Try to use all the values within the rating scale range as faithfully as possible and report a single number within the range of relatedness rating scale.",
                ],
            }
        ),
    )

    Judgment: Literal["internal", "external", "new"] = Field(
        ..., description="Judgment of the TRIAL value about its source."
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description=json.dumps(
            {
                "definition": "Confidence level in the correctness of the judgment about the generation type of value of 'word_2.",
                "confidence_scale": {
                    "1": "**Not at all confident.**",
                    "2": "**Slightly confident.**",
                    "3": "**Moderately confident.**",
                    "4": "**Fairly confident.**",
                    "5": "**Very confident.**",
                    "6": "**Highly confident.**",
                },
            }
        ),
    )


class ResponseRmStEIN(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description=json.dumps(
            {
                "definition": "Relatedness rating between the values of 'word_1' and 'word_2'.",
                "rating_scale": [
                    "Use any value in the range of 0 to 100 in relatedness percentage rating scale.",
                    "**0** percent signifies the words are **not at all related**.",
                    "**100** percent signifies the words **very highly related**.",
                    "Intermediate values between 0 and 100 denote intermediate values.",
                    "Relateness value is **stricity defined in the **percentage range of 0: not related at all, **TO** 100: very highly related**.Try to use all the values within the rating scale range as faithfully as possible and report a single number within the range of relatedness rating scale.",
                ],
            }
        ),
    )

    Judgment: Literal["external","internal", "new"] = Field(
        ..., description="Judgment of the TRIAL value about its source."
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description=json.dumps(
            {
                "definition": "Confidence level in the correctness of the judgment about the generation type of value of 'word_2.",
                "confidence_scale": {
                    "1": "**Not at all confident.**",
                    "2": "**Slightly confident.**",
                    "3": "**Moderately confident.**",
                    "4": "**Fairly confident.**",
                    "5": "**Very confident.**",
                    "6": "**Highly confident.**",
                },
            }
        ),
    )


# =============================================================================
# Reality monitoring — trial-chain field components
# =============================================================================

class Word2(BaseModel):
    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

class RelatednessRating(BaseModel):
    Relatedness_Rating: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description=json.dumps(
            {
                "definition": "Relatedness rating between the values of 'word_1' and 'word_2'.",
                "rating_scale": [
                    "Use any value in the range of 0 to 100 in relatedness percentage rating scale.",
                    "**0** percent signifies the words are **not at all related**.",
                    "**100** percent signifies the words **very highly related**.",
                    "Intermediate values between 0 and 100 denote intermediate values.",
                    "Relateness value is **stricity defined in the **percentage range of 0: not related at all, **TO** 100: very highly related**.Try to use all the values within the rating scale range as faithfully as possible and report a single number within the range of relatedness rating scale.",
                ],
            }
        ),
    )

class JudgmentIE(BaseModel):
    Judgment: Literal["internal", "external"] = Field(
        ..., description="Judgment about the type of generation of second word in word-pair."
    )

class JudgmentEI(BaseModel):
    Judgment: Literal["external", "internal"] = Field(
        ...,
        description="Judgment about the type of generation of second word in word-pair.",
    )

class JudgmentIEN(BaseModel):
    Judgment: Literal["internal", "external", "new"] = Field(
        ..., description="Judgment of the TRIAL value about its source."
    )

class JudgmentEIN(BaseModel):
    Judgment: Literal["external", "internal", "new"] = Field(
        ..., description="Judgment of the TRIAL value about its source."
    )

class Confidence16(BaseModel):
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type.",
    )


# =============================================================================
# Reality monitoring — trial-chain response classes
# =============================================================================

class Response_part_1_rm(BaseModel):
    """Response according to the instructions ."""

    Word_2: str = Field(
        ...,
        description="The given **OR** the imagined word depending on the nature of word pair.",
    )

    Rating: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relatedness rating between the values of 'word_1' and 'word_2'.",
    )



class Response_part_2_rm(BaseModel):
    Judgment: Literal["internal", "external"] = Field(
        ...,
        description="Judgment about the type of generation of second word in word-pair.",
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type.",
    )
class Response_part_2_rmrevo(BaseModel):
    Judgment: Literal["external","internal"] = Field(
        ...,
        description="Judgment about the type of generation of second word in word-pair.",
    )
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type.",
    )


# =============================================================================
# Reality monitoring — trial-chain unions
# =============================================================================

class AllResponseRMIE(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentIE, Confidence16]

class AllResponseRMEI(BaseModel):
    response: Union[Word2, RelatednessRating,JudgmentEI, Confidence16]

class AllResponseRMIEN(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentIEN, Confidence16]

class AllResponseRMEIN(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentEIN, Confidence16]


# =============================================================================
# Reality monitoring — multi-phase task chain (Task_1 / Task_2 / Task_3)
# =============================================================================

class Task_1_ResponseRate(BaseModel):
    """Response is the second Word for a given word pair in CURRENT TRIAL and How related/similar are the first word and second word in the word-pair on a scale of 0 (not at all related) to 100 (highly related) percentage"""

    response: str = Field(
        ...,
        description="The Second word for the given word-pair. For perceived trial it is same as the given second word in CURRENT TRIAL. For imagined trials with only first word and a blank (____), you complete the word-pair by giving a english word not used before in previous trials.",
    )
    rating: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Rate the relatedness of the first and second word.\n\
            Rate the relatedness based on the similarity of the words in the word-pair using the rating scale from 0% (not at all related) to 100% (highly related). The relatedness can be based on whether the 1st and 2nd words are: Phonetically (sound) related, Semantically (meaning) related, Can be grouped together in a common category. Try to use the rating scale appropriately.",
    )


class Task_2_ResponseRate(BaseModel):
    """Response weather the CURRENT TRIAL word is 'Upper-Case Word','Lower-Case Word','Non-word. Rate the confidence in the accuracy of your response on the scale of 1 (not at all confident) to 6 (highly confident)."""

    response: Literal["Upper-Case Word", "Lower-Case Word", "Non-word"] = Field(
        ...,
        description="Response indicating whether the second word is a Upper-Case or Lower-Case word or a non-word.\n \
            'Upper-Case Word' = The word is an upper-case word;\n \
            'Lower-Case Word' = The word is a lower-case word;\n \
            'Non-Word' = The word is a non-word, that is not an English word nor a meaningful word.\n \
            Always give a single response value on the CURRENT TRIAL.",
    )
    rating: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Rate the confidence on accuracy of your respnse on the scale of 1 (not at all confidence) to 6 (highly confidece). Rate the confidence on your subjective certainty that the response is correct. Intermediate values represent intermediate confidece levels. Try to use the confidece rating scale appropriately.",
    )


class Task_3_ResponseRate(BaseModel):
    """Response weather the CURRENT TRIAL word had a "Second Word Imagined (Internally generated)", "Second Word Perceived (Externally generated)", or is a "New Word". Rate the confidence in the accuracy of your response on the scale of 1 (not at all confident) to 6 (highly confident)."""

    response: Literal[
        "Second Word Imagined (Internally generated)",
        "Second Word Perceived (Externally generated)",
        "New Word",
    ] = Field(
        ...,
        description="Response indicating the source of the second word.\n \
            'Second Word Imagined (Internally generated)' = The second word was imagined internally;\n \
            'Second Word Perceived (Externally generated)' = The second word was perceived externally;\n \
            'New word' = The second word is a new word that was not perceived or imagined or mentioned in earlier task trials.\n \
            Always give a single response value on the given CURRENT TRIAL.",
    )
    rating: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Rate the confidence on accuracy of your respnse on the scale of 1 (not at all confidence) to 6 (highly confidece). Rate the confidence on your subjective certainty that the response is correct. Intermediate values represent intermediate confidece levels. Try to use the confidece rating scale appropriately.",
    )


# =============================================================================
# Cross-module re-exports
# =============================================================================
# `TwoResponses` is a generic shape that lives in ``parser_general``.
# Re-imported here for the convenience of code that finds task-specific
# parsers in this module.

from .parser_general import TwoResponses  # noqa: F401, E402
