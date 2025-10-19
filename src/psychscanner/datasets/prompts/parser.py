from __future__ import annotations

from pydantic import BaseModel, Field, confloat

import json

from pydantic import BaseModel, Field
from typing import Literal, Union, Any




## Parsers for vividness survey.
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

# Parsers for Zero-Shot Reality Monitoring Tasks
# Single Chat Parsers
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

# Trial Chain Parsers
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

class Confidence16(BaseModel):
    Confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="Confidence level in the correctness of the judgment about the generation type.",
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

# combine parsers for the trial chain
class AllResponseRMIE(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentIE, Confidence16]

class AllResponseRMEI(BaseModel):
    response: Union[Word2, RelatednessRating,JudgmentEI, Confidence16]

class AllResponseRMIEN(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentIEN, Confidence16]

class AllResponseRMEIN(BaseModel):
    response: Union[Word2, RelatednessRating, JudgmentEIN, Confidence16]


class TwoResponses(BaseModel):
    Response_1: str = Field(..., description="Response to first component of the task based on the instructions.")
    Response_2: str = Field(..., description="Response to second component of the task based on the instructions.")