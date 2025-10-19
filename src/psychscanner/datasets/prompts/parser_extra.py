from pydantic import BaseModel, Field, confloat
from typing import Literal, Union

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


class DefaultRatingParser(BaseModel):
    """Give response on a rating scale for the given item."""

    response: int | list[int] = Field(
        ...,
        description="Conclude your response by providing the RATING VALUE, enclosed in square brackets.\nYou must always return valid JSON fenced by a markdown code block. Do not return any additional text. Report only single rating value.",
    )


class DefaultParser(BaseModel):
    """
    give response on a rating scale for the given item.
    """

    response: list[str | int] = Field(
        ...,
        description="Conclude your response by providing the RESPONSE VALUE , enclosed in square brackets.\nOnly the number should be enclosed in square brackets.\nWrap the output in `json` tags, for example: .\nYou must always return valid JSON fenced by a markdown code block. Do not return any additional text.",
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


class TaskReadyConfidence(BaseModel):
    """Respond Ready and Rate the prospective task confidence on the scale of 1 to 6"""

    response: Literal["READY", "NO STOP"] = Field(
        ..., description="Give Response: READY to start the trials."
    )
    rating: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description="How confident are you to accurately perform the prospective trials. Rate on the scale of 1 (not at all confident) to 6 (highly confident).",
    )


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


class TaskResponse(BaseModel):
    ANSWER: Union[
        TaskReadyConfidence,
        Task_1_ResponseRate,
        Task_2_ResponseRate,
        Task_3_ResponseRate,
    ]


class SimpleResponseRating(BaseModel):
    response: str = Field(
        ...,
        description="answer 'response' based on the trial prompt. 'response' is a single word based on trial type in word dynamics trial or a single option if the Trial is about Lexical quality or about expeirence quality about the given Trial prompt from the given list options depending on the trial.",
    )
    rating: float = Field(
        ...,
        description="Give rating as instructed in trial instructions for the rating within the rating scale.",
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
