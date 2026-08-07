"""Advanced tutorial step: a custom parser for the O/C survey.

DefaultLiteralAgree (used in steps 01-03) only captures a bare rating.
This parser also captures a one-sentence justification, matching the
follow-up question introduced in 02_trial_chain_survey.json.

Run with: python 05_custom_parser.py
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgreeWithReason(BaseModel):
    """Rate your agreement with the statement and justify it briefly."""

    rating: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="Agreement rating: 1 = strongly disagree, 5 = strongly agree."
    )
    reason: str = Field(..., description="One sentence justifying the rating.")


if __name__ == "__main__":
    from psychscanner import ExpCard, ExpCardInit, ScannerModel

    card = ExpCardInit(
        model="gpt-4o-mini",
        family="openai",
        parser=AgreeWithReason,
        task_file="examples/tasks/example_survey.json",
        cogtype="no",
        nsim=1,
        projectname="tutorial_custom_parser",
    )
    results = ScannerModel(expcard=ExpCard(card)).run()
    print(results[0][0]["pred_resp"])  # {"rating": 4, "reason": "..."}
