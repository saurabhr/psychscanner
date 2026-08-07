"""Advanced tutorial step: declare a task card in Python instead of JSON.

A script is the better authoring tool once a card needs things JSON can't
express directly - here, an image stimulus built from a helper function and a
tool bound only on the trial that needs it. Small/static cards should stay
JSON (see the sibling *.json files in this folder); reach for a script when
generation logic (loops, computed paths, conditionals) would otherwise be
duplicated by hand across items.

Run with: python 04_multimodal_tools_survey.py
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from psychscanner.datasets.prompts.multimodal import image_block

STIMULI_DIR = Path(__file__).parent / "stimuli"


@tool
def image_zoom(region: str) -> str:
    """Return a zoomed-in crop of the stimulus image for the named region."""
    return f"zoomed crop of region '{region}' (stub - wire up real cropping here)"


def build_task() -> dict:
    """Same O/C survey lineage as steps 01-03, extended with one image trial."""
    return {
        "tasktype": "survey",
        "taskname": "tutorial_multimodal_tools_survey",
        "instructions": {
            "definition": [
                "You will rate your agreement with the following statements.",
                "Use a scale from 1 (strongly disagree) to 5 (strongly agree).",
                "For the image trial, zoom in on any region you need before answering.",
            ]
        },
        "contexts": ["Openness to Experience"],
        "contexts_id": ["O"],
        "context_present": True,
        "chain_type": "item",
        "items": {
            "O_1": [{"trcode": "O_1", "stimulus": "I enjoy trying new things and exploring new ideas."}],
            "O_2": [
                {
                    "trcode": "O_2",
                    "stimulus": [
                        image_block(STIMULI_DIR / "mood_board.png"),
                        {"type": "text", "text": "Rate how much this image reflects your openness to new experiences."},
                    ],
                    "tools": ["image_zoom"],
                }
            ],
        },
        "parser": "DefaultLiteralAgree",
    }


if __name__ == "__main__":
    from psychscanner import ExpCard, ExpCardInit, ScannerModel

    card = ExpCardInit(
        model="gpt-4o-mini",
        family="openai",
        task_file=build_task(),
        tools=[image_zoom],
        cogtype="no",
        nsim=1,
        projectname="tutorial_multimodal_tools",
    )
    results = ScannerModel(expcard=ExpCard(card)).run()
    print(results[0][0]["pred_resp"])
