"""Generate a Stroop color-naming task card via SweetPea, saved as .tcard.psyscan.

Follows docs/guides/sweetpea_task_cards.md's own worked example exactly
(color x word crossing, congruency derived via WithinTrial). Unlike the
n-back generator's Window-derived factor (which produced labels that
failed a manual check -- see generate_nback_sweetpea.py's docstring),
WithinTrial's same-trial congruency here was independently re-checked
against every trial (`color == word` vs `congruency == "con"`, 0
mismatches) before being trusted -- see demo() below.

Run from the psychscanner project root:
    python examples/tasks/generators/generate_stroop_sweetpea.py
"""
from __future__ import annotations

from pathlib import Path

from sweetpea import (
    AtMostKInARow,
    CMSGen,
    CrossBlock,
    DerivedLevel,
    Factor,
    WithinTrial,
    experiments_to_dicts,
    synthesize_trials,
)

from psychscanner import save_task_card

COLORS = ["red", "blue", "green", "brown"]
OUT_DIR = Path(__file__).parents[1]  # examples/tasks/
TASKNAME = "stroop_sweetpea_demo"

_INSTRUCTIONS = (
    "You will see a color word printed in a colored ink. Name the ink "
    "color, ignoring what the word says. Reply with only the ink color."
)


def _synthesize() -> list[dict]:
    color = Factor("color", COLORS)
    word = Factor("word", COLORS)

    def congruent(c, w):
        return c == w

    congruency = Factor("congruency", [
        DerivedLevel("con", WithinTrial(congruent, [color, word])),
        DerivedLevel("inc", WithinTrial(lambda c, w: not congruent(c, w), [color, word])),
    ])

    block = CrossBlock(
        [color, word, congruency], [color, word], [AtMostKInARow(4, congruency)]
    )
    experiments = synthesize_trials(block, 1, CMSGen)
    return experiments_to_dicts(block, experiments)[0]


def _to_task_card(trials: list[dict]) -> dict:
    # trcode must be "{contexts_id value}_<anything>" -- task_prompts.py's
    # gen_trial_promptdata does trcode.split("_")[0] to look up the trial's
    # context in contexts_id (same gotcha hit and documented in
    # generate_nback_sweetpea.py / generate_vlm_shapes_demo.py; caught here
    # again by the same real ValueError during mock-llm verification).
    items = {
        f"stroop_{i + 1:02d}": [
            {
                "trcode": f"stroop_{i + 1:02d}",
                "stimulus": f"The word '{t['word'].upper()}' is shown in {t['color']} ink.",
                "corrAns": t["color"],
            }
        ]
        for i, t in enumerate(trials)
    }
    return {
        "tasktype": "sc",
        "taskname": TASKNAME,
        "instructions": {"definition": [_INSTRUCTIONS]},
        "contexts": ["stroop"],
        "contexts_id": ["stroop"],
        "context_present": False,
        "chain_type": "item",
        "parser": None,
        "items": items,
    }


def demo() -> None:
    """Self-check: congruency labels are correct against a direct
    color==word comparison for every trial, before any file is written."""
    trials = _synthesize()
    mismatches = [t for t in trials if (t["color"] == t["word"]) != (t["congruency"] == "con")]
    assert not mismatches, f"congruency label mismatch: {mismatches}"
    card = _to_task_card(trials)
    assert card["taskname"] == TASKNAME
    assert len(card["items"]) == len(trials)
    print(f"demo() OK — {len(trials)} trials, 0 congruency-label mismatches")


if __name__ == "__main__":
    demo()
    trials = _synthesize()
    card = _to_task_card(trials)
    path = save_task_card(card, OUT_DIR / f"{TASKNAME}.tcard.psyscan")
    n_con = sum(1 for t in trials if t["congruency"] == "con")
    print(f"wrote {path} ({len(trials)} trials, {n_con} congruent / {len(trials) - n_con} incongruent)")
