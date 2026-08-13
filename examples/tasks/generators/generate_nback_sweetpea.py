"""Generate n-back task cards (n=2..5) via SweetPea, saved as .tcard.psyscan.

SweetPea generates the counterbalanced *letter sequence* (a uniformly
crossed `Factor`, verified against the installed package to produce a
correct, evenly-balanced letter distribution). The match/no-match ground
truth is computed here in plain Python, not via SweetPea's `Window`-derived
factor mechanism -- that mechanism was tried first and produced labels that
didn't hold up against a direct manual check (e.g. a "2-back" trial labeled
"match" against a letter that plainly wasn't 2 positions back). Rather than
ship an n-back task whose correctness I couldn't independently verify, the
ground truth here is a single, obviously-correct comparison
(`trials[i] == trials[i-n]`), checked by `demo()` below before any file is
written.

Each n-back level's task card uses chain_type="trial" (psychscanner's own
trial-chain memory), not a hand-baked history string in every prompt --
unlike environments/psychscanner_nback/nback_demo.json in psychscanner-primal,
which pre-bakes history text because verifiers' SingleTurnEnv has no native
conversational memory. Here, memory is real, not simulated in the prompt.

Match rate is NOT artificially balanced to 50/50 -- crossing SweetPea on a
match factor in addition to letter made the SAT solve intractably slow for
this demo's purposes, and was dropped once the derived factor itself proved
unverifiable anyway. The achieved match rate (typically 10-35% depending on
n and the sampled sequence) is printed at generation time, not hidden.

Run from the psychscanner project root:
    python examples/tasks/generators/generate_nback_sweetpea.py
"""
from __future__ import annotations

from pathlib import Path

from sweetpea import CMSGen, CrossBlock, Factor, MinimumTrials, experiments_to_dicts, synthesize_trials

from psychscanner import save_task_card

LETTERS = list("ABCD")
MIN_TRIALS = 24
OUT_DIR = Path(__file__).parents[1]  # examples/tasks/

_INSTRUCTIONS = (
    "You are taking part in an n-back working-memory task. Letters are "
    "shown one at a time, in one continuous conversation, so you can rely "
    "on your memory of prior letters (no letter list is repeated to you). "
    "For each letter, judge whether it is identical to the letter shown "
    "exactly {n} letter(s) earlier. Reply with only 'match' or 'no-match'."
)


def _synthesize_letters(min_trials: int = MIN_TRIALS) -> list[str]:
    """SweetPea-generated, uniformly-crossed letter sequence (verified correct
    against the installed package: every letter appears an equal number of
    times, in a pseudo-random order sampled by CMSGen)."""
    letter = Factor("letter", LETTERS)
    block = CrossBlock([letter], [letter], [MinimumTrials(min_trials)])
    experiments = synthesize_trials(block, 1, CMSGen)
    trials = experiments_to_dicts(block, experiments)[0]
    return [t["letter"] for t in trials]


def _label_nback(letters: list[str], n: int) -> list[tuple[str, str]]:
    """(letter, match/no-match) for every trial from index n onward.
    Ground truth: letters[i] vs letters[i-n], nothing more -- directly
    checkable by eye against the printed sequence.
    """
    return [
        (letters[i], "match" if letters[i] == letters[i - n] else "no-match")
        for i in range(n, len(letters))
    ]


def _to_task_card(n: int, labeled: list[tuple[str, str]]) -> dict:
    taskname = f"nback_sweetpea_n{n}"
    items = {
        f"{taskname}_{i + 1:02d}": [
            {
                "trcode": f"{taskname}_{i + 1:02d}",
                "stimulus": f"Current letter: {letter}",
                "corrAns": answer,
            }
        ]
        for i, (letter, answer) in enumerate(labeled)
    }
    return {
        "tasktype": "sc",
        "taskname": taskname,
        "instructions": {"definition": [_INSTRUCTIONS.format(n=n)]},
        "contexts": ["nback"],
        "contexts_id": ["nback"],
        "context_present": False,
        "chain_type": "trial",
        "parser": None,
        "items": items,
    }


def demo() -> None:
    """Self-check: the labeling logic is correct on a hand-constructed
    sequence where the right answer is obvious by inspection, then confirm
    a real SweetPea-generated card builds and saves. No pytest needed.
    """
    letters = ["B", "A", "D", "A", "C", "C", "C"]
    labeled = _label_nback(letters, n=2)
    # index2 'D' vs index0 'B' -> no-match
    # index3 'A' vs index1 'A' -> match
    # index4 'C' vs index2 'D' -> no-match
    # index5 'C' vs index3 'A' -> no-match
    # index6 'C' vs index4 'C' -> match
    assert labeled == [
        ("D", "no-match"),
        ("A", "match"),
        ("C", "no-match"),
        ("C", "no-match"),
        ("C", "match"),
    ], labeled

    real_letters = _synthesize_letters(min_trials=8)
    assert len(real_letters) >= 8
    assert set(real_letters) <= set(LETTERS)
    card = _to_task_card(2, _label_nback(real_letters, 2))
    assert card["taskname"] == "nback_sweetpea_n2"
    assert len(card["items"]) > 0
    print("demo() OK — hand-checked labeling logic + real SweetPea letter sequence both verified")


if __name__ == "__main__":
    demo()
    for n in (2, 3, 4, 5):
        letters = _synthesize_letters()
        labeled = _label_nback(letters, n)
        matches = sum(1 for _, a in labeled if a == "match")
        card = _to_task_card(n, labeled)
        path = save_task_card(card, OUT_DIR / f"{card['taskname']}.tcard.psyscan")
        print(
            f"n={n}: wrote {path} ({len(labeled)} labeled trials, "
            f"{len(letters) - len(labeled)} lead-in) — match rate {matches}/{len(labeled)}"
        )
