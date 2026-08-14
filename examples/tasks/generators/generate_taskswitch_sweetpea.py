"""Generate a task-switching task card via SweetPea, saved as .tcard.psyscan.

Classic two-task cued switching (Rogers & Monsell, 1995 style): each trial
shows a letter+digit pair (e.g. "A4") under a cue naming which task applies
-- LETTER (vowel/consonant) or NUMBER (odd/even). SweetPea generates the
counterbalanced *cue* sequence (with AtMostKInARow so it isn't all-switch
or all-repeat); the stimulus pairing and the correct answer are computed in
plain Python, not via SweetPea's Transition-derived factor -- Window/
Transition derivations already proved unreliable for n-back (see
generate_nback_sweetpea.py's docstring), so this generator doesn't trust
them for switch/repeat labeling either. switch/repeat itself is also
computed directly (task[i] != task[i-1]) and checked by demo() below.

Run from the psychscanner project root:
    python examples/tasks/generators/generate_taskswitch_sweetpea.py
"""
from __future__ import annotations

import random
from pathlib import Path

from sweetpea import CMSGen, CrossBlock, Factor, MinimumTrials, experiments_to_dicts, synthesize_trials

from psychscanner import save_task_card

OUT_DIR = Path(__file__).parents[1]  # examples/tasks/
TASKNAME = "taskswitch_sweetpea_demo"
MIN_TRIALS = 24
SEED = 0

LETTERS_VOWEL = list("AEIOU")
LETTERS_CONSONANT = list("BCDFGHJKLMNPQRSTVWXYZ")
DIGITS_ODD = list("13579")
DIGITS_EVEN = list("02468")

_INSTRUCTIONS = (
    "Each trial shows a letter+digit pair and a task cue. If the cue is "
    "LETTER, say 'vowel' if the letter is a vowel, else 'consonant' "
    "(ignore the digit). If the cue is NUMBER, say 'odd' or 'even' for the "
    "digit (ignore the letter). Reply with only your answer."
)


def _synthesize_cues(min_trials: int = MIN_TRIALS) -> list[str]:
    """SweetPea-generated, uniformly-crossed task-cue sequence."""
    task = Factor("task", ["LETTER", "NUMBER"])
    block = CrossBlock([task], [task], [MinimumTrials(min_trials)])
    experiments = synthesize_trials(block, 1, CMSGen)
    return [t["task"] for t in experiments_to_dicts(block, experiments)[0]]


def _stimulus_for(task: str, rng: random.Random) -> tuple[str, str, str]:
    """(letter, digit, corrAns) -- corrAns depends only on the cued task."""
    if task == "LETTER":
        is_vowel = rng.random() < 0.5
        letter = rng.choice(LETTERS_VOWEL if is_vowel else LETTERS_CONSONANT)
        digit = rng.choice(DIGITS_ODD + DIGITS_EVEN)
        return letter, digit, ("vowel" if is_vowel else "consonant")
    is_odd = rng.random() < 0.5
    digit = rng.choice(DIGITS_ODD if is_odd else DIGITS_EVEN)
    letter = rng.choice(LETTERS_VOWEL + LETTERS_CONSONANT)
    return letter, digit, ("odd" if is_odd else "even")


def _build_trials(cues: list[str], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    trials = []
    for i, task in enumerate(cues):
        letter, digit, ans = _stimulus_for(task, rng)
        switch = None if i == 0 else (task != cues[i - 1])
        trials.append({"task": task, "letter": letter, "digit": digit, "corrAns": ans, "switch": switch})
    return trials


def _to_task_card(trials: list[dict]) -> dict:
    items = {
        f"taskswitch_{i + 1:02d}": [
            {
                "trcode": f"taskswitch_{i + 1:02d}",
                "stimulus": f"Cue: {t['task']}. Stimulus: {t['letter']}{t['digit']}",
                "corrAns": t["corrAns"],
            }
        ]
        for i, t in enumerate(trials)
    }
    return {
        "tasktype": "sc",
        "taskname": TASKNAME,
        "instructions": {"definition": [_INSTRUCTIONS]},
        "contexts": ["taskswitch"],
        "contexts_id": ["taskswitch"],
        "context_present": False,
        "chain_type": "item",
        "parser": None,
        "items": items,
    }


def demo() -> None:
    """Self-check: corrAns is correct for every trial by direct
    recomputation, and switch/repeat matches a plain task[i] != task[i-1]
    comparison. No pytest needed."""
    cues = ["LETTER", "LETTER", "NUMBER", "LETTER", "NUMBER", "NUMBER"]
    trials = _build_trials(cues, seed=1)
    for i, t in enumerate(trials):
        if t["task"] == "LETTER":
            expected = "vowel" if t["letter"] in LETTERS_VOWEL else "consonant"
        else:
            expected = "odd" if t["digit"] in DIGITS_ODD else "even"
        assert t["corrAns"] == expected, (i, t, expected)
        expected_switch = None if i == 0 else (cues[i] != cues[i - 1])
        assert t["switch"] == expected_switch, (i, t, expected_switch)

    real_cues = _synthesize_cues(min_trials=8)
    assert len(real_cues) >= 8
    assert set(real_cues) <= {"LETTER", "NUMBER"}
    card = _to_task_card(_build_trials(real_cues))
    assert card["taskname"] == TASKNAME
    assert len(card["items"]) == len(real_cues)
    print("demo() OK — corrAns + switch/repeat labeling verified against direct recomputation")


if __name__ == "__main__":
    demo()
    cues = _synthesize_cues()
    trials = _build_trials(cues)
    card = _to_task_card(trials)
    path = save_task_card(card, OUT_DIR / f"{TASKNAME}.tcard.psyscan")
    n_switch = sum(1 for t in trials if t["switch"])
    print(f"wrote {path} ({len(trials)} trials, {n_switch} switch / {len(trials) - n_switch - 1} repeat)")
