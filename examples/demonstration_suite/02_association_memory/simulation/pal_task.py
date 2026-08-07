"""Task-construction and feedback helpers for the paired-associate learning
(PAL) demo, built on top of the real ``examples/tasks/pal50.json`` task card
(tasktype "episodic" — NOT the Reality Monitoring paradigm used elsewhere in
this repo).

``pal50.json`` is a STUDY -> TEST paired-associate task: 50 word pairs across
5 semantic-similarity levels (0, .25, .5, .75, 1.0 — see "contexts"/"contexts_id"
L0/L25/L50/L75/L100), 10 pairs per level. ``build_subset`` below trims that
down to a smaller, still-balanced (equal pairs per level) subset so the full
5-condition x N-model demo run stays fast, and reshapes trial ordering/`trcode`
sharing to realize each of the three memory-chaining conditions the framework
supports (see docs/guides/memory_types.md#chain_type):

  * single-turn      -- each trial its own stateless call     (memory=SingleTurn)
  * trial-chain      -- one pair's study+test share a thread, pairs independent
                         (memory=Convo, chain_type="trial")
  * episodic-chain   -- the whole study block then the whole test block share
                         one continuous conversation thread (memory=Convo,
                         chain_type="task") -- the "real" PAL session design.

Feedback conditions (4/5 in the demo) reuse the episodic-chain shape and add a
``FeedbackBase`` handler that scores each TEST response against the studied
word2.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from psychscanner import FeedbackBase

PAL50_PATH = Path(__file__).resolve().parents[3] / "examples" / "tasks" / "pal50.json"
LEVELS = ["L0", "L25", "L50", "L75", "L100"]
PAIRS_PER_LEVEL = 3  # subset size: 3 of 10 pairs/level -> 15 pairs, 30 trials/run


def load_pal50() -> dict:
    return json.loads(PAL50_PATH.read_text())


def build_subset(pairs_per_level: int = PAIRS_PER_LEVEL, share_trcode: bool = False) -> dict:
    """Return a pal50-shaped task dict restricted to the first
    ``pairs_per_level`` pairs of each similarity level.

    share_trcode=False (default): all STUDY items first, then all TEST items,
        in the same relative order as the source file -- a blocked study-then
        -test session (used for single-turn and episodic-chain conditions;
        ``card.memory``/``card.chain_type`` decide how much of it is
        remembered, not this function).
    share_trcode=True: each pair's study item and test item are given the
        SAME ``trcode`` (but distinct ``items`` dict keys, which is all the
        loader requires to stay unique) so they land on one LangGraph thread
        together, study immediately followed by its own test -- trial-chain.
        Different pairs still get different threads.
    """
    full = load_pal50()
    items = full["items"]
    pairs = [(level, f"{i:02d}") for level in LEVELS for i in range(1, pairs_per_level + 1)]

    out_items: dict = {}
    if share_trcode:
        for level, idx in pairs:
            enc_key, tst_key = f"{level}_enc_{idx}", f"{level}_tst_{idx}"
            enc, tst = copy.deepcopy(items[enc_key][0]), copy.deepcopy(items[tst_key][0])
            shared_trcode = f"{level}_pr_{idx}"
            enc["trcode"] = shared_trcode
            tst["trcode"] = shared_trcode
            out_items[enc_key] = [enc]
            out_items[tst_key] = [tst]
    else:
        for level, idx in pairs:
            enc_key = f"{level}_enc_{idx}"
            out_items[enc_key] = [copy.deepcopy(items[enc_key][0])]
        for level, idx in pairs:
            tst_key = f"{level}_tst_{idx}"
            out_items[tst_key] = [copy.deepcopy(items[tst_key][0])]

    task = copy.deepcopy(full)
    task["items"] = out_items
    task["taskname"] = f"pal_subset_{'trial' if share_trcode else 'block'}"
    return task


def _score(trial: dict, response: dict) -> tuple[bool, str]:
    given = str(response.get("recalled_word", "")).strip().lower()
    target = str(trial.get("word2", "")).strip().lower()
    return given == target, given


class CorrectIncorrectFeedback(FeedbackBase):
    """Categorical CORRECT/INCORRECT feedback injected after each TEST trial
    (feature 4: "Correct and Incorrect Feedback"). No feedback during study."""

    def on_response(self, trial: dict, response: dict) -> str | None:
        if trial.get("phase") != "test":
            return None
        ok, given = _score(trial, response)
        if ok:
            return f"CORRECT — '{trial['word1']}' was indeed paired with '{trial['word2']}'."
        return f"INCORRECT — you said '{given}', but the studied pair was '{trial['word1']}–{trial['word2']}'."


class RewardFeedback(FeedbackBase):
    """Numeric running-total reward feedback (+1 / +0 per test trial, reported
    as a cumulative score) — feature 5: "Reward Feedback". Reward-learning
    style signal rather than plain right/wrong text; one instance per
    simulated participant so the running total is per-participant."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.n_tested = 0

    def on_response(self, trial: dict, response: dict) -> str | None:
        if trial.get("phase") != "test":
            return None
        ok, _ = _score(trial, response)
        reward = 1.0 if ok else 0.0
        self.total_reward += reward
        self.n_tested += 1
        return f"Reward: {reward:+.1f}. Cumulative reward: {self.total_reward:.1f}/{self.n_tested} trials."
