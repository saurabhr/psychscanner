"""Generate a Prospect-Theory risky-choice task card, saved as .tcard.psyscan.

Design follows musslick/experiment_factor_discovery's real benchmark spec
(benchmarks/prospect_theory.md, config/synthetic_prospect_theory_benchmark.yaml,
fetched and read in full before writing this generator, not recalled from
memory): on each trial, two gambles (left/right) are shown, each with a gain
amount, a loss amount, and a gain probability (loss probability is implicit,
1 - gain_probability). Ground-truth hidden factors (expected value,
dominance, trial-to-trial transition) follow that spec's exact formulas.

Gamble parameters are sampled with plain Python `random.uniform()`, not
SweetPea's ContinuousFactor -- tried that first (the upstream benchmark's
own stated method), but it didn't actually sample when crossed the
straightforward way, and the upstream spec itself describes SweetPea's role
here as a uniform-random sampler behind "a dummy categorical crossing
scaffold" with no real counterbalancing structure (unlike Stroop/N-back/
Task-Switching, where the crossing itself is the point). `random.uniform`
draws from the exact same distributions with no ambiguity, so that's what's
used; documented here rather than silently substituted.

This is a preference task, not an accuracy task -- there is no `corrAns`.
Ground-truth hidden factors (needed to analyze *which* prospect-theory
regularities an agent's choices follow -- loss aversion, dominance
sensitivity, choice consistency after a side-switch) are written alongside
the task card as prospect_theory_ground_truth.csv, keyed by trcode.

Run from the psychscanner project root:
    python examples/tasks/generators/generate_prospect_theory.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from psychscanner import save_task_card

OUT_DIR = Path(__file__).parents[1]  # examples/tasks/
TASKNAME = "prospect_theory_demo"
N_TRIALS = 20  # upstream default is 200/participant; trimmed for a demo run
SEED = 0

_INSTRUCTIONS = (
    "On each trial you will see two gambles, Left and Right. Each gamble "
    "has a chance of a gain and a chance of a loss. Choose the gamble you "
    "prefer. Reply with only 'left' or 'right'."
)


def _sample_gamble(rng: random.Random) -> dict:
    return {
        "gain": round(rng.uniform(0, 100), 1),
        "loss": round(rng.uniform(0, 100), 1),
        "gain_probability": round(rng.uniform(0.05, 0.95), 3),
    }


def _expected_value(g: dict) -> float:
    p = g["gain_probability"]
    return p * g["gain"] - (1 - p) * g["loss"]


def _dominance(left: dict, right: dict) -> str:
    """Left dominates if >= gain, <= loss, >= gain_probability than right,
    with at least one strict advantage. Right dominance is symmetric."""
    left_ge = left["gain"] >= right["gain"] and left["loss"] <= right["loss"] and left["gain_probability"] >= right["gain_probability"]
    left_strict = left["gain"] > right["gain"] or left["loss"] < right["loss"] or left["gain_probability"] > right["gain_probability"]
    right_ge = right["gain"] >= left["gain"] and right["loss"] <= left["loss"] and right["gain_probability"] >= left["gain_probability"]
    right_strict = right["gain"] > left["gain"] or right["loss"] < left["loss"] or right["gain_probability"] > left["gain_probability"]
    if left_ge and left_strict:
        return "left_dominates"
    if right_ge and right_strict:
        return "right_dominates"
    return "no_dominance"


def _build_trials(n_trials: int = N_TRIALS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    trials = []
    prev_ev_diff = None
    for i in range(n_trials):
        left, right = _sample_gamble(rng), _sample_gamble(rng)
        left_ev, right_ev = _expected_value(left), _expected_value(right)
        ev_diff = left_ev - right_ev

        if prev_ev_diff is None:
            transition = None
        else:
            sign = lambda x: "left_better" if x > 0 else ("right_better" if x < 0 else "tie")
            transition = "repeat" if sign(ev_diff) == sign(prev_ev_diff) else "switch"
        prev_ev_diff = ev_diff

        trials.append({
            "trcode": f"prospect_{i + 1:02d}",
            "left": left, "right": right,
            "left_expected_value": round(left_ev, 2), "right_expected_value": round(right_ev, 2),
            "expected_value_difference": round(ev_diff, 2),
            "gain_difference": round(left["gain"] - right["gain"], 2),
            "loss_difference": round(left["loss"] - right["loss"], 2),
            "probability_difference": round(left["gain_probability"] - right["gain_probability"], 3),
            "dominance_relation": _dominance(left, right),
            "previous_expected_value_difference": round(prev_ev_diff, 2) if i > 0 else None,
            "value_difference_transition": transition,
        })
    return trials


def _stimulus(t: dict) -> str:
    left, right = t["left"], t["right"]
    return (
        f"Left: {left['gain_probability']:.0%} chance to gain {left['gain']}, "
        f"otherwise lose {left['loss']}.\n"
        f"Right: {right['gain_probability']:.0%} chance to gain {right['gain']}, "
        f"otherwise lose {right['loss']}."
    )


def _to_task_card(trials: list[dict]) -> dict:
    items = {
        t["trcode"]: [{"trcode": t["trcode"], "stimulus": _stimulus(t)}]
        for t in trials
    }
    return {
        "tasktype": "sc",
        "taskname": TASKNAME,
        "instructions": {"definition": [_INSTRUCTIONS]},
        "contexts": ["prospect"],
        "contexts_id": ["prospect"],
        "context_present": False,
        "chain_type": "item",
        "parser": None,
        "items": items,
    }


def _write_ground_truth(trials: list[dict], path: Path) -> None:
    fields = [
        "trcode", "left_gain", "left_loss", "left_gain_probability",
        "right_gain", "right_loss", "right_gain_probability",
        "left_expected_value", "right_expected_value", "expected_value_difference",
        "gain_difference", "loss_difference", "probability_difference",
        "dominance_relation", "previous_expected_value_difference", "value_difference_transition",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            row = {k: v for k, v in t.items() if k not in ("left", "right")}
            row["left_gain"], row["left_loss"], row["left_gain_probability"] = (
                t["left"]["gain"], t["left"]["loss"], t["left"]["gain_probability"]
            )
            row["right_gain"], row["right_loss"], row["right_gain_probability"] = (
                t["right"]["gain"], t["right"]["loss"], t["right"]["gain_probability"]
            )
            writer.writerow(row)


def demo() -> None:
    """Self-check against the benchmark spec's own formulas on a
    hand-constructed pair of trials, before any real generation runs."""
    left = {"gain": 80, "loss": 10, "gain_probability": 0.5}
    right = {"gain": 40, "loss": 20, "gain_probability": 0.5}
    # left EV = 0.5*80 - 0.5*10 = 35; right EV = 0.5*40 - 0.5*20 = 10
    assert _expected_value(left) == 35.0
    assert _expected_value(right) == 10.0
    # left has >= gain, <= loss, == gain_probability, strictly better on both -> left dominates
    assert _dominance(left, right) == "left_dominates"
    assert _dominance(right, left) == "right_dominates"
    tie = {"gain": 80, "loss": 10, "gain_probability": 0.5}
    assert _dominance(left, tie) == "no_dominance"

    trials = _build_trials(n_trials=6, seed=1)
    assert trials[0]["value_difference_transition"] is None
    assert all(t["value_difference_transition"] in ("repeat", "switch") for t in trials[1:])
    card = _to_task_card(trials)
    assert card["taskname"] == TASKNAME
    assert len(card["items"]) == 6
    print("demo() OK — expected-value/dominance/transition formulas verified against hand-worked trials")


if __name__ == "__main__":
    demo()
    trials = _build_trials()
    card = _to_task_card(trials)
    path = save_task_card(card, OUT_DIR / f"{TASKNAME}.tcard.psyscan")
    gt_path = OUT_DIR / f"{TASKNAME}_ground_truth.csv"
    _write_ground_truth(trials, gt_path)
    print(f"wrote {path} ({len(trials)} trials) and {gt_path}")
