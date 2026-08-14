"""Compare choice behavior across the 3 agents run in
../simulation/run_prospect_theory.py, joined against the task's own hidden
ground-truth factors (../../../tasks/prospect_theory_demo_ground_truth.csv)
-- there's no `corrAns` for a preference task, so "correct" isn't the
question; internal *rationality* is:

- EV-consistency: did the agent pick the higher-expected-value gamble?
- Dominance violations: on trials where one gamble strictly dominates the
  other (>= gain, <= loss, >= probability, at least one strict), did the
  agent pick the dominated one anyway? A rational, risk-neutral-or-not agent
  should never do this -- it's a strictly worse choice on every dimension.
- Switch rate by EV-transition: does the agent's own left/right choice flip
  more often on trials the ground truth labels "switch" (the higher-EV side
  changed from the previous trial) than "repeat"? Tests whether choices
  track the *trial-to-trial* value structure or something else (position
  stickiness, recency in the convo_memory condition, etc).

The MAP agent's raw response is already a bare "left"/"right" (its Actor
module is instructed to reply with only that word, and Search returns the
parsed side directly -- see ``map_agent.py``). The convo_memory and lats
agents are a plain ``model.invoke()`` with no such post-processing, so
llama-3.1-8b-instant answers in free-text chain-of-thought instead ("the
left option is likely to yield a higher return") despite the task
instructions asking for one word. Choice extraction below takes the *last*
left/right mention in the raw response, the same "parse the real signal
back out of free text" approach ``01_reward_task/analysis/analyze.py``
already uses for ``fb_response`` -- it's a proxy, not exact, and on trials
where the model never actually commits to an answer (hedges on both
options) it can pick up an incidental earlier mention instead of a real
decision; summarize() below reports how many responses had no left/right
mention at all so that failure mode stays visible rather than silently
folded into the parsed choices.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _analysis_common import PALETTE, demo_dirs, require_data, save_figure, set_pub_style, use_agg_backend, write_summary

DATA_DIR, OUT_DIR = demo_dirs(__file__)
GROUND_TRUTH_PATH = DATA_DIR.parents[2] / "tasks" / "prospect_theory_demo_ground_truth.csv"

_SIDE_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def _extract_choice(raw: str | None) -> str | None:
    matches = _SIDE_RE.findall(raw or "")
    return matches[-1].lower() if matches else None


def load_choices() -> tuple[pl.DataFrame, dict[str, int]]:
    gt = pl.read_csv(GROUND_TRUTH_PATH)
    rows = []
    unparsed = {}
    for csv_path in sorted(DATA_DIR.glob("prospect_*.csv")):
        agent = csv_path.stem.removeprefix("prospect_")
        df = pl.read_csv(csv_path)
        n_unparsed = 0
        for r in df.iter_rows(named=True):
            side = _extract_choice(r["pred_resp_raw"])
            if side is None:
                n_unparsed += 1  # ponytail: no left/right mention at all -- dropped, not imputed
                continue
            rows.append({"agent": agent, "trcode": r["trcode"], "trial_idx": r["trial_idx"], "choice": side})
        unparsed[agent] = n_unparsed
    choices = pl.DataFrame(rows)
    return choices.join(gt, on="trcode", how="inner"), unparsed


def summarize(df: pl.DataFrame, unparsed: dict[str, int]) -> None:
    lines = [f"Choices parsed: {len(df)} (agents: {sorted(df['agent'].unique())})"]
    lines.append(f"Unparsed responses (no left/right mention at all), by agent: {unparsed}\n")

    ev_consistent = df.with_columns(
        (
            ((pl.col("expected_value_difference") > 0) & (pl.col("choice") == "left"))
            | ((pl.col("expected_value_difference") < 0) & (pl.col("choice") == "right"))
        ).alias("chose_higher_ev")
    )
    lines.append("EV-consistency (chose the higher-expected-value gamble):")
    lines.append(str(ev_consistent.group_by("agent").agg(pl.col("chose_higher_ev").mean()).sort("agent")))
    lines.append("")

    dominance = df.filter(pl.col("dominance_relation") != "no_dominance").with_columns(
        (
            ((pl.col("dominance_relation") == "left_dominates") & (pl.col("choice") == "left"))
            | ((pl.col("dominance_relation") == "right_dominates") & (pl.col("choice") == "right"))
        ).alias("chose_dominant")
    )
    lines.append(f"Dominance trials: {len(dominance)}. Chose the dominant (strictly better) gamble:")
    if len(dominance):
        lines.append(str(dominance.group_by("agent").agg(pl.col("chose_dominant").mean()).sort("agent")))
    else:
        lines.append("(none in this slice)")
    lines.append("")

    transitions = df.filter(pl.col("value_difference_transition").is_not_null()).sort(["agent", "trial_idx"])
    switch_rows = []
    for agent in transitions["agent"].unique():
        sub = transitions.filter(pl.col("agent") == agent).sort("trial_idx")
        prev_choice = None
        for r in sub.iter_rows(named=True):
            if prev_choice is not None:
                switch_rows.append({
                    "agent": agent, "own_switch": r["choice"] != prev_choice,
                    "ev_transition": r["value_difference_transition"],
                })
            prev_choice = r["choice"]
    switches = pl.DataFrame(switch_rows)
    lines.append("Own choice-switch rate, by ground-truth EV-transition label:")
    if len(switches):
        lines.append(str(
            switches.group_by(["agent", "ev_transition"]).agg(pl.col("own_switch").mean())
            .sort(["agent", "ev_transition"])
        ))
    write_summary(OUT_DIR, lines)


def plot(df: pl.DataFrame) -> None:
    use_agg_backend()
    import matplotlib.pyplot as plt
    set_pub_style()

    ev_consistent = df.with_columns(
        (
            ((pl.col("expected_value_difference") > 0) & (pl.col("choice") == "left"))
            | ((pl.col("expected_value_difference") < 0) & (pl.col("choice") == "right"))
        ).alias("chose_higher_ev")
    )
    per_agent = ev_consistent.group_by("agent").agg(pl.col("chose_higher_ev").mean()).sort("agent")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(per_agent["agent"], per_agent["chose_higher_ev"], color=[PALETTE["blue"], PALETTE["vermillion"], PALETTE["bluish_green"]])
    ax.axhline(0.5, color=PALETTE["black"], linestyle="--", linewidth=1, label="chance")
    ax.set_ylabel("P(chose higher-EV gamble)")
    ax.set_title("Prospect Theory: expected-value consistency by agent architecture")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, OUT_DIR, "ev_consistency_by_agent.png")


def main() -> None:
    df, unparsed = load_choices()
    require_data(not df.is_empty(), DATA_DIR, "run ../simulation/run_prospect_theory.py first.")
    summarize(df, unparsed)
    plot(df)


if __name__ == "__main__":
    main()
