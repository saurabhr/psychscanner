"""Compare reward accumulation and choice behavior across the agent types
run in ../simulation/run_reward_task.py, from the CSVs in ../data/.

Each CSV row's fb_response is BanditFeedback's own text, e.g.
"You chose C and received reward 4.94. Current value estimates - ...".
That string already carries the ground-truth arm + reward for the round, so
parsing it back out (rather than re-deriving reward some other way) is the
lazy and correct source of truth here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _analysis_common import (
    PALETTE, demo_dirs, require_data, save_figure, set_pub_style, use_agg_backend, write_summary,
)

DATA_DIR, OUT_DIR = demo_dirs(__file__)

ARM_MEANS = {"A": 2.0, "B": 6.0, "C": 4.0}
OPTIMAL_ARM = max(ARM_MEANS, key=ARM_MEANS.get)  # "B"
OPTIMAL_MEAN = ARM_MEANS[OPTIMAL_ARM]

FB_RE = re.compile(r"You chose ([ABC]) and received reward (-?\d+\.\d+)")


def load_rounds() -> pl.DataFrame:
    rows = []
    for csv_path in sorted(DATA_DIR.glob("bandit_*.csv")):
        agent = csv_path.stem.removeprefix("bandit_")
        df = pl.read_csv(csv_path)
        for r in df.iter_rows(named=True):
            m = FB_RE.search(r["fb_response"] or "")
            if not m:
                continue  # ponytail: unparsed fb_response rows are dropped, not imputed
            arm, reward = m.group(1), float(m.group(2))
            rows.append({
                "agent": agent, "sim_idx": r["sim_idx"], "trial_idx": r["trial_idx"],
                "arm": arm, "reward": reward,
            })
    return pl.DataFrame(rows)


def summarize(rounds: pl.DataFrame) -> None:
    lines = []
    lines.append(f"Rounds parsed: {len(rounds)}")
    lines.append(f"Optimal arm: {OPTIMAL_ARM} (true mean {OPTIMAL_MEAN})\n")

    per_agent = (
        rounds.group_by("agent")
        .agg(
            pl.len().alias("n_rounds"),
            pl.col("reward").mean().alias("mean_reward"),
            pl.col("reward").sum().alias("total_reward"),
            (pl.col("arm") == OPTIMAL_ARM).mean().alias("pct_optimal_arm"),
        )
        .sort("mean_reward", descending=True)
    )
    lines.append("Per-agent summary:")
    lines.append(str(per_agent))
    lines.append("")

    arm_dist = (
        rounds.group_by(["agent", "arm"]).agg(pl.len().alias("n"))
        .sort(["agent", "arm"])
    )
    lines.append("Arm selection counts:")
    lines.append(str(arm_dist))
    lines.append("")

    # Late-game (last third of rounds) mean reward, i.e. did behavior converge
    max_trial = rounds["trial_idx"].max()
    cutoff = max_trial - (max_trial // 3)
    late = rounds.filter(pl.col("trial_idx") >= cutoff)
    late_summary = (
        late.group_by("agent")
        .agg(pl.col("reward").mean().alias("late_mean_reward"),
             (pl.col("arm") == OPTIMAL_ARM).mean().alias("late_pct_optimal_arm"))
        .sort("late_mean_reward", descending=True)
    )
    lines.append(f"Late-game (trial_idx >= {cutoff}) summary:")
    lines.append(str(late_summary))

    write_summary(OUT_DIR, lines)


def plot(rounds: pl.DataFrame) -> None:
    use_agg_backend()
    import matplotlib.pyplot as plt
    set_pub_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in sorted(rounds["agent"].unique()):
        sub = rounds.filter(pl.col("agent") == agent).sort(["sim_idx", "trial_idx"])
        # mean cumulative reward across sims, aligned by trial_idx
        avg_by_trial = (
            sub.group_by("trial_idx").agg(pl.col("reward").mean().alias("mean_reward"))
            .sort("trial_idx")
        )
        cum = avg_by_trial["mean_reward"].cum_sum() / (avg_by_trial["trial_idx"] + 1)
        ax.plot(avg_by_trial["trial_idx"], cum, marker="o", markersize=5, linewidth=1.8, label=agent)

    ax.axhline(OPTIMAL_MEAN, color=PALETTE["black"], linestyle="--", linewidth=1,
               label=f"optimal arm ({OPTIMAL_ARM}) mean")
    ax.set_xlabel("Round")
    ax.set_ylabel("Running average reward")
    ax.set_title("3-armed bandit: running average reward by agent type")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, OUT_DIR, "reward_by_agent.png")


def main() -> None:
    rounds = load_rounds()
    require_data(not rounds.is_empty(), DATA_DIR, "run the simulation first.")
    summarize(rounds)
    plot(rounds)


if __name__ == "__main__":
    main()
