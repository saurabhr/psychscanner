"""Regenerate figures/*.png from the cached processed/exp1_1_combined.csv,
without re-running the LLM simulations in run.py. Same plotting logic as
run.py's figure section — kept in sync by hand since it's a one-off replot
utility, not part of the run pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
from _analysis_common import PALETTE, set_pub_style

set_pub_style()

PROC_DIR = HERE / "processed"
FIG_DIR = HERE / "figures"
MODEL = "smollm2:360m-instruct-fp16"

CONDITIONS = [
    {"memory": "SingleTurn", "chain_type": "item", "label": "item_ST"},
    {"memory": "Convo", "chain_type": "task", "label": "task_Cv"},
    {"memory": "Convo", "chain_type": "trial", "label": "trial_Cv"},
]

combined = pl.read_csv(PROC_DIR / "exp1_1_combined.csv").with_columns(
    pl.col("resp_Vividness").cast(pl.Float64),
    (pl.col("trial_idx").cast(pl.Int64) + 1).alias("trial_num"),
)

cond_labels = [c["label"] for c in CONDITIONS]
persona_colors = {"weird": PALETTE["blue"], "non_weird": PALETTE["vermillion"]}

# ── Figure 1: trial trajectory ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, cond in zip(axes, CONDITIONS):
    sub = combined.filter(pl.col("memory_cond") == cond["label"])
    for p_name in ("weird", "non_weird"):
        grp = sub.filter(pl.col("persona") == p_name)
        traj = (
            grp.group_by("trial_num")
            .agg(pl.col("resp_Vividness").mean().alias("mean_vividness"))
            .sort("trial_num")
        )
        ax.plot(
            traj["trial_num"], traj["mean_vividness"],
            marker="o", markersize=5,
            color=persona_colors[p_name],
            label=p_name.replace("_", "-"),
        )
    ax.set_title(cond["label"], fontsize=11)
    ax.set_xlabel("Trial number (1–16)")
    if ax is axes[0]:
        ax.set_ylabel("Mean vividness (1–5)")
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 17))
    ax.set_xticklabels(range(1, 17), fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"Exp 1.1 — VVIQ-16 Trial Trajectory  ·  model: {MODEL}  ·  n=5 per cell",
    fontsize=12, y=1.02,
)
plt.tight_layout()
fig.savefig(FIG_DIR / "trial_trajectory.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figures/trial_trajectory.png")

# ── Figure 2: total VVIQ per cell ────────────────────────────────────────
totals_pl = (
    combined
    .filter(pl.col("resp_Vividness").is_not_null())
    .group_by(["sim_idx", "persona", "memory_cond"])
    .agg(pl.col("resp_Vividness").sum().alias("total_vviq"))
)
rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(10, 5))
x_positions = np.arange(len(cond_labels))
offset = {"weird": -0.15, "non_weird": 0.15}

for p_name in ("weird", "non_weird"):
    sub = totals_pl.filter(pl.col("persona") == p_name)
    for xi, c_label in enumerate(cond_labels):
        vals = sub.filter(pl.col("memory_cond") == c_label)["total_vviq"].to_numpy().astype(float)
        if vals.size == 0:
            continue
        jx = xi + offset[p_name] + rng.uniform(-0.05, 0.05, size=vals.size)
        ax.scatter(
            jx, vals,
            color=persona_colors[p_name],
            s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
            label=p_name.replace("_", "-") if xi == 0 else "_nolegend_",
        )
        ax.plot(
            [xi + offset[p_name] - 0.08, xi + offset[p_name] + 0.08],
            [np.mean(vals), np.mean(vals)],
            color=persona_colors[p_name], lw=2,
        )

ax.set_xticks(x_positions)
ax.set_xticklabels(cond_labels, fontsize=11)
ax.set_ylabel("Total VVIQ score (sum of 16 items, max=80)")
ax.set_title(
    f"Exp 1.1 — Total VVIQ per Persona × Memory cell\n"
    f"model: {MODEL}  ·  n=5 per condition"
)
ax.legend(title="Persona", frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_DIR / "total_vviq_violin.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figures/total_vviq_violin.png")
