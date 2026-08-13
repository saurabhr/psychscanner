"""Analysis for Demonstration 03 — Personality Survey (BFI-44).

Reads ../data/processed/combined.csv (produced by
../simulation/run_personality_survey.py), reverse-scores the BFI-44 items,
computes per-trait scores per simulated participant, and compares
distributions across the 2 persona x 4 memory conditions.

Outputs (this directory):
  trait_scores.csv       -- one row per (persona, memory_cond, sim, trait): mean item score
  condition_summary.csv  -- one row per (persona, memory_cond, trait): n, mean, sd
  persona_effect.csv     -- per trait: weird vs non_weird_jung mean diff + Cohen's d
  memory_effect.csv      -- per trait: each memory_cond vs `stateless` mean diff + Cohen's d
  figures/trait_by_condition.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _analysis_common import save_figure, use_agg_backend

use_agg_backend()
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HERE      = Path(__file__).parent
DEMO_ROOT = HERE.parent
COMBINED  = DEMO_ROOT / "data" / "processed" / "combined.csv"
FIG_DIR   = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# BFI-44 reverse-scored items (score = 6 - raw rating)
REVERSE_ITEMS = {
    "A_01", "A_03", "A_06", "A_08",
    "C_02", "C_04", "C_05", "C_09",
    "E_02", "E_05", "E_07",
    "N_02", "N_05", "N_07",
    "O_07", "O_09",
}
TRAITS = ["E", "A", "C", "N", "O"]
TRAIT_NAMES = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
               "N": "Neuroticism", "O": "Openness"}
MEMORY_CONDS = ["stateless", "conversation", "conversation_windowed", "summary_windowed"]

combined = pl.read_csv(COMBINED)
if "resp_rating" not in combined.columns:
    raise SystemExit("resp_rating column missing from combined.csv — check parser output.")

scored = (
    combined
    .filter(pl.col("resp_rating").is_not_null())
    .with_columns([
        pl.col("trcode").str.slice(0, 1).alias("trait"),
        pl.when(pl.col("trcode").is_in(list(REVERSE_ITEMS)))
          .then(6 - pl.col("resp_rating"))
          .otherwise(pl.col("resp_rating"))
          .alias("rating_scored"),
    ])
)

# ── per-sim, per-trait mean score ────────────────────────────────────────
trait_scores = (
    scored
    .group_by(["sim_idx", "persona", "memory_cond", "trait"])
    .agg(pl.col("rating_scored").mean().alias("mean_score"))
    .sort(["persona", "memory_cond", "trait", "sim_idx"])
)
trait_scores.write_csv(HERE / "trait_scores.csv")
print(f"Saved trait_scores.csv ({trait_scores.shape[0]} rows)")

# ── condition summary: n, mean, sd per (persona, memory_cond, trait) ────
condition_summary = (
    trait_scores
    .group_by(["persona", "memory_cond", "trait"])
    .agg([
        pl.len().alias("n"),
        pl.col("mean_score").mean().alias("mean"),
        pl.col("mean_score").std().alias("sd"),
    ])
    .sort(["trait", "persona", "memory_cond"])
)
condition_summary.write_csv(HERE / "condition_summary.csv")
print(f"Saved condition_summary.csv ({condition_summary.shape[0]} rows)")

cs = condition_summary.to_pandas()
ts = trait_scores.to_pandas()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d. Returns nan if either group has <2 obs."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_sd == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_sd


# ── persona effect per trait (averaged across memory conditions) ────────
rows = []
for trait in TRAITS:
    a = ts[(ts.trait == trait) & (ts.persona == "weird")]["mean_score"].to_numpy()
    b = ts[(ts.trait == trait) & (ts.persona == "non_weird_jung")]["mean_score"].to_numpy()
    rows.append({
        "trait": trait, "trait_name": TRAIT_NAMES[trait],
        "weird_mean": a.mean(), "weird_sd": a.std(ddof=1), "weird_n": len(a),
        "non_weird_jung_mean": b.mean(), "non_weird_jung_sd": b.std(ddof=1), "non_weird_jung_n": len(b),
        "diff_weird_minus_nonweird": a.mean() - b.mean(),
        "cohens_d": cohens_d(a, b),
    })
persona_effect = pl.DataFrame(rows)
persona_effect.write_csv(HERE / "persona_effect.csv")
print(f"Saved persona_effect.csv\n{persona_effect}")

# ── memory effect per trait: each condition vs stateless control ────────
rows = []
for trait in TRAITS:
    ctrl = ts[(ts.trait == trait) & (ts.memory_cond == "stateless")]["mean_score"].to_numpy()
    for cond in MEMORY_CONDS[1:]:
        cvals = ts[(ts.trait == trait) & (ts.memory_cond == cond)]["mean_score"].to_numpy()
        rows.append({
            "trait": trait, "trait_name": TRAIT_NAMES[trait], "memory_cond": cond,
            "stateless_mean": ctrl.mean(), "cond_mean": cvals.mean(),
            "diff_from_stateless": cvals.mean() - ctrl.mean(),
            "cohens_d": cohens_d(cvals, ctrl),
        })
memory_effect = pl.DataFrame(rows)
memory_effect.write_csv(HERE / "memory_effect.csv")
print(f"Saved memory_effect.csv\n{memory_effect}")

# ── figure: per-trait mean +/- sd across the 4 memory conditions, by persona ─
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
persona_colors = {"weird": "steelblue", "non_weird_jung": "tomato"}
x_pos = np.arange(len(MEMORY_CONDS))
offset = {"weird": -0.15, "non_weird_jung": 0.15}
for ax, trait in zip(axes, TRAITS):
    for p_name, color in persona_colors.items():
        sub = cs[(cs.trait == trait) & (cs.persona == p_name)].set_index("memory_cond")
        means = [sub.loc[c, "mean"] if c in sub.index else np.nan for c in MEMORY_CONDS]
        sds = [sub.loc[c, "sd"] if c in sub.index else 0 for c in MEMORY_CONDS]
        ax.errorbar(x_pos + offset[p_name], means, yerr=sds, fmt="o-", color=color,
                    label=p_name.replace("_", "-"), capsize=3, markersize=5)
    ax.set_title(TRAIT_NAMES[trait], fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace("_", "\n") for c in MEMORY_CONDS], fontsize=7)
    ax.set_ylim(0.5, 5.5)
    if ax is axes[0]:
        ax.set_ylabel("Mean scored rating (1-5)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("BFI-44 per-trait scores by persona x memory condition (mean +/- SD, n=5/cell)",
             fontsize=11, y=1.03)
plt.tight_layout()
save_figure(fig, FIG_DIR, "trait_by_condition.png", bbox_inches="tight")
plt.close()
