"""Exp 1.1 — VVIQ-16: Persona (WEIRD × non-WEIRD) × Memory (3 levels) × 5 runs.

Run from the psychscanner project root:
    conda activate psyscan
    python examples/demonstration_suite/03_personality_survey/advanced/exp1_1_vviq16/run.py

Outputs
-------
raw/exp1_1_{persona}_{memory}.csv      — 80 rows each (5 sims × 16 trials)
processed/exp1_1_combined.csv          — 480 rows, all conditions
figures/trial_trajectory.png
figures/total_vviq_violin.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[4]          # psychscanner/
ADV_ROOT  = HERE.parents[0]          # .../03_personality_survey/advanced/

TASK_FILE   = REPO_ROOT / "examples" / "tasks" / "vviq16.json"
PERSONA_DIR = ADV_ROOT / "personas"
RAW_DIR     = HERE / "raw"
PROC_DIR    = HERE / "processed"
FIG_DIR     = HERE / "figures"

for d in (RAW_DIR, PROC_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── experiment parameters ──────────────────────────────────────────────────────
MODEL  = "smollm2:360m-instruct-fp16"
FAMILY = "ollama"
PARAMS = {"temperature": 0}

CONDITIONS = [
    {"memory": "SingleTurn", "chain_type": "item",  "label": "item_ST"},
    {"memory": "Convo",      "chain_type": "task",  "label": "task_Cv"},
    {"memory": "Convo",      "chain_type": "trial", "label": "trial_Cv"},
]

# Persona: axis file (1 stmt) × traits file (5 stmts) → 5 participants per condition
TRAITS_FILE = PERSONA_DIR / "traits_n5.json"
PERSONAS = {
    "weird":     PERSONA_DIR / "weird"     / "weird_axis.json",
    "non_weird": PERSONA_DIR / "non_weird" / "non_weird_axis.json",
}

# ── run simulations ────────────────────────────────────────────────────────────
frames: list[pl.DataFrame] = []

for persona_name, axis_file in PERSONAS.items():
    for cond in CONDITIONS:
        run_label = f"exp1_1_{persona_name}_{cond['label']}"
        print(f"\n{'='*60}")
        print(f"  {run_label}")
        print(f"{'='*60}")

        card = ExpCardInit(
            model         = MODEL,
            family        = FAMILY,
            parameters    = PARAMS,
            cogtype       = "custom",
            persona_files = [axis_file, TRAITS_FILE],
            task_file     = TASK_FILE,
            parser        = "1",         # → DefaultLiteralVivid15 from task JSON
            memory        = cond["memory"],
            chain_type    = cond["chain_type"],
            projectname   = run_label,
            proj_dir      = RAW_DIR,
            tunnel_status = "0",
        )
        expcard = ExpCard(card)
        scanner = ScannerModel(expcard=expcard)
        scanner.run()

        csv_path = RAW_DIR / f"{run_label}.csv"
        df = to_csv(scanner, path=csv_path)

        # tag with experimental condition columns
        df = df.with_columns([
            pl.lit(persona_name).alias("persona"),
            pl.lit(cond["label"]).alias("memory_cond"),
        ])
        frames.append(df)
        print(f"  saved {len(df)} rows → {csv_path.name}")

# ── combine all conditions ─────────────────────────────────────────────────────
combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "exp1_1_combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} → {combined_path}")

# ── analysis & figures ─────────────────────────────────────────────────────────
if "resp_Vividness" not in combined.columns:
    print("WARNING: resp_Vividness column missing — check parser output. Skipping figures.")
else:
    df_pd = combined.to_pandas()
    df_pd["resp_Vividness"] = df_pd["resp_Vividness"].astype(float)
    df_pd["trial_num"] = df_pd["trial_idx"].astype(int) + 1

    cond_labels = [c["label"] for c in CONDITIONS]
    persona_colors = {"weird": "steelblue", "non_weird": "tomato"}

    # ── Figure 1: trial trajectory ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        sub = df_pd[df_pd["memory_cond"] == cond["label"]]
        for p_name in ("weird", "non_weird"):
            grp = sub[sub["persona"] == p_name]
            traj = grp.groupby("trial_num")["resp_Vividness"].mean()
            ax.plot(
                traj.index, traj.values,
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
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Exp 1.1 — VVIQ-16 Trial Trajectory  ·  model: {MODEL}  ·  n=5 per cell",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trial_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/trial_trajectory.png")

    # ── Figure 2: total VVIQ per cell ─────────────────────────────────────────
    # sum the 16 items per simulation
    totals_pl = (
        combined
        .filter(pl.col("resp_Vividness").is_not_null())
        .group_by(["sim_idx", "persona", "memory_cond"])
        .agg(pl.col("resp_Vividness").sum().alias("total_vviq"))
    )
    totals = totals_pl.to_pandas()

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(cond_labels))
    offset = {"weird": -0.15, "non_weird": 0.15}

    for p_name in ("weird", "non_weird"):
        sub = totals[totals["persona"] == p_name]
        for xi, c_label in enumerate(cond_labels):
            vals = sub[sub["memory_cond"] == c_label]["total_vviq"].values.astype(float)
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
    ax.legend(title="Persona")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "total_vviq_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/total_vviq_violin.png")

    # ── per-scene descriptives → analysis/ ────────────────────────────────────
    scene_stats = (
        combined
        .filter(pl.col("resp_Vividness").is_not_null())
        .with_columns(
            pl.col("trcode").str.slice(0, 2).alias("scene")
        )
        .group_by(["persona", "memory_cond", "scene"])
        .agg([
            pl.col("resp_Vividness").mean().alias("mean_vividness"),
            pl.col("resp_Vividness").std().alias("sd_vividness"),
            pl.len().alias("n"),
        ])
        .sort(["persona", "memory_cond", "scene"])
    )
    scene_stats.write_csv(HERE / "analysis" / "scene_descriptives.csv")
    print("Saved analysis/scene_descriptives.csv")

print(f"\nExp 1.1 complete.  {combined.shape[0]} trial records across 6 conditions.")
