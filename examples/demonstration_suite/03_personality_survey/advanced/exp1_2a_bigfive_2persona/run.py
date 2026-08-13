"""Exp 1.2a — BFI-44: Persona (WEIRD × non-WEIRD+Jung) × SummaryK(2) × MemoryK(2) + Control.

Run from the psychscanner project root:
    conda activate psyscan
    python examples/demonstration_suite/03_personality_survey/advanced/exp1_2a_bigfive_2persona/run.py

Design
------
Persona (2): WEIRD, non-WEIRD+Jung
Memory conditions (5 per persona):
  item_ST              — chain_type=item, memory=SingleTurn (control baseline)
  task_sk1_mk4         — chain_type=task, summary_k=1,  memory_k=4
  task_sk1_mk16        — chain_type=task, summary_k=1,  memory_k=16
  task_sk5_mk4         — chain_type=task, summary_k=5,  memory_k=4
  task_sk5_mk16        — chain_type=task, summary_k=5,  memory_k=16

Participants per condition: 5 (axis(1) × traits_n5(5) Cartesian product)

Outputs
-------
raw/exp1_2a_{persona}_{cond}.csv      — 220 rows each (5 sims × 44 trials)
processed/exp1_2a_combined.csv        — 2200 rows, all conditions
figures/per_trait_per_cell.png
figures/trial_chain_vs_control_contrast.png
figures/radar_per_persona.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import shutil

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

# ── paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[4]   # psychscanner/
ADV_ROOT  = HERE.parents[0]   # .../03_personality_survey/advanced/

TASK_FILE   = REPO_ROOT / "examples" / "tasks" / "bfi44.json"
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
    {"memory": "SingleTurn", "chain_type": "item", "summary_k": 0,  "memory_k": -1, "label": "item_ST"},
    {"memory": "Convo",      "chain_type": "task", "summary_k": 1,  "memory_k": 4,  "label": "task_sk1_mk4"},
    {"memory": "Convo",      "chain_type": "task", "summary_k": 1,  "memory_k": 16, "label": "task_sk1_mk16"},
    {"memory": "Convo",      "chain_type": "task", "summary_k": 5,  "memory_k": 4,  "label": "task_sk5_mk4"},
    {"memory": "Convo",      "chain_type": "task", "summary_k": 5,  "memory_k": 16, "label": "task_sk5_mk16"},
]

TRAITS_FILE = PERSONA_DIR / "traits_n5.json"
PERSONAS = {
    "weird":          [PERSONA_DIR / "weird"     / "weird_axis.json",     TRAITS_FILE],
    "non_weird_jung": [PERSONA_DIR / "non_weird" / "non_weird_axis.json",
                       PERSONA_DIR / "jung"      / "jung_axis.json",
                       TRAITS_FILE],
}

# BFI-44 reverse-scored items (score = 6 − raw rating)
REVERSE_ITEMS = {
    "A_01", "A_03", "A_06", "A_08",
    "C_02", "C_04", "C_05", "C_09",
    "E_02", "E_05", "E_07",
    "N_02", "N_05", "N_07",
    "O_07", "O_09",
}

# ── run simulations ────────────────────────────────────────────────────────────
frames: list[pl.DataFrame] = []

for persona_name, persona_files in PERSONAS.items():
    for cond in CONDITIONS:
        run_label = f"exp1_2a_{persona_name}_{cond['label']}"
        print(f"\n{'='*60}")
        print(f"  {run_label}")
        print(f"{'='*60}")

        csv_path = RAW_DIR / f"{run_label}.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            df = df.with_columns([
                pl.lit(persona_name).alias("persona"),
                pl.lit(cond["label"]).alias("memory_cond"),
                pl.lit(cond["summary_k"]).alias("summary_k"),
                pl.lit(cond["memory_k"]).alias("memory_k"),
            ])
            frames.append(df)
            print(f"  skipped (CSV exists): {csv_path.name}  ({len(df)} rows)")
            continue

        # Clear stale session tunnel directory so a fresh run can proceed
        session_dir = RAW_DIR / run_label
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"  cleared stale session dir: {session_dir.name}")

        card = ExpCardInit(
            model         = MODEL,
            family        = FAMILY,
            parameters    = PARAMS,
            cogtype       = "custom",
            persona_files = persona_files,
            task_file     = TASK_FILE,
            parser        = "1",          # → DefaultLiteralAgree from task JSON
            memory        = cond["memory"],
            chain_type    = cond["chain_type"],
            summary_k     = cond["summary_k"],
            memory_k      = cond["memory_k"],
            projectname   = run_label,
            proj_dir      = RAW_DIR,
            tunnel_status = "0",
        )
        expcard = ExpCard(card)
        scanner = ScannerModel(expcard=expcard)
        scanner.run()

        df = to_csv(scanner, path=csv_path)

        df = df.with_columns([
            pl.lit(persona_name).alias("persona"),
            pl.lit(cond["label"]).alias("memory_cond"),
            pl.lit(cond["summary_k"]).alias("summary_k"),
            pl.lit(cond["memory_k"]).alias("memory_k"),
        ])
        frames.append(df)
        print(f"  saved {len(df)} rows → {csv_path.name}")

# ── combine ────────────────────────────────────────────────────────────────────
combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "exp1_2a_combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} → {combined_path}")

# ── scoring ────────────────────────────────────────────────────────────────────
if "resp_rating" not in combined.columns:
    print("WARNING: resp_rating column missing — check parser. Skipping scoring and figures.")
else:
    # Add trait label from trcode prefix; apply reverse scoring
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

    # Per-sim, per-trait mean score
    trait_scores = (
        scored
        .group_by(["sim_idx", "persona", "memory_cond", "summary_k", "memory_k", "trait"])
        .agg(pl.col("rating_scored").mean().alias("mean_score"))
        .sort(["persona", "memory_cond", "trait", "sim_idx"])
    )
    trait_scores.write_csv(HERE / "analysis" / "trait_scores.csv")
    print("Saved analysis/trait_scores.csv")

    trait_scores_pd = trait_scores.to_pandas()

    TRAITS     = ["E", "A", "C", "N", "O"]
    TRAIT_NAMES = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
                   "N": "Neuroticism",  "O": "Openness"}
    COND_LABELS = [c["label"] for c in CONDITIONS]
    persona_colors = {"weird": "steelblue", "non_weird_jung": "tomato"}

    # ── Figure 1: per-trait mean score across conditions ───────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, trait in zip(axes, TRAITS):
        sub = trait_scores_pd[trait_scores_pd["trait"] == trait]
        x_pos = np.arange(len(COND_LABELS))
        offset = {"weird": -0.18, "non_weird_jung": 0.18}
        rng = np.random.default_rng(42)
        for p_name in ("weird", "non_weird_jung"):
            p_sub = sub[sub["persona"] == p_name]
            for xi, c_label in enumerate(COND_LABELS):
                vals = p_sub[p_sub["memory_cond"] == c_label]["mean_score"].values.astype(float)
                if vals.size == 0:
                    continue
                jx = xi + offset[p_name] + rng.uniform(-0.06, 0.06, size=vals.size)
                ax.scatter(jx, vals, color=persona_colors[p_name], s=50, alpha=0.8,
                           edgecolors="white", linewidths=0.4,
                           label=p_name.replace("_", "-") if xi == 0 else "_nolegend_")
                ax.plot([xi + offset[p_name] - 0.1, xi + offset[p_name] + 0.1],
                        [np.mean(vals), np.mean(vals)],
                        color=persona_colors[p_name], lw=2)
        ax.set_title(TRAIT_NAMES[trait], fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace("_", "\n") for c in COND_LABELS], fontsize=6)
        ax.set_ylim(0.5, 5.5)
        if ax is axes[0]:
            ax.set_ylabel("Mean scored rating (1–5)")
        ax.legend(fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Exp 1.2a — BFI-44 per-trait scores across conditions\n"
        f"model: {MODEL}  ·  n=5 per cell",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_trait_per_cell.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/per_trait_per_cell.png")

    # ── Figure 2: chained vs control contrast ──────────────────────────────────
    ctrl   = trait_scores_pd[trait_scores_pd["memory_cond"] == "item_ST"]
    chained = trait_scores_pd[trait_scores_pd["memory_cond"] != "item_ST"]

    ctrl_mean = (
        ctrl.groupby(["persona", "trait"])["mean_score"].mean().reset_index()
        .rename(columns={"mean_score": "ctrl_mean"})
    )
    chain_mean = (
        chained.groupby(["persona", "memory_cond", "trait"])["mean_score"].mean().reset_index()
    )
    merged = chain_mean.merge(ctrl_mean, on=["persona", "trait"])
    merged["delta"] = merged["mean_score"] - merged["ctrl_mean"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    chain_conds = [c["label"] for c in CONDITIONS if c["label"] != "item_ST"]
    for ax, trait in zip(axes, TRAITS):
        sub = merged[merged["trait"] == trait]
        x_pos = np.arange(len(chain_conds))
        offset = {"weird": -0.18, "non_weird_jung": 0.18}
        for p_name in ("weird", "non_weird_jung"):
            p_sub = sub[sub["persona"] == p_name]
            vals = [p_sub[p_sub["memory_cond"] == c]["delta"].values for c in chain_conds]
            means = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x_pos + offset[p_name], means, width=0.32,
                   color=persona_colors[p_name], alpha=0.75,
                   label=p_name.replace("_", "-"))
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_title(TRAIT_NAMES[trait], fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace("_", "\n") for c in chain_conds], fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("Δ from item_ST control (scored rating)")
        ax.legend(fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Exp 1.2a — Chained vs control delta per trait\n"
        f"model: {MODEL}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trial_chain_vs_control_contrast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/trial_chain_vs_control_contrast.png")

    # ── Figure 3: OCEAN radar per persona (averaged across chain conditions) ───
    from matplotlib.patches import FancyArrowPatch

    persona_profiles = {}
    for p_name in ("weird", "non_weird_jung"):
        p_sub = trait_scores_pd[
            (trait_scores_pd["persona"] == p_name) &
            (trait_scores_pd["memory_cond"] != "item_ST")
        ]
        profile = [p_sub[p_sub["trait"] == t]["mean_score"].mean() for t in TRAITS]
        persona_profiles[p_name] = profile

    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for p_name, color in persona_colors.items():
        vals = persona_profiles.get(p_name, [3] * 5)
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, color=color, linewidth=2,
                label=p_name.replace("_", "-"))
        ax.fill(angles, vals_plot, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), [TRAIT_NAMES[t] for t in TRAITS], fontsize=10)
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(["2", "3", "4", "5"], fontsize=7)
    ax.set_title(
        f"Exp 1.2a — OCEAN radar (mean over chain conditions)\nmodel: {MODEL}",
        pad=20, fontsize=11,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "radar_per_persona.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/radar_per_persona.png")

print(f"\nExp 1.2a complete.  {combined.shape[0]} trial records across {len(PERSONAS) * len(CONDITIONS)} conditions.")
