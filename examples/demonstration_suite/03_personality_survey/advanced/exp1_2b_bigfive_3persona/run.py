"""Exp 1.2b — BFI-44: Persona(3) × SummaryK(2) × MemoryK(2) + Control.

Identical to Exp 1.2a except a third persona condition is added: Zhang.
Run from the psychscanner project root:
    conda activate psyscan
    python examples/demonstration_suite/03_personality_survey/advanced/exp1_2b_bigfive_3persona/run.py

Design
------
Persona (3): WEIRD, non-WEIRD+Jung, Zhang
Memory conditions (5 per persona): same as Exp 1.2a
  item_ST (control), task_sk1_mk4, task_sk1_mk16, task_sk5_mk4, task_sk5_mk16

Zhang condition: 5 individuals sampled from Zhang et al. (persona_200.json),
  used without cultural-axis framing.  persona_files=[zhang_axis] → 5 participants.

Total: 3 personas × 5 conditions × 5 simulations = 3 750 trial records (75 × 50 items = 3750; wait, 44 items × 5 sims × 15 = 3300).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[4]
ADV_ROOT  = HERE.parents[0]

TASK_FILE   = REPO_ROOT / "examples" / "tasks" / "bfi44.json"
PERSONA_DIR = ADV_ROOT / "personas"
RAW_DIR     = HERE / "raw"
PROC_DIR    = HERE / "processed"
FIG_DIR     = HERE / "figures"

for d in (RAW_DIR, PROC_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
    # axis(1) × traits(5) = 5 participants
    "weird":          [PERSONA_DIR / "weird"     / "weird_axis.json",     TRAITS_FILE],
    # axis(1) × jung(1) × traits(5) = 5 participants
    "non_weird_jung": [PERSONA_DIR / "non_weird" / "non_weird_axis.json",
                       PERSONA_DIR / "jung"      / "jung_axis.json",
                       TRAITS_FILE],
    # zhang(5) alone = 5 participants (Zhang personas are complete individual descriptions)
    "zhang":          [PERSONA_DIR / "zhang" / "zhang_axis.json"],
}

REVERSE_ITEMS = {
    "A_01", "A_03", "A_06", "A_08",
    "C_02", "C_04", "C_05", "C_09",
    "E_02", "E_05", "E_07",
    "N_02", "N_05", "N_07",
    "O_07", "O_09",
}

frames: list[pl.DataFrame] = []

for persona_name, persona_files in PERSONAS.items():
    for cond in CONDITIONS:
        run_label = f"exp1_2b_{persona_name}_{cond['label']}"
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

        session_dir = RAW_DIR / run_label
        if session_dir.exists():
            shutil.rmtree(session_dir)

        card = ExpCardInit(
            model         = MODEL,
            family        = FAMILY,
            parameters    = PARAMS,
            cogtype       = "custom",
            persona_files = persona_files,
            task_file     = TASK_FILE,
            parser        = "1",
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

combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "exp1_2b_combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} → {combined_path}")

if "resp_rating" not in combined.columns:
    print("WARNING: resp_rating missing. Skipping figures.")
else:
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
    persona_colors = {"weird": "steelblue", "non_weird_jung": "tomato", "zhang": "forestgreen"}

    # Figure 1: OCEAN radar per persona (averaged across chain conditions)
    from matplotlib.patches import FancyArrowPatch
    persona_profiles = {}
    for p_name in persona_colors:
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
        vals = persona_profiles.get(p_name, [3] * 5) + [persona_profiles.get(p_name, [3]*5)[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=p_name.replace("_", "-"))
        ax.fill(angles, vals, color=color, alpha=0.12)
    ax.set_thetagrids(np.degrees(angles[:-1]), [TRAIT_NAMES[t] for t in TRAITS], fontsize=10)
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(["2", "3", "4", "5"], fontsize=7)
    ax.set_title(f"Exp 1.2b — OCEAN radar (3 personas)\nmodel: {MODEL}", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "radar_3personas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/radar_3personas.png")

    # Figure 2: per-trait contrast across all 3 personas × memory conditions
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, trait in zip(axes, TRAITS):
        sub = trait_scores_pd[trait_scores_pd["trait"] == trait]
        x_pos = np.arange(len(COND_LABELS))
        p_list = list(persona_colors.keys())
        offsets = {p: (i - 1) * 0.22 for i, p in enumerate(p_list)}
        rng = np.random.default_rng(42)
        for p_name, color in persona_colors.items():
            p_sub = sub[sub["persona"] == p_name]
            for xi, c_label in enumerate(COND_LABELS):
                vals = p_sub[p_sub["memory_cond"] == c_label]["mean_score"].values.astype(float)
                if vals.size == 0:
                    continue
                jx = xi + offsets[p_name] + rng.uniform(-0.04, 0.04, size=vals.size)
                ax.scatter(jx, vals, color=color, s=45, alpha=0.8,
                           edgecolors="white", linewidths=0.4,
                           label=p_name.replace("_","-") if xi == 0 else "_nolegend_")
                ax.plot([xi + offsets[p_name] - 0.08, xi + offsets[p_name] + 0.08],
                        [np.mean(vals), np.mean(vals)], color=color, lw=2)
        ax.set_title(TRAIT_NAMES[trait], fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace("_", "\n") for c in COND_LABELS], fontsize=6)
        ax.set_ylim(0.5, 5.5)
        if ax is axes[0]:
            ax.set_ylabel("Mean scored rating (1–5)")
        ax.legend(fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Exp 1.2b — BFI-44 per-trait (3 personas)\nmodel: {MODEL}", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_trait_3personas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/per_trait_3personas.png")

print(f"\nExp 1.2b complete.  {combined.shape[0]} trial records across {len(PERSONAS) * len(CONDITIONS)} conditions.")
