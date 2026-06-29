"""Exp 2.2 — PAL-50: Memory(3) × Similarity(5) × No Feedback.

Identical to Exp 2.1 except feedback=False.
Run from the psychscanner project root:
    conda activate psyscan
    python examples/advanced/set2_episodic/exp2_2_no_feedback/run.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv
from psychscanner.parsers import get_parser

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
TASK_FILE = REPO_ROOT / "examples" / "tasks" / "pal50.json"
RAW_DIR   = HERE / "raw"
PROC_DIR  = HERE / "processed"
FIG_DIR   = HERE / "figures"

for d in (RAW_DIR, PROC_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL  = "smollm2:360m-instruct-fp16"
FAMILY = "ollama"
PARAMS = {"temperature": 0}

PAIRS = {
    "L0":   [("cat","umbrella"),("moon","pencil"),("tree","sock"),
              ("river","wallet"),("bird","hammer"),("dog","ladder"),
              ("sky","carrot"),("book","engine"),("house","guitar"),("sun","kettle")],
    "L25":  [("lion","tiger"),("rose","daisy"),("oak","pine"),
              ("sofa","table"),("apple","orange"),("wrench","saw"),
              ("blue","red"),("doctor","teacher"),("piano","drum"),("sheep","goat")],
    "L50":  [("bee","honey"),("fire","smoke"),("pen","ink"),
              ("arrow","target"),("seed","plant"),("winter","snow"),
              ("kitchen","cook"),("hospital","nurse"),("spider","web"),("key","lock")],
    "L75":  [("salt","pepper"),("bread","butter"),("black","white"),
              ("night","day"),("knife","fork"),("cup","saucer"),
              ("stamp","envelope"),("needle","thread"),("chalk","board"),("water","wave")],
    "L100": [("big","large"),("happy","joyful"),("fast","quick"),
              ("small","tiny"),("dark","dim"),("cold","chilly"),
              ("warm","cozy"),("bright","radiant"),("afraid","scared"),("angry","furious")],
}
SIM_MAP = {"L0": 0.0, "L25": 0.25, "L50": 0.50, "L75": 0.75, "L100": 1.0}

PairedAssociateRecall = get_parser("PairedAssociateRecall")


def _build_trial_cv_task() -> dict:
    base = json.loads(TASK_FILE.read_text())
    base["chain_type"] = "trial"
    base["items"] = {}
    for level, pairs in PAIRS.items():
        for i, (w1, w2) in enumerate(pairs, 1):
            trcode = f"{level}_pair_{i:02d}"
            base["items"][trcode] = [
                {
                    "trcode": trcode,
                    "stimulus": (
                        f"STUDY PHASE — Please remember this word pair: "
                        f"the word '{w1}' is paired with '{w2}'."
                    ),
                    "phase": "encoding",
                    "word1": w1, "word2": w2,
                    "similarity": SIM_MAP[level],
                    "parser": None,
                    "fb": False,
                },
                {
                    "trcode": trcode,
                    "stimulus": (
                        f"TEST PHASE — What word was paired with '{w1}' "
                        f"during the study phase?"
                    ),
                    "phase": "test",
                    "word1": w1, "word2": w2,
                    "similarity": SIM_MAP[level],
                    "parser": "PairedAssociateRecall",
                    "fb": False,   # no feedback in Exp 2.2
                },
            ]
    return base


TRIAL_CV_TASK = _build_trial_cv_task()

CONDITIONS = [
    {
        "label":      "item_ST",
        "memory":     "SingleTurn",
        "chain_type": "item",
        "task_file":  TASK_FILE,
        "trial_parsers": None,
    },
    {
        "label":      "task_Cv",
        "memory":     "Convo",
        "chain_type": "task",
        "task_file":  TASK_FILE,
        "trial_parsers": None,
    },
    {
        "label":      "trial_Cv",
        "memory":     "Convo",
        "chain_type": "trial",
        "task_file":  TRIAL_CV_TASK,
        "trial_parsers": [None, PairedAssociateRecall],
    },
]

frames: list[pl.DataFrame] = []

for cond in CONDITIONS:
    run_label = f"exp2_2_{cond['label']}"
    print(f"\n{'='*60}")
    print(f"  {run_label}")
    print(f"{'='*60}")

    csv_path = RAW_DIR / f"{run_label}.csv"
    if csv_path.exists():
        df = pl.read_csv(csv_path)
        df = df.with_columns(pl.lit(cond["label"]).alias("memory_cond"))
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
        cogtype       = "no",
        nsim          = 5,
        task_file     = cond["task_file"],
        parser        = None,
        trial_parsers = cond["trial_parsers"],
        memory        = cond["memory"],
        chain_type    = cond["chain_type"],
        feedback      = False,
        projectname   = run_label,
        proj_dir      = RAW_DIR,
        tunnel_status = "0",
    )
    expcard = ExpCard(card)
    scanner = ScannerModel(expcard=expcard)
    scanner.run()

    df = to_csv(scanner, path=csv_path)
    df = df.with_columns(pl.lit(cond["label"]).alias("memory_cond"))
    frames.append(df)
    print(f"  saved {len(df)} rows → {csv_path.name}")

combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "exp2_2_combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} → {combined_path}")

if "resp_recalled_word" in combined.columns:
    test_rows = (
        combined
        .filter(
            pl.col("trcode").str.contains("_tst_") |
            pl.col("trcode").str.contains("_pair_")
        )
        .filter(pl.col("resp_recalled_word").is_not_null())
        .with_columns([
            pl.col("trcode").str.extract(r"^(L\d+)", 1).alias("level"),
            pl.when(
                pl.col("resp_recalled_word").str.to_lowercase() ==
                pl.col("word2").str.to_lowercase()
            ).then(1).otherwise(0).alias("correct"),
        ])
        .with_columns(
            pl.col("level").replace(
                {"L0": 0.0, "L25": 0.25, "L50": 0.50, "L75": 0.75, "L100": 1.0}
            ).cast(pl.Float64).alias("similarity")
        )
    )
    test_rows.write_csv(HERE / "analysis" / "test_scored.csv")
    print("Saved analysis/test_scored.csv")

    # Cross-sub-exp contrast with Exp 2.1 (load if exists)
    exp2_1_scored = (
        HERE.parent / "exp2_1_with_feedback" / "analysis" / "test_scored.csv"
    )
    if exp2_1_scored.exists():
        df_21 = pl.read_csv(exp2_1_scored).with_columns(pl.lit("with_fb").alias("feedback"))
        df_22 = test_rows.with_columns(pl.lit("no_fb").alias("feedback"))
        contrast = pl.concat([df_21, df_22], how="diagonal")

        sim_vals = [0.0, 0.25, 0.50, 0.75, 1.0]
        COND_LABELS = [c["label"] for c in CONDITIONS]
        cond_colors = {"item_ST": "gray", "task_Cv": "steelblue", "trial_Cv": "tomato"}
        contrast_pd = contrast.to_pandas()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, cond_label in zip(axes, COND_LABELS):
            sub = contrast_pd[contrast_pd["memory_cond"] == cond_label]
            for fb_label, ls in [("with_fb", "-"), ("no_fb", "--")]:
                fb_sub = sub[sub["feedback"] == fb_label]
                acc = fb_sub.groupby("similarity")["correct"].mean()
                ax.plot(
                    acc.index, acc.values,
                    marker="o", linestyle=ls,
                    color=cond_colors[cond_label],
                    label=fb_label, linewidth=2,
                )
            ax.set_title(cond_label)
            ax.set_xticks(sim_vals)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Similarity")
            if ax is axes[0]:
                ax.set_ylabel("Proportion correct")
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"Exp 2.1 vs 2.2 — Feedback contrast × memory condition\nmodel: {MODEL}",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp2_1_vs_2_2_feedback_contrast.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved figures/exp2_1_vs_2_2_feedback_contrast.png")

print(f"\nExp 2.2 complete.  {combined.shape[0]} trial records across {len(CONDITIONS)} conditions.")
