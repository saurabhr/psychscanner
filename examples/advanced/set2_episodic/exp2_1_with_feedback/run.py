"""Exp 2.1 — Paired-Associate Learning (PAL-50): Memory(3) × Similarity(5) × With Feedback.

Run from the psychscanner project root:
    conda activate psyscan
    python examples/advanced/set2_episodic/exp2_1_with_feedback/run.py

Design
------
Task: PAL-50 — 50 word pairs at 5 semantic similarity levels (10 per level).
Memory architectures (3 levels):
  item_ST   — chain_type=item, memory=SingleTurn (no cross-trial memory, baseline)
  task_Cv   — chain_type=task, memory=Convo (full encoding-then-test in one thread)
  trial_Cv  — chain_type=trial, memory=Convo (encoding+test per pair in one thread)

Similarity levels: L0=0.0, L25=0.25, L50=0.50, L75=0.75, L100=1.0
Simulations: nsim=5 per condition (cogtype='no', no persona conditioning)
Feedback: Correct/Incorrect + correct answer revealed after each test trial.

Outputs
-------
raw/exp2_1_{cond}.csv              — 500 rows each (5 sims × 100 trials)
processed/exp2_1_combined.csv      — 1 500 rows total
figures/accuracy_x_similarity.png
figures/confidence_x_accuracy.png
figures/externalization_bias.png
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

from psychscanner import ExpCard, ExpCardInit, FeedbackBase, ScannerModel, to_csv
from psychscanner.parsers import get_parser

# ── paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[3]   # psychscanner/
TASK_FILE = REPO_ROOT / "examples" / "tasks" / "pal50.json"
RAW_DIR   = HERE / "raw"
PROC_DIR  = HERE / "processed"
FIG_DIR   = HERE / "figures"

for d in (RAW_DIR, PROC_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────────────────
MODEL  = "smollm2:360m-instruct-fp16"
FAMILY = "ollama"
PARAMS = {"temperature": 0}

# Word pairs used to construct the trial_Cv task (must match pal50.json)
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


# ── feedback ───────────────────────────────────────────────────────────────────
class PALFeedback(FeedbackBase):
    """Three-part feedback: Correct/Incorrect, recalled word, and correct answer."""

    def on_response(self, trial: dict, response: dict) -> str | None:
        if trial.get("phase") != "test":
            return None

        target = trial.get("word2", "").lower().strip()
        recalled = response.get("recalled_word", "").lower().strip() if response else ""
        correct = recalled == target

        if correct:
            fb = {
                "result": "Correct",
                "feedback": (
                    f"You correctly recalled that '{trial['word1']}' was paired with '{trial['word2']}'."
                ),
            }
        else:
            fb = {
                "result": "Incorrect",
                "your_recall": response.get("recalled_word", "") if response else "",
                "correct_answer": (
                    f"The word paired with '{trial['word1']}' was '{trial['word2']}'."
                ),
            }
        return json.dumps(fb)


# ── build trial_Cv task dict (50 dual-stimulus items) ─────────────────────────
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
                    "fb": True,
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
        "task_file":  TRIAL_CV_TASK,   # dict, not path
        "trial_parsers": [None, PairedAssociateRecall],
    },
]

# ── run simulations ────────────────────────────────────────────────────────────
frames: list[pl.DataFrame] = []

for cond in CONDITIONS:
    run_label = f"exp2_1_{cond['label']}"
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
        model          = MODEL,
        family         = FAMILY,
        parameters     = PARAMS,
        cogtype        = "no",
        nsim           = 5,
        task_file      = cond["task_file"],
        parser         = None,          # per-stimulus parser embedded in item dicts
        trial_parsers  = cond["trial_parsers"],
        memory         = cond["memory"],
        chain_type     = cond["chain_type"],
        feedback       = True,
        feedback_fn    = PALFeedback,
        projectname    = run_label,
        proj_dir       = RAW_DIR,
        tunnel_status  = "0",
    )
    expcard = ExpCard(card)
    scanner = ScannerModel(expcard=expcard)
    scanner.run()

    df = to_csv(scanner, path=csv_path)
    df = df.with_columns(pl.lit(cond["label"]).alias("memory_cond"))
    frames.append(df)
    print(f"  saved {len(df)} rows → {csv_path.name}")

# ── combine ────────────────────────────────────────────────────────────────────
combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "exp2_1_combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} → {combined_path}")

# ── scoring ────────────────────────────────────────────────────────────────────
test_df_exists = "resp_recalled_word" in combined.columns and "resp_confidence" in combined.columns
if not test_df_exists:
    print("WARNING: resp_recalled_word or resp_confidence missing — check parser. Skipping figures.")
else:
    # Keep only test-phase rows; add accuracy and similarity
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

    test_pd = test_rows.to_pandas()
    COND_LABELS = [c["label"] for c in CONDITIONS]
    cond_colors = {"item_ST": "gray", "task_Cv": "steelblue", "trial_Cv": "tomato"}
    sim_vals = [0.0, 0.25, 0.50, 0.75, 1.0]

    # ── Figure 1: accuracy × similarity × memory condition ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond_label in COND_LABELS:
        sub = test_pd[test_pd["memory_cond"] == cond_label]
        acc = sub.groupby("similarity")["correct"].mean()
        ax.plot(acc.index, acc.values, marker="o", label=cond_label,
                color=cond_colors[cond_label], linewidth=2, markersize=6)
    ax.set_xlabel("Semantic similarity level")
    ax.set_ylabel("Proportion correct")
    ax.set_xticks(sim_vals)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Exp 2.1 — PAL accuracy × similarity × memory\nmodel: {MODEL}, n=5 per condition")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_x_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/accuracy_x_similarity.png")

    # ── Figure 2: confidence × accuracy calibration ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, cond_label in zip(axes, COND_LABELS):
        sub = test_pd[test_pd["memory_cond"] == cond_label]
        cal = sub.groupby("resp_confidence")["correct"].mean()
        ax.bar(cal.index.astype(int), cal.values, color=cond_colors[cond_label], alpha=0.75)
        ax.plot([1, 6], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Confidence (1–6)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(cond_label)
        if ax is axes[0]:
            ax.set_ylabel("Proportion correct")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Exp 2.1 — Confidence calibration by condition\nmodel: {MODEL}", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "confidence_x_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/confidence_x_accuracy.png")

print(f"\nExp 2.1 complete.  {combined.shape[0]} trial records across {len(CONDITIONS)} conditions.")
