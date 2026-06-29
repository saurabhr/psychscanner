"""Exp 4.1 — VISER Counting: OCR-VLM × With VISER.

Run from the psychscanner project root:
    conda activate psyscan
    python examples/advanced/set4_viser/exp4_1_ocr_viser/run.py

Model : gemma3:12b  (Ollama local)
Task  : Counting task — 10 scenes × 1 study + 1 probe
VISER : Yes — grid overlay on image + sequential prompt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[4] / ".env")

sys.path.insert(0, str(Path(__file__).parents[1]))
from _shared import build_task, VISERFeedback

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

HERE     = Path(__file__).parent
RAW_DIR  = HERE / "raw"
PROC_DIR = HERE / "processed"
FIG_DIR  = HERE / "figures"
ANA_DIR  = HERE / "analysis"

for d in (RAW_DIR, PROC_DIR, FIG_DIR, ANA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL  = "gemma3:12b"
FAMILY = "ollama"
PARAMS = {"temperature": 0}

RUN_LABEL = "exp4_1_ocr_viser"
CSV_PATH  = RAW_DIR / f"{RUN_LABEL}.csv"

print(f"\n{'='*60}")
print(f"  {RUN_LABEL}")
print(f"{'='*60}")

if CSV_PATH.exists():
    print(f"  skipped (CSV exists): {CSV_PATH.name}")
    df = pl.read_csv(CSV_PATH)
else:
    session_dir = RAW_DIR / RUN_LABEL
    if session_dir.exists():
        shutil.rmtree(session_dir)

    task_dict = build_task(ocr=True, viser=True)

    card = ExpCardInit(
        model         = MODEL,
        family        = FAMILY,
        parameters    = PARAMS,
        cogtype       = "no",
        nsim          = 2,
        task_file     = task_dict,
        parser        = None,
        memory        = "Convo",
        chain_type    = "task",
        summary_k     = 0,
        memory_k      = -1,
        feedback      = True,
        feedback_fn   = VISERFeedback,
        projectname   = RUN_LABEL,
        proj_dir      = RAW_DIR,
        tunnel_status = "0",
    )
    expcard = ExpCard(card)
    scanner = ScannerModel(expcard=expcard)
    scanner.run()

    df = to_csv(scanner, path=CSV_PATH)
    print(f"  saved {len(df)} rows → {CSV_PATH.name}")

# ── Analysis & figures ────────────────────────────────────────────────────────

probe_df = df.filter(pl.col("phase") == "probe") if "phase" in df.columns else df

if "resp_recalled_number" not in probe_df.columns:
    print("WARNING: resp_recalled_number column missing — skipping figures.")
else:
    scored = probe_df.with_columns([
        (pl.col("resp_recalled_number").cast(pl.Int64, strict=False)
         == pl.col("target_count").cast(pl.Int64, strict=False))
        .cast(pl.Int64)
        .alias("correct"),
    ])
    scored.write_csv(ANA_DIR / "probe_scored.csv")
    print(f"Saved analysis/probe_scored.csv  ({len(scored)} rows)")

    accuracy = scored["correct"].mean()
    print(f"Accuracy: {accuracy:.3f}")

    # Figure: accuracy bar
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0], [accuracy], color="steelblue", alpha=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["OCR + VISER"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Exp 4.1 — Counting Accuracy\nModel: {MODEL}", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/accuracy.png")

print(f"\nExp 4.1 complete.  {len(df)} total rows.")

# ── Regenerate comprehensive figure ──────────────────────────────────────────
try:
    sys.path.insert(0, str(HERE.parent / "figures"))
    import make_figures as _mf
    _mf.make_figure()
except Exception as _e:
    print(f"  make_figures skipped: {_e}")