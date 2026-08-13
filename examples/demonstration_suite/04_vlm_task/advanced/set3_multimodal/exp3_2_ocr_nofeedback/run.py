"""Exp 3.2 — BPT-10: OCR-VLM × No Feedback.

Identical to Exp 3.1 except feedback=False.
Run from the psychscanner project root:
    conda activate psyscan
    python examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal/exp3_2_ocr_nofeedback/run.py
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

load_dotenv(Path(__file__).parents[6] / ".env")

sys.path.insert(0, str(Path(__file__).parents[1]))
from _shared import build_task, BPTFeedback

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

HERE     = Path(__file__).parent
RAW_DIR  = HERE / "raw"
PROC_DIR = HERE / "processed"
FIG_DIR  = HERE / "figures"
ANA_DIR  = HERE / "analysis"

for d in (RAW_DIR, PROC_DIR, FIG_DIR, ANA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL  = "nvidia/nemotron-nano-12b-v2-vl:free"
FAMILY = "openrouter"
PARAMS = {"temperature": 0}

RUN_LABEL = "exp3_2_ocr_nofeedback"
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

    task_dict = build_task(ocr=True, feedback=False)

    card = ExpCardInit(
        model         = MODEL,
        family        = FAMILY,
        parameters    = PARAMS,
        cogtype       = "no",
        nsim          = 5,
        task_file     = task_dict,
        parser        = None,
        memory        = "Convo",
        chain_type    = "task",
        summary_k     = 0,
        memory_k      = -1,
        feedback      = False,
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
         == pl.col("number").cast(pl.Int64, strict=False))
        .cast(pl.Int64)
        .alias("correct"),
    ])
    scored.write_csv(ANA_DIR / "probe_scored.csv")

    scored_pd = scored.to_pandas()

    fig, ax = plt.subplots(figsize=(7, 4))
    by_pos = scored_pd.groupby("serial_position")["correct"].mean()
    ax.plot(by_pos.index, by_pos.values, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(1 / 90, color="grey", linestyle="--", alpha=0.5, label="chance (1/90)")
    ax.set_xlabel("Serial position of probe target")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Exp 3.2 — Lost-in-the-Middle\nOCR-VLM + No Feedback | model: {MODEL}", fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "serial_position_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    if "probe_target_modality" in scored_pd.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        by_mod = scored_pd.groupby("probe_target_modality")["correct"].agg(["mean", "sem"])
        ax.bar(by_mod.index, by_mod["mean"], yerr=by_mod["sem"],
               color=["steelblue", "tomato"][:len(by_mod)], capsize=5, alpha=0.8)
        ax.set_xlabel("Probe target modality")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Exp 3.2 — Attentional Residue\nOCR-VLM + No Feedback | model: {MODEL}", fontsize=10
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "modality_switch_effect.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved figures for {RUN_LABEL}")

print(f"\nExp 3.2 complete.  {len(df)} total rows.")

# ── Regenerate comprehensive figure ──────────────────────────────────────────
try:
    sys.path.insert(0, str(HERE.parent / "figures"))
    import make_figures as _mf
    _mf.make_figure()
except Exception as _e:
    print(f"  make_figures skipped: {_e}")
