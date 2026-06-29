"""Exp 3.4 — BPT-10: non-OCR-VLM × No Feedback.

Also generates exp3_all_combined.png if all 4 sub-experiments have completed.
Run from the psychscanner project root:
    conda activate psyscan
    python examples/advanced/set3_multimodal/exp3_4_nonocr_nofeedback/run.py
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
from _shared import build_task, BPTFeedback

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

HERE      = Path(__file__).parent
SET3_ROOT = Path(__file__).parents[1]
RAW_DIR   = HERE / "raw"
PROC_DIR  = HERE / "processed"
FIG_DIR   = HERE / "figures"
ANA_DIR   = HERE / "analysis"

for d in (RAW_DIR, PROC_DIR, FIG_DIR, ANA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL  = "mistralai/mistral-nemo-12b-instruct:free"
FAMILY = "openrouter"
PARAMS = {"temperature": 0}

RUN_LABEL = "exp3_4_nonocr_nofeedback"
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

    task_dict = build_task(ocr=False, feedback=False)

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

# ── Per-exp figures ───────────────────────────────────────────────────────────

probe_df = df.filter(pl.col("phase") == "probe") if "phase" in df.columns else df

if "resp_recalled_number" in probe_df.columns:
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
    ax.plot(by_pos.index, by_pos.values, "o-", color="tomato", linewidth=2, markersize=6)
    ax.axhline(1 / 90, color="grey", linestyle="--", alpha=0.5, label="chance (1/90)")
    ax.set_xlabel("Serial position of probe target")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Exp 3.4 — Lost-in-the-Middle\nnon-OCR + No Feedback | model: {MODEL}", fontsize=10
    )
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
               color=["tomato", "orange"][:len(by_mod)], capsize=5, alpha=0.8)
        ax.set_xlabel("Probe target modality")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Exp 3.4 — Attentional Residue\nnon-OCR + No Feedback | model: {MODEL}",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "modality_switch_effect.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved figures for {RUN_LABEL}")

# ── Cross-experiment combined figure (generated last) ────────────────────────

EXP_META = [
    ("exp3_1_ocr_feedback",    "OCR + FB",    "steelblue",  "-"),
    ("exp3_2_ocr_nofeedback",  "OCR − FB",    "steelblue",  "--"),
    ("exp3_3_nonocr_feedback", "non-OCR + FB","tomato",     "-"),
    ("exp3_4_nonocr_nofeedback","non-OCR − FB","tomato",    "--"),
]

all_frames = []
for label, _, _, _ in EXP_META:
    p = SET3_ROOT / label.replace("exp3_", f"exp3_")[5:] / "raw" / f"{label}.csv"
    # build paths relative to set3 root
    subdir_map = {
        "exp3_1_ocr_feedback":     "exp3_1_ocr_feedback",
        "exp3_2_ocr_nofeedback":   "exp3_2_ocr_nofeedback",
        "exp3_3_nonocr_feedback":  "exp3_3_nonocr_feedback",
        "exp3_4_nonocr_nofeedback":"exp3_4_nonocr_nofeedback",
    }
    csv_p = SET3_ROOT / subdir_map[label] / "raw" / f"{label}.csv"
    if csv_p.exists():
        sub = pl.read_csv(csv_p)
        sub = sub.with_columns(pl.lit(label).alias("experiment"))
        all_frames.append(sub)

if len(all_frames) == 4:
    combined = pl.concat(all_frames, how="diagonal")
    combined.write_csv(SET3_ROOT / "exp3_all_combined.csv")
    print(f"Saved set3_multimodal/exp3_all_combined.csv")

    if "resp_recalled_number" in combined.columns and "serial_position" in combined.columns:
        combined_pd = combined.to_pandas()
        combined_pd["correct"] = (
            combined_pd["resp_recalled_number"].astype("Int64")
            == combined_pd["number"].astype("Int64")
        ).astype(float)
        combined_pd = combined_pd[combined_pd["phase"] == "probe"] if "phase" in combined_pd.columns else combined_pd

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for (label, nice_label, color, ls), _, _, _ in zip(EXP_META, *([EXP_META]*3)):
            pass
        for label, nice_label, color, ls in EXP_META:
            sub = combined_pd[combined_pd["experiment"] == label]
            if sub.empty:
                continue
            by_pos = sub.groupby("serial_position")["correct"].mean().sort_index()
            axes[0].plot(by_pos.index, by_pos.values, color=color, linestyle=ls,
                         linewidth=2, marker="o", markersize=5, label=nice_label)

        axes[0].axhline(1 / 90, color="grey", linestyle=":", alpha=0.5)
        axes[0].set_xlabel("Serial position")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Lost-in-the-Middle: all 4 sub-experiments", fontsize=10)
        axes[0].set_xticks(range(1, 11))
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)
        axes[0].spines[["top", "right"]].set_visible(False)

        if "probe_target_modality" in combined_pd.columns:
            x_labels = sorted(combined_pd["probe_target_modality"].dropna().unique())
            x_pos = np.arange(len(x_labels))
            w = 0.18
            for i, (label, nice_label, color, ls) in enumerate(EXP_META):
                sub = combined_pd[combined_pd["experiment"] == label]
                if sub.empty:
                    continue
                by_mod = sub.groupby("probe_target_modality")["correct"].mean()
                heights = [by_mod.get(xl, 0) for xl in x_labels]
                offset = (i - 1.5) * w
                axes[1].bar(x_pos + offset, heights, width=w, color=color,
                             alpha=(0.9 if ls == "-" else 0.5), label=nice_label)
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(x_labels)
            axes[1].set_xlabel("Probe target modality")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Attentional Residue: all 4 sub-experiments", fontsize=10)
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].legend(fontsize=8)
            axes[1].grid(axis="y", linestyle="--", alpha=0.35)
            axes[1].spines[["top", "right"]].set_visible(False)

        fig.suptitle("Set 3 Multimodal — Behavioral Profiling (VLM-type × Feedback)", fontsize=11)
        plt.tight_layout()
        combined_fig_path = SET3_ROOT / "figures" / "exp3_all_combined.png"
        combined_fig_path.parent.mkdir(exist_ok=True)
        fig.savefig(combined_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved set3_multimodal/figures/exp3_all_combined.png")
else:
    print(f"Cross-experiment figure: {len(all_frames)}/4 sub-experiments complete — skipping.")

print(f"\nExp 3.4 complete.  {len(df)} total rows.")

# ── Regenerate comprehensive figure ──────────────────────────────────────────
try:
    sys.path.insert(0, str(HERE.parent / "figures"))
    import make_figures as _mf
    _mf.make_figure()
except Exception as _e:
    print(f"  make_figures skipped: {_e}")
