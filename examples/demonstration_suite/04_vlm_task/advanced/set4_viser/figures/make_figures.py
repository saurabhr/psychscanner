"""Set 4 — VISER Counting — Results figure.

Generates:
  set4_viser/figures/exp4_results_figure.png

Bar plot of accuracy for the four conditions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SET4 = Path(__file__).parents[1]
sys.path.insert(0, str(SET4))

try:
    import polars as pl
    _HAVE_POLARS = True
except ImportError:
    _HAVE_POLARS = False

# ── Condition metadata ────────────────────────────────────────────────────────

CONDITIONS = [
    dict(label="exp4_1_ocr_viser",     nice="OCR + VISER",     color="steelblue"),
    dict(label="exp4_2_ocr_no_viser",   nice="OCR - VISER",     color="steelblue"),
    dict(label="exp4_3_nonocr_viser",  nice="Non-OCR + VISER", color="tomato"),
    dict(label="exp4_4_nonocr_no_viser",nice="Non-OCR - VISER", color="tomato"),
]


def load_accuracy(c: dict) -> float | None:
    csv_p = SET4 / c["label"] / "raw" / f"{c['label']}.csv"
    if not csv_p.exists() or not _HAVE_POLARS:
        return None
    df = pl.read_csv(csv_p)
    if "phase" in df.columns:
        df = df.filter(pl.col("phase") == "probe")
    if df.is_empty():
        return None
    if "resp_recalled_number" not in df.columns or "target_count" not in df.columns:
        return None
    correct = (
        df["resp_recalled_number"].cast(pl.Int64, strict=False)
        == df["target_count"].cast(pl.Int64, strict=False)
    ).sum()
    total = len(df)
    return correct / total if total > 0 else 0.0


def make_figure():
    accuracies = []
    labels = []
    colors = []
    for c in CONDITIONS:
        acc = load_accuracy(c)
        if acc is not None:
            accuracies.append(acc)
            labels.append(c["nice"])
            colors.append(c["color"])
        else:
            accuracies.append(0)
            labels.append(c["nice"] + " (no data)")
            colors.append("grey")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Set 4 — VISER Counting Accuracy\nModel: nvidia/nemotron-nano-12b-v2-vl:free", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(SET4 / "figures" / "exp4_results_figure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/exp4_results_figure.png")


if __name__ == "__main__":
    make_figure()