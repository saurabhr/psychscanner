"""Set 3 — Multimodal BPT-10 — Tutorial-09-style results figure.

Generates:
  set3_multimodal/figures/exp3_results_figure.png

Layout (4 rows):
  Row 0  — Aggregate accuracy per condition (bar + scatter, mean ± SE)
  Row 1  — Serial position curves (Lost-in-the-Middle), both VLM types
  Row 2  — Per-sim serial-position trajectories (one panel per condition)
  Row 3  — Attentional Residue: accuracy by probe_target_modality

Run from the psychscanner project root:
    conda activate psyscan
    python examples/advanced/set3_multimodal/figures/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SET3 = Path(__file__).parents[1]
sys.path.insert(0, str(SET3))

try:
    import polars as pl
    import pandas as pd
    _HAVE_DATA_LIBS = True
except ImportError:
    _HAVE_DATA_LIBS = False

# ── Condition metadata ────────────────────────────────────────────────────────

CONDITIONS = [
    dict(label="exp3_1_ocr_feedback",     nice="OCR + FB",     color="steelblue", ls="-",  vlm="OCR",     fb=True),
    dict(label="exp3_2_ocr_nofeedback",   nice="OCR − FB",     color="steelblue", ls="--", vlm="OCR",     fb=False),
    dict(label="exp3_3_nonocr_feedback",  nice="non-OCR + FB", color="tomato",    ls="-",  vlm="non-OCR", fb=True),
    dict(label="exp3_4_nonocr_nofeedback",nice="non-OCR − FB", color="tomato",    ls="--", vlm="non-OCR", fb=False),
]

MODALITY_COLORS = {
    "image":    "#e07b39",
    "text":     "#5b8db8",
    "described":"#9a6fb0",
    "direct":   "#5b8db8",
}

SERIAL_POSITIONS = list(range(1, 11))
IMAGE_POSITIONS  = {1, 3, 5, 7, 9}   # 0-indexed study positions + 1


# ── Data loading ──────────────────────────────────────────────────────────────

def load_condition(c: dict) -> "pd.DataFrame | None":
    csv_p = SET3 / c["label"] / "raw" / f"{c['label']}.csv"
    if not csv_p.exists() or not _HAVE_DATA_LIBS:
        return None
    df = pl.read_csv(csv_p)
    if "phase" in df.columns:
        df = df.filter(pl.col("phase") == "probe")
    if df.is_empty():
        return None
    # score
    if "resp_recalled_number" in df.columns and "number" in df.columns:
        df = df.with_columns(
            (pl.col("resp_recalled_number").cast(pl.Int64, strict=False)
             == pl.col("number").cast(pl.Int64, strict=False))
            .cast(pl.Int64)
            .alias("correct")
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("correct"))
    pdf = df.to_pandas()
    pdf["nice"]  = c["nice"]
    pdf["color"] = c["color"]
    pdf["vlm"]   = c["vlm"]
    pdf["fb"]    = c["fb"]
    return pdf


def _sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# ── Helper: draw "No data yet" placeholder ───────────────────────────────────

def _no_data(ax: plt.Axes, msg: str = "No data yet") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="grey", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def make_figure() -> None:
    frames: dict[str, "pd.DataFrame | None"] = {c["label"]: load_condition(c) for c in CONDITIONS}
    n_loaded = sum(f is not None for f in frames.values())
    print(f"Loaded {n_loaded}/4 sub-experiments.")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(4, 4, figure=fig,
                             height_ratios=[1.1, 1.0, 1.0, 1.0],
                             hspace=0.52, wspace=0.38)

    # Row 0: aggregate bar (full width)
    ax_agg = fig.add_subplot(gs[0, :])

    # Row 1: serial position curves — 2 panels (OCR left, non-OCR right)
    ax_sp_ocr    = fig.add_subplot(gs[1, :2])
    ax_sp_nonocr = fig.add_subplot(gs[1, 2:])

    # Row 2: per-sim trajectories — 4 panels (one per condition)
    ax_traj = [fig.add_subplot(gs[2, i]) for i in range(4)]

    # Row 3: Attentional Residue — 4 panels
    ax_mod = [fig.add_subplot(gs[3, i]) for i in range(4)]

    # ── ROW 0: Aggregate ─────────────────────────────────────────────────────
    x_pos = np.arange(len(CONDITIONS))
    bar_w = 0.55

    for i, (c, cond) in enumerate(zip(CONDITIONS, CONDITIONS)):
        df = frames[cond["label"]]
        if df is None or "correct" not in df.columns or df["correct"].isna().all():
            ax_agg.bar(i, 0, width=bar_w, color=cond["color"], alpha=0.25,
                       edgecolor=cond["color"], linewidth=1.5,
                       linestyle="--" if not cond["fb"] else "-")
            continue

        mean_acc = df["correct"].mean()
        se_acc   = _sem(df["correct"].dropna())

        ax_agg.bar(i, mean_acc, width=bar_w, color=cond["color"],
                   alpha=(0.85 if cond["fb"] else 0.45),
                   edgecolor=cond["color"], linewidth=1.5)
        ax_agg.errorbar(i, mean_acc, yerr=se_acc,
                        fmt="none", color="black", capsize=5, linewidth=1.5)

        # jitter per-sim means
        if "sim_idx" in df.columns:
            sim_means = df.groupby("sim_idx")["correct"].mean()
            jx = np.random.default_rng(42).uniform(-0.12, 0.12, len(sim_means))
            ax_agg.scatter(i + jx, sim_means.values,
                           color="white", edgecolors=cond["color"],
                           s=40, zorder=5, linewidth=1.2, alpha=0.9)

    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels([c["nice"] for c in CONDITIONS], fontsize=10)
    ax_agg.set_ylabel("Mean accuracy", fontsize=10)
    ax_agg.set_ylim(-0.05, 1.15)
    ax_agg.set_title(
        "Exp 3 — BPT-10: Aggregate Recall Accuracy  (VLM-type × Feedback)",
        fontsize=12, fontweight="bold", pad=8
    )
    ax_agg.axhline(1 / 90, color="grey", linestyle=":", linewidth=1, alpha=0.6,
                   label="Chance (1/90)")
    ax_agg.legend(fontsize=8, loc="upper right")
    ax_agg.grid(axis="y", linestyle="--", alpha=0.3)
    ax_agg.spines[["top", "right"]].set_visible(False)

    # Condition legend patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.85, label="OCR + FB"),
        Patch(facecolor="steelblue", alpha=0.45, label="OCR − FB"),
        Patch(facecolor="tomato",    alpha=0.85, label="non-OCR + FB"),
        Patch(facecolor="tomato",    alpha=0.45, label="non-OCR − FB"),
    ]
    ax_agg.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="grey", linestyle=":", label="Chance (1/90)")
    ], fontsize=8, loc="upper right", ncol=3)

    # ── ROW 1: Serial position curves ────────────────────────────────────────
    for ax, vlm_type, title_suffix in [
        (ax_sp_ocr,    "OCR",     "OCR-VLM"),
        (ax_sp_nonocr, "non-OCR", "non-OCR VLM"),
    ]:
        has_data = False
        for cond in CONDITIONS:
            if cond["vlm"] != vlm_type:
                continue
            df = frames[cond["label"]]
            if df is None or "serial_position" not in df.columns:
                continue
            has_data = True
            by_pos = df.groupby("serial_position")["correct"].mean().sort_index()
            ax.plot(by_pos.index, by_pos.values,
                    color=cond["color"], linestyle=cond["ls"],
                    linewidth=2, marker="o", markersize=5, label=cond["nice"],
                    zorder=3)

            # jitter per-sim, per-position scatter
            if "sim_idx" in df.columns:
                rng = np.random.default_rng(7)
                for pos in SERIAL_POSITIONS:
                    sub = df[df["serial_position"] == pos]["correct"].dropna()
                    if len(sub) == 0:
                        continue
                    jx = rng.uniform(-0.15, 0.15, len(sub))
                    ax.scatter(pos + jx, sub.values,
                               color=cond["color"], alpha=0.25, s=18, zorder=2)

        if not has_data:
            _no_data(ax, f"No data yet\n({title_suffix})")
        else:
            # Shade image-slot positions (odd serial positions where image stimuli appear)
            for pos in IMAGE_POSITIONS:
                ax.axvspan(pos - 0.4, pos + 0.4, color="#f0e6d3", alpha=0.4, zorder=0)
            ax.axhline(1 / 90, color="grey", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_xlabel("Serial position of probe target", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_title(
                f"Lost-in-the-Middle — {title_suffix}\n"
                r"(shaded $\rightarrow$ image-slot positions)",
                fontsize=9
            )
            ax.set_xticks(SERIAL_POSITIONS)
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    # ── ROW 2: Per-sim trajectory panels ─────────────────────────────────────
    for i, cond in enumerate(CONDITIONS):
        ax = ax_traj[i]
        df = frames[cond["label"]]
        if df is None or "serial_position" not in df.columns:
            _no_data(ax, cond["nice"])
            ax.set_title(cond["nice"], fontsize=8, color=cond["color"])
            continue

        rng = np.random.default_rng(42)
        if "sim_idx" in df.columns:
            for sim_i, sim_df in df.groupby("sim_idx"):
                by_pos = sim_df.groupby("serial_position")["correct"].mean().sort_index()
                ax.plot(by_pos.index, by_pos.values,
                        color=cond["color"], alpha=0.25, linewidth=1, zorder=2)

        # Mean trajectory
        by_pos_mean = df.groupby("serial_position")["correct"].mean().sort_index()
        ax.plot(by_pos_mean.index, by_pos_mean.values,
                color=cond["color"], linewidth=2.5, marker="o",
                markersize=6, zorder=4, label="Mean")

        # SE shading
        if "sim_idx" in df.columns:
            sim_by_pos = df.groupby(["sim_idx", "serial_position"])["correct"].mean().reset_index()
            se_by_pos  = sim_by_pos.groupby("serial_position")["correct"].std() / np.sqrt(
                sim_by_pos.groupby("serial_position")["correct"].count()
            )
            lo = by_pos_mean - se_by_pos.reindex(by_pos_mean.index).fillna(0)
            hi = by_pos_mean + se_by_pos.reindex(by_pos_mean.index).fillna(0)
            ax.fill_between(by_pos_mean.index, lo, hi, color=cond["color"], alpha=0.15, zorder=1)

        # Shade image-slot positions
        for pos in IMAGE_POSITIONS:
            ax.axvspan(pos - 0.4, pos + 0.4, color="#f0e6d3", alpha=0.5, zorder=0)

        ax.axhline(1 / 90, color="grey", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Serial position", fontsize=8)
        ax.set_ylabel("Accuracy" if i == 0 else "", fontsize=8)
        ax.set_title(cond["nice"], fontsize=9, color=cond["color"], fontweight="bold")
        ax.set_xticks(SERIAL_POSITIONS)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # ── ROW 3: Attentional Residue ────────────────────────────────────────────
    for i, cond in enumerate(CONDITIONS):
        ax = ax_mod[i]
        df = frames[cond["label"]]
        if df is None or "probe_target_modality" not in df.columns:
            _no_data(ax, cond["nice"])
            ax.set_title(cond["nice"], fontsize=8, color=cond["color"])
            continue

        modalities = sorted(df["probe_target_modality"].dropna().unique())
        means = [df[df["probe_target_modality"] == m]["correct"].mean() for m in modalities]
        ses   = [_sem(df[df["probe_target_modality"] == m]["correct"].dropna()) for m in modalities]
        colors = [MODALITY_COLORS.get(m, cond["color"]) for m in modalities]

        ax.bar(range(len(modalities)), means, yerr=ses,
               color=colors, alpha=0.8, capsize=5, edgecolor="white")

        # jitter per-sim per-modality
        if "sim_idx" in df.columns:
            rng = np.random.default_rng(42)
            for j, mod in enumerate(modalities):
                sim_means = df[df["probe_target_modality"] == mod].groupby("sim_idx")["correct"].mean()
                jx = rng.uniform(-0.15, 0.15, len(sim_means))
                ax.scatter(j + jx, sim_means.values,
                           color="white", edgecolors=colors[j],
                           s=35, zorder=5, linewidth=1.2)

        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels(modalities, fontsize=7, rotation=15)
        ax.set_ylabel("Accuracy" if i == 0 else "", fontsize=8)
        ax.set_title(
            f"{cond['nice']}\nAttentional Residue",
            fontsize=8, color=cond["color"]
        )
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1 / 90, color="grey", linestyle=":", linewidth=1, alpha=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    # ── Row labels ────────────────────────────────────────────────────────────
    for ax, label in [
        (ax_sp_ocr,    "Lost-in-the-Middle"),
        (ax_traj[0],   "Per-sim Trajectories"),
        (ax_mod[0],    "Attentional Residue"),
    ]:
        ax.annotate(
            label, xy=(-0.22, 0.5), xycoords="axes fraction",
            ha="right", va="center", fontsize=9,
            fontweight="bold", color="#444444", rotation=90,
        )

    fig.suptitle(
        "Set 3 — Multimodal Behavioral Profiling (BPT-10)\n"
        "VLM-type (OCR vs. non-OCR)  ×  Feedback (Yes vs. No)   |   n = 5 sims per condition",
        fontsize=13, fontweight="bold", y=0.995
    )

    out = Path(__file__).parent / "exp3_results_figure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    make_figure()
