"""Analysis for the introspective self-report demo (../simulation/run_introspection_task.py).

Two independent questions, mirroring the two halves of the paper this demo
adapts (Plunkett et al., 2025) plus one extension:

1. Self-report accuracy (the paper's own headline stat, Sec 2.2/4.2): for
   each agent x attribute, average the repeated introspection reports and
   correlate (Pearson r) against the agent's true target weight -- computed
   separately for "trained" agents (got the worked-example preamble, the
   in-context analog of the paper's Experiment 2 fine-tuning) vs "control"
   agents, to see whether the analog training helped.

2. Extension beyond the paper: the decision trials (2AFC choice + 1-4
   confidence rating) are scored as a signal-detection problem via
   metasignal -- stim = the option the agent's target weights actually
   favor, resp = the model's choice, conf = its self-rated confidence.
   d'/meta-d'/M-ratio measure how well the model's choices and confidence
   track its own instilled (if only in-context) preferences. The paper
   itself has no confidence ratings or SDT analysis; this is grounded in
   its decision-trial structure but goes beyond what it reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _analysis_common import demo_dirs, require_data, save_figure, use_agg_backend, write_summary

DATA_DIR, OUT_DIR = demo_dirs(__file__)
N_RATINGS = 4


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(DATA_DIR / "decisions.csv") if (DATA_DIR / "decisions.csv").stat().st_size else pd.DataFrame()
    reports = pd.read_csv(DATA_DIR / "introspection_reports.csv") if (DATA_DIR / "introspection_reports.csv").stat().st_size else pd.DataFrame()
    return decisions, reports


def self_report_accuracy(reports: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    avg = (
        reports.groupby(["agent", "attribute", "trained"], as_index=False)
        .agg(reported_weight=("reported_weight", "mean"), target_weight=("target_weight", "first"))
    )
    lines.append(f"Introspection reports parsed: {len(reports)} (agent x attribute x rep rows)")
    lines.append(f"Averaged agent x attribute points: {len(avg)}\n")

    lines.append("Self-report accuracy (reported vs target attribute weight):")
    for trained, label in [(True, "trained (worked-example preamble)"), (False, "control")]:
        sub = avg[avg["trained"] == trained]
        if len(sub) < 3:
            lines.append(f"  {label}: n={len(sub)} -- too few points for a correlation")
            continue
        r, p = stats.pearsonr(sub["reported_weight"], sub["target_weight"])
        lines.append(f"  {label}: r = {r:.3f} (p = {p:.3f}, n = {len(sub)})")
    lines.append(
        "  Paper's own numbers for reference (GPT-4o, Fig. 2/3): r=.54 before / .74 after "
        "introspection-training fine-tuning (Exp 1 -> 2); r=.46 -> .71 for generalization to "
        "native preferences (Exp 3). This demo's 'trained' condition is a same-context worked-"
        "example analog, not fine-tuning -- see readme.md."
    )
    lines.append("")
    return avg


def decision_sdt(decisions: pd.DataFrame, lines: list[str]) -> pd.DataFrame | None:
    valid = decisions.dropna(subset=["model_choice", "confidence"]).copy()
    lines.append(f"Decision trials: {len(decisions)} total, {len(valid)} with a parseable choice + confidence")
    if valid.empty:
        lines.append("  No valid decision trials -- skipping accuracy/SDT analysis.\n")
        return None

    valid["correct"] = valid["model_choice"] == valid["correct_option"]
    lines.append(f"  Choice accuracy (matches the option the agent's target weights favor): {valid['correct'].mean():.3f}\n")

    stim = (valid["correct_option"] == "A").astype(int).to_numpy()
    resp = (valid["model_choice"] == "A").astype(int).to_numpy()
    conf = valid["confidence"].astype(int).clip(1, N_RATINGS).to_numpy()

    from metasignal import stdpy
    measures = stdpy.compute_all_measures(stim, resp, conf, n_ratings=N_RATINGS, return_type="dict")
    lines.append("SDT / metacognition measures on the decision trials (via metasignal):")
    for key in ["dprime", "criterion", "meta_d", "M_ratio", "AUC2", "mean_conf"]:
        lines.append(f"  {key}: {measures.get(key):.3f}" if measures.get(key) is not None else f"  {key}: n/a")
    lines.append("")
    return valid


def plot(avg: pd.DataFrame) -> None:
    use_agg_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for trained, color, label in [(True, "tab:purple", "trained"), (False, "tab:blue", "control")]:
        sub = avg[avg["trained"] == trained]
        ax.scatter(sub["target_weight"], sub["reported_weight"], color=color, alpha=0.7, label=label)

    lo, hi = -100, 100
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="perfect report")
    ax.set_xlabel("target attribute weight")
    ax.set_ylabel("reported attribute weight")
    ax.set_title("Self-report accuracy: reported vs. target attribute weights")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, OUT_DIR, "reported_vs_target_weights.png")


def main() -> None:
    decisions, reports = load()
    require_data(not decisions.empty or not reports.empty, DATA_DIR, "run the simulation first.")

    lines: list[str] = []
    avg = self_report_accuracy(reports, lines) if not reports.empty else None
    decision_sdt(decisions, lines) if not decisions.empty else None
    write_summary(OUT_DIR, lines)

    if avg is not None and not avg.empty:
        plot(avg)


if __name__ == "__main__":
    main()
