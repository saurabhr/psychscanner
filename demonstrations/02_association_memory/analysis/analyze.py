"""Analyze recall accuracy from the 5-condition PAL simulation in ../data/*.csv.

Answers the two questions the demo is meant to surface:
  1. Does recall accuracy differ across the 5 memory/feedback conditions
     (single_turn, trial_chain, episodic_chain, feedback_ci, feedback_reward)?
  2. Does word-pair semantic similarity (L0..L100, from each trcode's prefix)
     help or hurt recall, and does that interact with condition?

``to_csv`` output does not carry the task JSON's word1/word2/similarity/phase
fields (see pal_task.py / cognitive_tasks.md — only trcode/stimulus/resp_*
survive), so this script re-derives:
  - which rows are TEST trials: only test items use the structured
    ``PairedAssociateRecall`` parser, so ``resp_recalled_word`` is populated
    only for them (study/encoding rows are unstructured -> NaN). This also
    correctly handles the trial-chain condition, where study+test rows of one
    pair share an identical (renamed) trcode and can't be told apart by
    trcode alone.
  - similarity level: from the trcode prefix (L0/L25/.../L100), matching
    pal50.json's contexts_id.
  - the correct answer (word2): by loading the same pal50.json and mapping
    each ORIGINAL "<level>_tst_<idx>" trcode -> its word2. Trial-chain rows
    carry a renamed "<level>_pr_<idx>" trcode, so those are matched back to
    the original test key by (level, idx) instead.

Usage:
    source .venv311/bin/activate
    python demonstrations/02_association_memory/analysis/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent
PAL50_PATH = DEMO_DIR.parent.parent / "examples" / "tasks" / "pal50.json"

LEVEL_TO_SIM = {"L0": 0.0, "L25": 0.25, "L50": 0.5, "L75": 0.75, "L100": 1.0}


def word2_by_test_trcode() -> dict[str, str]:
    items = json.loads(PAL50_PATH.read_text())["items"]
    return {trcode: trials[0]["word2"] for trcode, trials in items.items() if "_tst_" in trcode}


def resolve_word2(trcode: str, w2: dict[str, str], ordinal_map: dict[tuple[str, str], str]) -> str | None:
    if trcode in w2:
        return w2[trcode]
    parts = trcode.split("_pr_")  # trial-chain renamed trcode: "<level>_pr_<idx>"
    if len(parts) == 2:
        return ordinal_map.get((parts[0], parts[1]))
    return None


def load_scored_trials() -> pd.DataFrame:
    csvs = sorted(DATA_DIR.glob("*__*.csv"))
    if not csvs:
        raise SystemExit(f"No result CSVs found in {DATA_DIR} — run simulation/run_simulation.py first.")

    w2 = word2_by_test_trcode()
    ordinal_map = {tuple(trcode.split("_tst_")): word2 for trcode, word2 in w2.items()}

    frames = []
    for path in csvs:
        condition, model_tag = path.stem.split("__", 1)
        df = pd.read_csv(path)
        if "resp_recalled_word" not in df.columns:
            continue
        df = df[df["resp_recalled_word"].notna()].copy()  # test trials only (see module docstring)
        if df.empty:
            continue
        df["condition"] = condition
        df["model"] = model_tag
        df["level"] = df["trcode"].str.split("_").str[0]
        df["similarity"] = df["level"].map(LEVEL_TO_SIM)
        df["word2"] = df["trcode"].map(lambda t: resolve_word2(t, w2, ordinal_map))
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["recalled_word"] = out["resp_recalled_word"].astype(str)
    out["correct"] = (
        out["recalled_word"].str.strip().str.lower() == out["word2"].astype(str).str.strip().str.lower()
    )
    return out


def main() -> None:
    df = load_scored_trials()
    df.to_csv(OUT_DIR / "scored_trials.csv", index=False)

    lines = []
    lines.append(
        f"Scored {len(df)} test trials across {df['condition'].nunique()} conditions "
        f"and {df['model'].nunique()} model(s).\n"
    )

    lines.append("=== Accuracy by condition (pooled across models & similarity levels) ===")
    by_cond = df.groupby("condition")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"})
    lines.append(by_cond.to_string())
    lines.append("")

    lines.append("=== Accuracy by condition x model ===")
    by_cond_model = (
        df.groupby(["condition", "model"])["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"})
    )
    lines.append(by_cond_model.to_string())
    lines.append("")

    lines.append("=== Accuracy by similarity level (pooled across conditions & models) ===")
    by_level = df.groupby("similarity")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"})
    lines.append(by_level.to_string())
    lines.append("")

    lines.append("=== Accuracy by condition x similarity level (the interaction) ===")
    pivot = df.pivot_table(index="condition", columns="similarity", values="correct", aggfunc="mean")
    lines.append(pivot.to_string())
    lines.append("")

    lines.append("=== Confidence (1-6) vs correctness, by condition (calibration check) ===")
    if "resp_confidence" in df.columns:
        conf = df.groupby(["condition", "correct"])["resp_confidence"].mean().unstack()
        lines.append(conf.to_string())
    else:
        lines.append("(resp_confidence column not present)")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "analysis_output.txt").write_text(report + "\n")
    by_cond.to_csv(OUT_DIR / "accuracy_by_condition.csv")
    pivot.to_csv(OUT_DIR / "accuracy_by_condition_x_similarity.csv")


if __name__ == "__main__":
    main()
