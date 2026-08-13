"""Score the 4 VLM-task condition CSVs and summarize accuracy by memory x feedback.

Reads examples/demonstration_suite/04_vlm_task/data/{singleturn,summary}_{reward,accuracy}.csv
(written by simulation/run_vlm_task.py), scores each trial's raw response
against its ground-truth color/shape (same loose word-match used by the
feedback handlers), and writes:

  analysis/scored_trials.csv   -- every trial with a `correct` column
  analysis/condition_summary.csv -- accuracy by memory x feedback (+ by block)

Run with:
    source .venv/bin/activate
    python examples/demonstration_suite/04_vlm_task/analysis/analyze.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _analysis_common import demo_dirs, require_data

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR, OUT_DIR = demo_dirs(__file__)

sys.path.insert(0, str(DEMO_DIR / "simulation"))
from stimuli import generate  # noqa: E402

# image name -> (color, shape) ground truth; trcode is "shapes_<image>_b<block>"
# (to_csv only keeps a fixed column set -- trcode is the only place the trial's
# custom keys like image/color/shape/block survive into the CSV, see
# psyscan_io.py::_trials_to_rows -- so we re-derive them from it here).
_TRUTH = {item["name"]: (item["color"], item["shape"]) for item in generate()}
_TRCODE_RE = re.compile(r"^shapes_(?P<image>img\d)_b(?P<block>\d)$")

CONDITIONS = [
    ("SingleTurn", "reward"), ("SingleTurn", "accuracy"),
    ("Summary", "reward"), ("Summary", "accuracy"),
]


def _is_correct(text: str, color: str, shape: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(rf"\b{color}\b", t)) and bool(re.search(rf"\b{shape}\b", t))


def load_scored() -> pl.DataFrame:
    frames = []
    for memory, feedback in CONDITIONS:
        path = DATA_DIR / f"{memory.lower()}_{feedback}.csv"
        if not path.exists():
            print(f"  missing {path.name}, skipping")
            continue
        df = pl.read_csv(path)
        images, blocks, colors, shapes, corrects = [], [], [], [], []
        for row in df.iter_rows(named=True):
            m = _TRCODE_RE.match(row["trcode"])
            image, block = m.group("image"), int(m.group("block"))
            color, shape = _TRUTH[image]
            images.append(image)
            blocks.append(block)
            colors.append(color)
            shapes.append(shape)
            corrects.append(_is_correct(row["pred_resp_raw"], color, shape))
        df = df.with_columns([
            pl.lit(memory).alias("memory"),
            pl.lit(feedback).alias("feedback"),
            pl.Series("image", images),
            pl.Series("block", blocks),
            pl.Series("color", colors),
            pl.Series("shape", shapes),
            pl.Series("correct", corrects).cast(pl.Int64),
        ])
        frames.append(df)
    require_data(frames, DATA_DIR, "run simulation/run_vlm_task.py first.")
    return pl.concat(frames, how="diagonal_relaxed")


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["memory", "feedback"])
        .agg([
            pl.col("correct").mean().alias("accuracy"),
            pl.len().alias("n_trials"),
        ])
        .sort(["memory", "feedback"])
    )


def summarize_by_block(df: pl.DataFrame) -> pl.DataFrame:
    if "block" not in df.columns:
        return pl.DataFrame()
    return (
        df.group_by(["memory", "feedback", "block"])
        .agg([pl.col("correct").mean().alias("accuracy"), pl.len().alias("n_trials")])
        .sort(["memory", "feedback", "block"])
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored = load_scored()
    scored.write_csv(OUT_DIR / "scored_trials.csv")
    print(f"Saved analysis/scored_trials.csv ({len(scored)} rows)")

    cond_summary = summarize(scored)
    cond_summary.write_csv(OUT_DIR / "condition_summary.csv")
    print("\nAccuracy by memory x feedback:")
    print(cond_summary)

    block_summary = summarize_by_block(scored)
    if not block_summary.is_empty():
        block_summary.write_csv(OUT_DIR / "block_summary.csv")
        print("\nAccuracy by memory x feedback x block (repeat exposure):")
        print(block_summary)

    overall_by_feedback = (
        scored.group_by("feedback").agg(pl.col("correct").mean().alias("accuracy")).sort("feedback")
    )
    overall_by_memory = (
        scored.group_by("memory").agg(pl.col("correct").mean().alias("accuracy")).sort("memory")
    )
    print("\nOverall accuracy by feedback type:")
    print(overall_by_feedback)
    print("\nOverall accuracy by memory type:")
    print(overall_by_memory)
