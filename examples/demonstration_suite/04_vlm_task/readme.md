# Psychscanner Features Demonstrated

- 1: VLM task with Single-Turn and Summary
- 2: Reward and Accuracy Feedback

## What's here

A shape-naming VQA task (6 PIL-drawn color+shape images, each shown twice per
participant across two blocks) run across 2 memory conditions x 2 feedback
conditions, using real vision-model calls (OpenRouter, `google/gemma-4-26b-a4b-it:free`).

| Feature bullet | Condition(s) |
|---|---|
| 1. Single-Turn / Summary memory | `singleturn_*` (`memory=SingleTurn`, stateless per trial); `summary_*` (`memory=Convo`, `memory_k=6`, `summary_k=3` -- rolling summary once history overflows) |
| 2. Reward / Accuracy feedback | `*_reward` (`RewardFeedback`: +1/-1 score only, answer withheld); `*_accuracy` (`AccuracyFeedback`: explicit correct/incorrect + the right answer) |

- `simulation/stimuli.py` -- generates the 6 shape/color PNGs (no reusable image stimuli existed in the repo or in the sibling `img_vivid_task/`, which is text-only)
- `simulation/run_vlm_task.py` -- runs all 4 conditions (2x2), idempotent CSV output
- `data/*.csv` -- per-condition trial output (12 trials/participant x 2 participants)
- `analysis/analyze.py` -- scores each trial (color+shape word match), accuracy by condition and by block (repeat exposure)
- `reports/report.md` -- results write-up

Run order:
```bash
source .venv/bin/activate
python examples/demonstration_suite/04_vlm_task/simulation/run_vlm_task.py
python examples/demonstration_suite/04_vlm_task/analysis/analyze.py
```
