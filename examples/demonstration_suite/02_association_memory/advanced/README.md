# Advanced Episodic (PAL-50) Experiments

Two sub-experiments testing whether **memory architecture** affects paired-associate
learning across 5 semantic similarity levels, using **`ollama /
smollm2:360m-instruct-fp16`**, temperature 0, n=5 simulated participants per
condition. These sit alongside the main
[`02_association_memory`](../readme.md) demo — same topic (memory architecture ×
recall), a more elaborate factorial design over 5 similarity levels with feedback
as an explicit third factor.

**Task:** PAL-50 — 50 word pairs (10 per level: 0.0 / 0.25 / 0.50 / 0.75 / 1.0), study
phase then test phase. Parser: `PairedAssociateRecall` (recalled_word + confidence 1–6).

**Memory conditions:**

| Label | chain_type | memory | Description |
|-------|-----------|--------|-------------|
| item_ST | item | SingleTurn | Independent trials — pure prior knowledge baseline |
| task_Cv | task | Convo | All trials in one thread — full episodic memory |
| trial_Cv | trial | Convo | 50 dual-stimulus items — within-pair encoding+test |

## Exp 2.1 — With Feedback
```
exp2_1_with_feedback/run.py
```
Feedback after each test trial: Correct/Incorrect + correct answer revealed.

## Exp 2.2 — No Feedback
```
exp2_2_no_feedback/run.py
```
Also generates `figures/exp2_1_vs_2_2_feedback_contrast.png` (cross-sub-exp comparison).

## Running order

```bash
source psyscan/bin/activate

python examples/demonstration_suite/02_association_memory/advanced/exp2_1_with_feedback/run.py
python examples/demonstration_suite/02_association_memory/advanced/exp2_2_no_feedback/run.py
```

All `run.py` scripts are **idempotent**: re-running skips conditions where output CSVs
already exist and clears stale session directories automatically.
