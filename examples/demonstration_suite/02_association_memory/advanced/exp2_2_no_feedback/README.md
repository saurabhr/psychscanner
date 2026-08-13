# Exp 2.2 — PAL-50: Memory(3) × Similarity(5) × No Feedback

**Status:** READY — all parameters resolved.

Identical design to Exp 2.1 except **no feedback** is given after retrieval trials.
See [Exp 2.1 README](../exp2_1_with_feedback/README.md) for full design.

## Key difference from Exp 2.1

`feedback=False` in ExpCardInit; `fb=False` on all test stimuli in the trial_Cv task dict.

## Outputs

- `raw/exp2_2_{cond}.csv` (3 files)
- `processed/exp2_2_combined.csv`
- `analysis/test_scored.csv`
- `figures/exp2_1_vs_2_2_feedback_contrast.png` — generated after both exps complete

## Headline Test

> Does the episodic-system (task_Cv) advantage over item_ST depend on feedback?
> Key comparison: `(2.1 − 2.2)` accuracy delta per memory architecture × similarity level.
