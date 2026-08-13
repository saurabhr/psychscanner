# Exp 2.1 — PAL-50: Memory(3) × Similarity(5) × With Feedback

**Status:** READY — all parameters resolved.

---

## Design

**Goal:** Test whether memory architecture affects paired-associate recall accuracy across
5 levels of semantic similarity, with correctness feedback after each test trial.

**Task:** PAL-50 — 50 word pairs, 10 per similarity level, study phase then test phase.

| Factor | Levels | Notes |
|--------|--------|-------|
| **Memory architecture** | item_ST, task_Cv, trial_Cv | 3 levels |
| **Similarity** | 0.0, 0.25, 0.50, 0.75, 1.0 | semantic similarity of word pairs |
| **Feedback** | Yes (this exp) | all three: Correct/Incorrect + correct answer |

**No persona conditioning** — `cogtype="no"`, `nsim=5`.

### Memory conditions

| Label | chain_type | memory | Description |
|-------|-----------|--------|-------------|
| item_ST | item | SingleTurn | Each trial independent — baseline (prior knowledge only) |
| task_Cv | task | Convo | All trials in one thread — full encoding-then-test memory |
| trial_Cv | trial | Convo | Each pair in own thread — encoding + immediate test per pair |

### Similarity operationalization

| Level | sim | Example pair | Relation |
|-------|-----|-------------|---------|
| L0 | 0.0 | cat–umbrella | Unrelated (random) |
| L25 | 0.25 | lion–tiger | Same superordinate category |
| L50 | 0.50 | bee–honey | Semantic associate |
| L75 | 0.75 | salt–pepper | Strong canonical pair |
| L100 | 1.0 | big–large | Near-synonym |

### Task structure

- **Study phase (50 trials):** "STUDY PHASE — Please remember this word pair: the word '{w1}' is paired with '{w2}'."
- **Test phase (50 trials):** "TEST PHASE — What word was paired with '{w1}' during the study phase?"
- **Parser:** `PairedAssociateRecall` — `recalled_word` + `confidence` (1–6)
- **Feedback (Exp 2.1 only):** JSON with result (Correct/Incorrect), recalled word, correct answer

### trial_Cv variant

For `trial_Cv`, the task dict is constructed dynamically in run.py as 50 dual-stimulus
items (encoding + test per pair, same trcode → same thread). Parser list: `[None, PairedAssociateRecall]`.

## DVs

- Recall accuracy (recalled_word == word2, exact match)
- Confidence (1–6)
- Confidence calibration (confidence vs. accuracy)

## Outputs

- `raw/exp2_1_{cond}.csv` (3 files)
- `processed/exp2_1_combined.csv`
- `analysis/test_scored.csv`
- `figures/accuracy_x_similarity.png`
- `figures/confidence_x_accuracy.png`

## Headline Test

> Does memory architecture (item_ST < trial_Cv < task_Cv) modulate accuracy across
> similarity levels? Does the advantage of task_Cv grow for low-similarity (arbitrary) pairs?

## Cross-sub-exp

Compare `(2.1 − 2.2)` accuracy delta per memory condition: does feedback help more
for low-similarity pairs or in specific memory architectures?
