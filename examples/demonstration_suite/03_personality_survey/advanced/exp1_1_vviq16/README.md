# Exp 1.1 — VVIQ-16 Imagery Questionnaire

**Goal:** Measure how WEIRD vs. non-WEIRD persona conditioning and memory regime affect
a model's self-reported mental imagery vividness on the VVIQ-16, benchmarked against
OpenPsychometrics norms.

---

## Factors

| Factor | Levels | PsychScanner mapping |
|--------|--------|----------------------|
| **Persona** | WEIRD, non-WEIRD | `cogtype="custom"`, persona_files = axis × traits_n5 |
| **Memory** | item_ST, task_Cv, trial_Cv | `memory` + `chain_type` in ExpCardInit |

**Crossed design:** 2 × 3 = 6 conditions × 5 runs = **30 simulations × 16 trials = 480 trial records**

### Persona construction

Each condition uses a Cartesian product of two persona files:

```
persona_files = [axis.json (1 stmt), traits_n5.json (5 stmts)]
→ product → 5 participants, each: "WEIRD stmt. Individual trait stmt."
```

Traits file: `personas/traits_n5.json` — 5 diverse individual profiles (seed=42 sample
from the 200-statement reference file).

### Memory conditions

| Label | `memory` | `chain_type` | Effect |
|-------|----------|--------------|--------|
| item_ST | SingleTurn | item | Each of 16 trials is a fully independent call |
| task_Cv | Convo | task | All 16 trials share one conversation thread |
| trial_Cv | Convo | trial | Each trial has its own thread (≈ stateless for VVIQ) |

> **Note on trial_Cv:** Because each VVIQ trial is a single-turn exchange, `trial_Cv`
> is functionally equivalent to `item_ST` for this instrument. It is retained for
> consistency with the multi-turn RM tasks in Set 2.

---

## Dependent Variables

| DV | Column in CSV | Range |
|----|--------------|-------|
| Per-item vividness | `resp_Vividness` | 1–5 |
| Trial position | `trial_idx` | 0–15 |
| Scene-item code | `trcode` | S1_1 … S4_4 |
| Total VVIQ | derived: `sum(resp_Vividness)` per sim | 16–80 |

---

## Outputs

| File | Description |
|------|-------------|
| `raw/exp1_1_*.csv` | Per-condition CSVs (6 files, 80 rows each) |
| `processed/exp1_1_combined.csv` | All 480 trial records, wide format |
| `figures/trial_trajectory.png` | Mean vividness per trial position, per condition |
| `figures/total_vviq_violin.png` | Total VVIQ distribution per Persona × Memory cell |
| `figures/vviq_vs_benchmark.png` | Per-scene means vs. OpenPsychometrics norms |

---

## Headline Test

> Does WEIRD vs. non-WEIRD persona conditioning shift the total VVIQ score, and does
> this shift depend on the memory regime?

Primary contrast: **Persona main effect on total VVIQ** (WEIRD − non-WEIRD).
Secondary contrast: **Persona × Memory interaction** (does memory amplify or attenuate
the persona effect?).

---

## Benchmark

**OpenPsychometrics VVIQ** — download the open dataset from
[https://openpsychometrics.org/](https://openpsychometrics.org/) and place the cleaned
CSV at `analysis/benchmark_vviq.csv`. The benchmark comparison figure requires this file.

---

## Running

```bash
# From the psychscanner project root:
source psyscan/bin/activate
python examples/demonstration_suite/03_personality_survey/advanced/exp1_1_vviq16/run.py
```

---

## Notes

- **Temperature = 0:** All 5 runs within each persona condition share the same axis
  statement, so variation comes entirely from the trait component of the persona. At
  temperature 0 the model is deterministic, meaning the 5 trait-matched participants
  produce 5 genuinely distinct contexts but identical within-context responses. Raise
  `temperature` in config for stochastic variation across runs.
- **Ollama context-overflow stall (observed):** Simulation i=2 of `non_weird_trial_Cv`
  stalled at trial 13 for ~30 minutes before recovering. Root cause: `smollm2:360m`
  has a limited context window; with `chain_type="trial"` each thread is fresh, so this
  was likely a transient Ollama scheduling issue rather than a context overflow. Not a
  PsychScanner bug. Mitigation: set `OLLAMA_REQUEST_TIMEOUT` or use a model with a
  larger context window for `task_Cv` experiments.
- **Observed data pattern:** `trial_Cv` (Convo + per-trial thread) produced lower mean
  vividness (≈3.0) than `item_ST` (≈4.0) and `task_Cv` (≈4.1) at temperature=0 with
  smollm2. This is an architecture-level effect of how LangGraph Convo memory
  initializes the graph node relative to SingleTurn — not a PsychScanner bug.
- **Package patches:** none required. All tutorials passed (see tutorial check below).
- **Tutorial check (post-run):** `tests/test_readme_quickstart.py` — PASS (no package
  changes were made).
