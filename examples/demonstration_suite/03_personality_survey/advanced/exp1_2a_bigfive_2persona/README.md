# Exp 1.2a — Big Five (BFI-44), 2-Persona

**Status:** READY — all parameters resolved.

---

## Design

**Goal:** Measure how WEIRD vs. non-WEIRD+Jung persona conditioning, Summary K, and Memory K
affect a model's BFI-44 self-report, benchmarked against OpenPsychometrics norms.

**Instrument:** BFI-44 (John & Srivastava, 1999) — dominant in LLM personality research (PersonaLLM
NAACL 2024, Big5-Chat ACL 2025, LMLPA). 44 items, OCEAN traits, 1–5 agreement Likert scale.

| Factor | Levels | Notes |
|--------|--------|-------|
| **Persona** | WEIRD, non-WEIRD+Jung | axis × traits_n5 → 5 participants each |
| **Summary K** | 1, 5 | only active in chain conditions |
| **Memory K** | 4, 16 | only active in chain conditions |
| **Control** | item_ST | `chain_type=item`, `memory=SingleTurn` |

**Conditions per persona:**
| Label | chain_type | memory | summary_k | memory_k |
|-------|-----------|--------|-----------|---------|
| item_ST | item | SingleTurn | — | — |
| task_sk1_mk4 | task | Convo | 1 | 4 |
| task_sk1_mk16 | task | Convo | 1 | 16 |
| task_sk5_mk4 | task | Convo | 5 | 4 |
| task_sk5_mk16 | task | Convo | 5 | 16 |

**Total:** 2 personas × 5 conditions × 5 simulations = 2 200 trial records.

## Persona files

| Persona | persona_files |
|---------|--------------|
| WEIRD | `personas/weird/weird_axis.json` × `personas/traits_n5.json` |
| non-WEIRD+Jung | `personas/non_weird/non_weird_axis.json` × `personas/jung/jung_axis.json` × `personas/traits_n5.json` |

**Jung axis:** Dimension-decomposed natural-language statement (introversion-intuition framing).
Cartesian product: 1 × 1 × 5 = 5 participants — matches WEIRD condition count.

## DVs

- Per-item Likert agreement (1–5, reverse-scored where applicable)
- Per-trait mean score (OCEAN × sim × condition)
- Full OCEAN profile per simulation

## Outputs

- `raw/exp1_2a_{persona}_{cond}.csv` (10 files, 220 rows each)
- `processed/exp1_2a_combined.csv` (2 200 rows)
- `analysis/trait_scores.csv`
- `figures/per_trait_per_cell.png`
- `figures/trial_chain_vs_control_contrast.png`
- `figures/radar_per_persona.png`

## Headline Test

> Does WEIRD vs. non-WEIRD+Jung persona conditioning shift the OCEAN profile?
> Does Summary K moderate the cross-item context effect relative to the item_ST control?

## Benchmark

OpenPsychometrics BFI-44 norms — download and place at `analysis/benchmark_bigfive.csv`.
