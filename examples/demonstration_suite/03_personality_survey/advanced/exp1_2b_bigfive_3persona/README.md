# Exp 1.2b — Big Five (BFI-44), 3-Persona

**Status:** READY — all parameters resolved.

Extends Exp 1.2a by adding a **Zhang** persona condition (individuals sampled from the
Zhang et al. dataset, `persona_200.json`).

---

## Design

| Factor | Levels | Notes |
|--------|--------|-------|
| **Persona** | WEIRD, non-WEIRD+Jung, Zhang | 3 levels |
| **Summary K** | 1, 5 | chain conditions only |
| **Memory K** | 4, 16 | chain conditions only |
| **Control** | item_ST | `chain_type=item`, `memory=SingleTurn` |

**Total:** 3 personas × 5 conditions × 5 simulations = 3 300 trial records (3 × 5 × 5 × 44).

## Persona files

| Persona | persona_files | Participants |
|---------|--------------|-------------|
| WEIRD | `weird_axis` × `traits_n5` | 1×5=5 |
| non-WEIRD+Jung | `non_weird_axis` × `jung_axis` × `traits_n5` | 1×1×5=5 |
| Zhang | `zhang_axis` (5 individuals, standalone) | 5×1=5 |

**Zhang axis**: 5 individuals sampled from `persona_200.json` (seed=7, excluding `traits_n5`
sample). Each statement is a complete individual description from the Zhang et al. dataset;
no cultural-axis framing is layered on top.

## Outputs

- `raw/exp1_2b_{persona}_{cond}.csv` (15 files, 220 rows each)
- `processed/exp1_2b_combined.csv` (3 300 rows)
- `analysis/trait_scores.csv`
- `figures/radar_3personas.png` — OCEAN radar overlaying all 3 personas
- `figures/per_trait_3personas.png` — per-OCEAN-trait scores across conditions

## Headline Test

> Does the Zhang (non-WEIRD, non-Jungian) persona profile differ from WEIRD and
> non-WEIRD+Jung? Does the cultural/typological framing shift OCEAN scores beyond
> what individual trait variation accounts for?
