# Advanced Persona/Survey Experiments

Three sub-experiments testing how **persona conditioning** and **cross-item memory**
affect model self-report, all using **`ollama / smollm2:360m-instruct-fp16`**,
temperature 0, n=5 simulated participants per condition. These sit alongside the
main [`03_personality_survey`](../readme.md) demo — same topic (persona × survey),
more elaborate factorial designs with explicit hypotheses and literature grounding
(PersonaLLM NAACL 2024, Big5-Chat ACL 2025).

| Exp | Task | Status |
|-----|------|--------|
| 1.1 | VVIQ-16 imagery survey | Data collected |
| 1.2a | BFI-44, 2-persona (WEIRD / non-WEIRD+Jung) | Running |
| 1.2b | BFI-44, 3-persona (adds Zhang) | Ready to run |

## Exp 1.1 — VVIQ-16
```
exp1_1_vviq16/run.py
```
**Factors:** Persona(WEIRD/non-WEIRD) × Memory(item_ST / task_Cv / trial_Cv)
**Task:** 16-item VVIQ vividness rating (1–5 Likert)
**Output:** 480 rows, `figures/trial_trajectory.png`, `figures/total_vviq_violin.png`

## Exp 1.2a — BFI-44, 2-Persona
```
exp1_2a_bigfive_2persona/run.py
```
**Factors:** Persona(WEIRD/non-WEIRD+Jung) × SummaryK(1,5) × MemoryK(4,16) + Control
**Task:** 44-item BFI-44 OCEAN personality inventory (1–5 agreement Likert)
**Participants:** axis(1) × traits_n5(5) = 5 per condition
**Output:** 2 200 rows, OCEAN radar, per-trait figures

## Exp 1.2b — BFI-44, 3-Persona
```
exp1_2b_bigfive_3persona/run.py
```
**Adds Zhang** persona: 5 individuals from Zhang et al. (`persona_200.json`), used
without cultural-axis framing.
**Output:** 3 300 rows, 3-persona OCEAN radar

## Shared resources

| Path | Description |
|------|-------------|
| `personas/weird/weird_axis.json` | WEIRD cultural axis (1 stmt) |
| `personas/non_weird/non_weird_axis.json` | non-WEIRD axis (1 stmt) |
| `personas/jung/jung_axis.json` | Jungian I+N typology axis (1 stmt) |
| `personas/zhang/zhang_axis.json` | Zhang et al. individual descriptions (5 stmts) |
| `personas/traits_n5.json` | 5 diverse trait profiles from `persona_200.json` |
| `examples/tasks/vviq16.json` | VVIQ-16 task |
| `examples/tasks/bfi44.json` | BFI-44 task (44 items, `DefaultLiteralAgree` parser) |

## Running order

```bash
source psyscan/bin/activate

python examples/demonstration_suite/03_personality_survey/advanced/exp1_1_vviq16/run.py
python examples/demonstration_suite/03_personality_survey/advanced/exp1_2a_bigfive_2persona/run.py
python examples/demonstration_suite/03_personality_survey/advanced/exp1_2b_bigfive_3persona/run.py
```

All `run.py` scripts are **idempotent**: re-running skips conditions where output CSVs
already exist and clears stale session directories automatically.
