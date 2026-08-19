# Personality Survey

**Status: core design 1 of 8 cells collected.** Its `advanced/exp1_1_vviq16`
arm (VVIQ-16) is complete and reported in the main manuscript.

Source: [`examples/demonstration_suite/03_personality_survey/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/03_personality_survey)

## Features demonstrated

1. Two levels of persona
2. Independent (stateless) trial sampling
3. Conversation and summary memory (stateful)
4. Windowing conversation and summary memory

A BFI-44 (Big Five Inventory, 44 items / OCEAN traits) survey run across 2
persona conditions x 4 memory conditions, real Groq (`llama-3.3-70b-versatile`)
calls throughout.

| Feature | Condition(s) |
|---|---|
| Two levels of persona | `weird`, `non_weird_jung` (persona files under `advanced/personas/`) |
| Independent stateless sampling | `stateless` (`memory=SingleTurn`) |
| Conversation / summary memory | `conversation` (`memory=Convo`, unbounded); `summary_windowed` (`memory=Convo` + `summary_k`) |
| Windowing convo/summary memory | `conversation_windowed` (`memory_k=8`, no summary); `summary_windowed` (`memory_k=8`, `summary_k=4`) |

The `advanced/exp1_1_vviq16` arm runs the built-in VVIQ-16 imagery
questionnaire this way and is complete; `exp1_2a_bigfive_2persona` and
`exp1_2b_bigfive_3persona` extend the persona axis further.

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/03_personality_survey/simulation/run_personality_survey.py
python examples/demonstration_suite/03_personality_survey/analysis/analyze_personality_survey.py
```

Data: `data/raw/*.csv` (per-cell), `data/processed/combined.csv` (all cells).

## See also

- [Basic Survey](../../archive/examples_basic_survey.md) (archived) — the
  minimal single-persona VVIQ-16 walkthrough this supersedes
- [Multi-Persona](../../archive/examples_multi_persona.md) (archived) — the
  `persona_statements` JSON format walkthrough this supersedes
- [Memory Types guide](../../guides/memory_types.md)
