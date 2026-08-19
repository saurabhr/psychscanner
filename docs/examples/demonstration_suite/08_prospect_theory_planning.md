# Prospect Theory Planning

**Status: Complete**, reported in `preprint-full`/NC.

Source: [`examples/demonstration_suite/08_prospect_theory_planning/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/08_prospect_theory_planning)

## Feature demonstrated

A custom multi-module LangGraph agent (MAP) vs. two comparison agents. Task:
Prospect Theory risky-choice gambles
(`examples/tasks/prospect_theory_demo.tcard.psyscan`, generated per
musslick/experiment_factor_discovery's benchmark spec — see
`examples/tasks/generators/generate_prospect_theory.py`), run through 3
agent architectures from `src/psychscanner/agents/`:

- `make_map_agent` — Webb, Mondal & Momennejad (2025)'s Modular Agentic
  Planner, built from the [*Nature Communications* paper](https://doi.org/10.1038/s41467-025-63804-5)
- a conversation-memory baseline (`memory="Convo"`)
- `make_lats_agent` — Monte Carlo tree search, pre-existing

Real Groq (`llama-3.1-8b-instant`) calls throughout.

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/08_prospect_theory_planning/simulation/run_prospect_theory.py
python examples/demonstration_suite/08_prospect_theory_planning/analysis/analyze.py
```

Data: `data/prospect_convo_memory.csv`, `data/prospect_lats.csv`,
`data/prospect_map.csv`. Report:
[`reports/report.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/08_prospect_theory_planning/reports/report.md).
