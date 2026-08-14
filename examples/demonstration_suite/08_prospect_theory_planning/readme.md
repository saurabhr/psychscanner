# Psychscanner Features Demonstrated

- 1: A custom multi-module LangGraph agent (MAP) vs. two comparison agents
  - Task: Prospect Theory risky-choice gambles (`examples/tasks/prospect_theory_demo.tcard.psyscan`,
    generated per musslick/experiment_factor_discovery's benchmark spec — see
    `examples/tasks/generators/generate_prospect_theory.py`), run through 3
    agent architectures from `src/psychscanner/agents/`:
    `make_map_agent` (new — Webb, Mondal & Momennejad 2025's Modular Agentic
    Planner, https://doi.org/10.1038/s41467-025-63804-5), a conversation-memory
    baseline (`memory="Convo"`), and `make_lats_agent` (Monte Carlo tree
    search, pre-existing). Real Groq (`llama-3.1-8b-instant`) calls throughout.
    See `simulation/`, `data/`, `analysis/`, `reports/report.md`.
