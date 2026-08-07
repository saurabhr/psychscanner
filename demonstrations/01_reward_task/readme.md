# Psychscanner Features Demonstrated

- 1: Passing different Agents
  - Task: 3-armed bandit (`FeedbackBase`-driven reward-learning task, adapted from
    `examples/17_cognitive_rl_bandits_and_pd.ipynb`), run through 3 agent types
    from `src/psychscanner/agents/` -- `CustomAgent` (baseline), `make_basic_reflection_agent`,
    `make_planner_executor_agent` -- against the same `ExpCard`/`ScannerModel` pipeline,
    real Groq (`llama-3.1-8b-instant`) calls throughout. See `simulation/`, `data/`,
    `analysis/`, `reports/report.md`.
