# Reward Task

**Status: Complete** — real run data and an analyzed report exist.

Source: [`examples/demonstration_suite/01_reward_task/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/01_reward_task)

## Feature demonstrated

Passing different agents into the same task harness. A 3-armed bandit
(`FeedbackBase`-driven reward-learning task, adapted from
[`17_cognitive_rl_bandits_and_pd.ipynb`](../../tutorials/17_cognitive_rl_bandits_and_pd.ipynb))
is run through 3 agent types from `src/psychscanner/agents/` — `CustomAgent`
(baseline), `make_basic_reflection_agent`, `make_planner_executor_agent` —
against the same `ExpCard`/`ScannerModel` pipeline, all via
`ScannerModel.run(custom_agent=...)`. Real Groq (`llama-3.1-8b-instant`)
calls throughout, no mocking.

## Result summary

3 agents played 6 rounds each of a hidden-mean bandit (arms A/B/C, true
means 2.0/6.0/4.0). The plain baseline (`CustomAgent`) accumulated the most
reward (16.14 total), ahead of basic-reflection (14.14) and
planner/executor (12.14) — none came close to the optimal arm, and all three
spent most rounds on the *worst* arm. Full write-up, per-round data, and the
reward-by-agent figure: [`reports/report.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/01_reward_task/reports/report.md).

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/01_reward_task/simulation/run_reward_task.py
python examples/demonstration_suite/01_reward_task/analysis/analyze.py
```

## See also

- [Feedback Loop](../../archive/examples_feedback_loop.md) (archived) — the
  `FeedbackBase` interface this demo exercises
- [Cognitive Tasks guide](../../guides/cognitive_tasks.md)
