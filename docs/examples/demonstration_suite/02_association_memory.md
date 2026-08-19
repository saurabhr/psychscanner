# Association Memory

**Status: Skeleton** — readme and folder only, not yet run.

Source: [`examples/demonstration_suite/02_association_memory/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/02_association_memory)

## Features demonstrated

1. Single-turn trials
2. Trial-chain trials
3. Episodic-chain trials
4. Correct/incorrect feedback
5. Reward feedback

A paired-associate-learning style task (`simulation/pal_task.py`,
`simulation/run_simulation.py`) exercising the different trial chain types
and feedback mechanisms `ExpCard`/`ScannerModel` support, complementing the
single agent-comparison focus of [Reward Task](01_reward_task.md).

The `advanced/` arms (`exp2_1_with_feedback`, `exp2_2_no_feedback`) run
feedback-on vs feedback-off comparisons — see their own `README.md`s.

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/02_association_memory/simulation/run_simulation.py
python examples/demonstration_suite/02_association_memory/analysis/analyze.py
```

## See also

- [Feedback Loop](../../archive/examples_feedback_loop.md) (archived) — the
  `FeedbackBase` interface this demo exercises
- [Memory Types guide](../../guides/memory_types.md) — single-turn vs
  trial-chain vs episodic-chain
