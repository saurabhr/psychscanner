# VLM Task

**Status:** 1 of 4 conditions collected; the 2x2 `advanced/set3`/`set4`
factorial figures exist on disk but are excluded from the papers as
unreproducible (source CSVs no longer present).

Source: [`examples/demonstration_suite/04_vlm_task/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/04_vlm_task)

## Features demonstrated

1. VLM task with single-turn and summary memory
2. Reward and accuracy feedback

A shape-naming VQA task (6 PIL-drawn color+shape images, each shown twice
per participant across two blocks) run across 2 memory conditions x 2
feedback conditions, real vision-model calls via OpenRouter
(`google/gemma-4-26b-a4b-it:free`).

| Feature | Condition(s) |
|---|---|
| Single-turn / summary memory | `singleturn_*` (`memory=SingleTurn`); `summary_*` (`memory=Convo`, `memory_k=6`, `summary_k=3`) |
| Reward / accuracy feedback | `*_reward` (`RewardFeedback`: +1/-1 only); `*_accuracy` (`AccuracyFeedback`: correct/incorrect + right answer) |

Stimuli are generated on the fly by `simulation/stimuli.py` (no reusable
image stimuli existed in the repo, including the text-only `img_vivid_task/`).

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/04_vlm_task/simulation/run_vlm_task.py
python examples/demonstration_suite/04_vlm_task/analysis/analyze.py
```

## See also

- [Feedback Loop](../../archive/examples_feedback_loop.md) (archived)
- [Cognitive Tasks guide](../../guides/cognitive_tasks.md)
