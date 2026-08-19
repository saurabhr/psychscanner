# Introspective Self-Report

**Status: Complete**, reported in `preprint-full`/NC.

Source: [`examples/demonstration_suite/07_introspective_selfreport/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/07_introspective_selfreport)

## Feature demonstrated

Direct chat-model invocation for a custom (non-task-card) protocol: a
multi-attribute 2AFC choice + introspective self-report task, adapted from
Plunkett, Morris, Reddy & Morales (2025),
["Self-Interpretability: LLMs Can Describe Complex Internal Processes that
Drive Their Decisions"](https://subjectivitylab.org/papers/2025_Plunkett_LLMs_Introspection.pdf).
The paper fine-tunes GPT-4o/4o-mini to internalize random attribute weights,
then tests whether the model can self-report those weights, and whether a
second fine-tuning round on the self-reporting task itself improves accuracy.

Two substitutions from the paper, both driven by not assuming an OpenAI
fine-tuning budget: preferences are instilled via an in-context few-shot
preamble instead of gradient fine-tuning, and "introspection training" is
stood in for by a worked-example block of other agents' accurate reports
(prepended for half the agents, "trained"; withheld for the rest, "control")
— a simulated analog, not a replication, of the paper's fine-tuning
experiments. See [`reports/report.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/07_introspective_selfreport/reports/report.md)
for the full comparison against the paper's own numbers.

Because the protocol (few-shot preamble → decision trials → introspection
trials) doesn't fit the single-template task-card grammar, trials are driven
directly via `psychscanner.memories.llm_chat_model` +
`langchain_core.messages`, not `ExpCard`/`ScannerModel`. Real Ollama calls
only (`smollm2:360m-instruct-fp16`), no mocking.

**Extension beyond the paper:** each decision trial also elicits a 1-4
confidence rating, scored as a signal-detection problem via `metasignal`
(d'/meta-d'/M-ratio) — how well the model's choices and confidence track its
own instilled preferences.

## psychscanner-primal integration

The decision trials (the part with a real per-trial correct/incorrect
signal — primal's own inclusion bar) are built into a standalone task card
by `simulation/build_primal_task_card.py`, intended for
`psychscanner-primal/examples/tasks/introspection_weights_demo.json`,
runnable through primal's own `ExpCard`/`ScannerModel`/`task_library`
quickstart pattern. **Note:** that file is not currently present in
`psychscanner-primal/examples/tasks/` (checked 2026-08-19) — either it was
never committed there or was since removed; re-run the build script if you
need it. The introspection (self-report) trials have no scalar ground
truth to score, so they're intentionally left out of primal.

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/07_introspective_selfreport/simulation/run_introspection_task.py
python examples/demonstration_suite/07_introspective_selfreport/analysis/analyze.py
```
