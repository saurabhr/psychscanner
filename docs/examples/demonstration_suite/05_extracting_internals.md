# Extracting Internals

**Status: Complete** — real, non-mock run data and an analyzed report
exist. See [`reports/report.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/05_extracting_internals/reports/report.md).
Toy-scale (GPT-2 small, sycophancy trait): extraction and linear probing
both succeed (100% probe accuracy vs. 20% shuffled baseline at several
layers), but activation steering produces no detectable behavioral shift at
any layer or magnitude tested — a real, disclosed limitation of applying
the method to a small non-instruction-tuned base model, not a bug.

Source: [`examples/demonstration_suite/05_extracting_internals/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/05_extracting_internals)

## Features demonstrated

1. Residual-stream activation extraction (nnsight / nnterp backend)
2. Linear probing for persona-trait directions
3. Activation steering along extracted trait vectors

Reference study: *Persona Vectors: Monitoring and Controlling Character
Traits in Language Models* (Chen et al., Anthropic, 2025) — extracts
activation directions for traits (e.g. sycophancy, hallucination, "evil"
persona), monitors them during generation, and steers behavior by
adding/subtracting the vector. Maps directly onto psychscanner's existing
persona trials, using the model internals already exposed by the
nnsight/nnterp backends.

Unlike the `ChatNNsightModel`/`ChatNNterpModel` backends used elsewhere in
the framework (which only capture activations, not intervene), steering
requires raw nnsight `trace()` calls outside the `ExpCard`/`ScannerModel`
pipeline — the same pattern Demo 06's ROME/Othello scripts use.

## Run it

```bash
source .venv311/bin/activate
python examples/demonstration_suite/05_extracting_internals/simulation/extract_persona_vector.py
python examples/demonstration_suite/05_extracting_internals/simulation/probe_persona_direction.py
python examples/demonstration_suite/05_extracting_internals/simulation/steer_generation.py
```

## See also

- [Advanced Demonstration](06_advanced_demonstration.md) — builds on this
  same internals-extraction backend
