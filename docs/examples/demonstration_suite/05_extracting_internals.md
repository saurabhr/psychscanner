# Extracting Internals

**Status: Skeleton** — not yet run.

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

## See also

- [Advanced Demonstration](06_advanced_demonstration.md) — builds on this
  same internals-extraction backend
