# Results: Persona-Vector Extraction, Probing, and Steering (Sycophancy, GPT-2 small)

We ran a toy-scale reproduction of Chen et al. 2025's persona-vector method
("Persona Vectors: Monitoring and Controlling Character Traits in Language
Models", Anthropic) on GPT-2 small (12 layers) via nnsight's `trace()` API —
the same mechanism `src/psychscanner/nnsight_backend/chat_nnsight.py` uses
elsewhere in the framework, and the same primitive `run_rome_lite.py` (Demo
06) already relies on. One deliberate simplification is stated up front:
GPT-2 small is a base LM with no chat template or instruction-tuning, so
"persona" here is a short descriptive text prefix
("You are a flattering, sycophantic assistant...") rather than a
system-prompt field a chat-tuned model was trained to attend to specially —
a weaker, noisier operationalization than the paper's, measured rather than
assumed to transfer.

**Extraction.** For 12 common-misconception statements ("The earth is
flat.", "Vaccines cause autism.", ...), we built contrastive prompt pairs
(sycophantic-persona prefix vs. neutral-persona prefix + the same statement)
and captured the residual-stream activation at the prompt's last token,
every layer. The per-layer direction (sycophantic mean − neutral mean,
normalized) had a raw pre-normalization norm that grew smoothly with depth,
from 4.86 at layer 0 to 28.73 at layer 11 — consistent with GPT-2's known
residual-stream norm growth, not yet evidence of a meaningful "persona"
signal on its own.

**Probing: the direction is strongly linearly decodable.** A per-layer
logistic-regression probe (80/20 split, Adam, 400 steps — same pattern as
Demo 06's Othello probe) classified sycophantic- vs. neutral-persona
activations at 100% accuracy vs. a 20% shuffled-label baseline at 6 of 12
layers (1, 2, 3, 4, 6, and tied structurally with 7's 100%/40%), falling to
60% (still above the 20% shuffled baseline) by layers 8–11. Layer 1 was
selected as "best" by largest real-vs-shuffled gap (+0.80, tied with 2, 3,
4, 6 — layer 1 taken as the earliest of the tie). This much separability
this early is itself informative: at layer 1, the model has barely begun
processing the prompt, so what's linearly decodable there is more likely
the surface lexical difference between the two persona descriptions
("flattering, sycophantic" vs. "honest, objective") than an integrated
behavioral disposition — a hypothesis the steering test below tests
directly.

**Steering: no detectable behavioral shift, at any layer or magnitude
tested.** On 8 held-out statements never used for extraction or probing, we
generated three completions per statement from an identical neutral prompt
(no persona instruction) — unsteered, +8×direction ("sycophantic" push),
and −8×direction ("honest" push) — at the probe-selected layer (1). All 24
completions scored `neither` under a keyword-based agree/pushback
classifier (0% agree, 0% pushback, 0% mixed, 100% neither in every
condition); several statements produced byte-identical completions across
all three conditions. To rule out this being a magnitude or layer-choice
artifact rather than a real ceiling, we additionally swept scale ∈
{0, ±8, ±20, ±40} across layers {1, 4, 6, 9} on one held-out statement:
moderate scales (≤20) mostly reproduced the unsteered echo/hedge completion
unchanged; large scales (±40, ~5–8× the natural per-layer diff norm) never
produced agreement or pushback language either — instead the model
degenerated into incoherent or off-topic text ("The assistant is a
blue-shaped object that is a blue-shaped object...", "Assistant:\nAssistant:\nAssistant:...").
Pushing harder made the model breaks *before* it ever exhibits the intended
behavioral axis, at every layer tried.

**Why this is a real finding, not a broken demo.** GPT-2 small's
"Assistant:"-style completions for these statements are dominated by
echoing the input verbatim or hedging ("I'm not sure..."), regardless of
steering condition — a base LM never trained on instruction-following
dialogue doesn't appear to have a well-formed internal "agree vs. disagree"
behavioral repertoire for a linear direction to steer *toward*, even though
a direction correlated with the persona-description wording is easily
linearly decodable. This is consistent with Chen et al.'s method being
validated on instruction-tuned models, and directly tests (rather than
assumes) whether it transfers to a small non-instruction-tuned base model —
it doesn't, on this behavioral measure.

**Takeaways.** (1) The mechanical claim holds: nnsight's `trace()` API
supports contrastive-activation extraction, linear probing (via the same
torch-only pattern Demo 06 already established), and repeated-`trace()`
steered generation, entirely on a small local model with no GPU. (2) Linear
decodability of a contrastive direction (the probe result) and behavioral
controllability via activation steering (the steering result) are separate
claims — this demo's probe succeeds while its steering test fails, which is
itself the more informative result than either alone. (3) Chen et al.'s
method is validated on instruction-tuned models; this toy-scale, non-
instruction-tuned reproduction suggests that's not an incidental detail —
some minimum of instruction-following training may be a precondition for a
persona direction to be behaviorally steerable, not just decodable.

## Reproduce

```bash
source .venv311/bin/activate
python examples/demonstration_suite/05_extracting_internals/simulation/extract_persona_vector.py
python examples/demonstration_suite/05_extracting_internals/simulation/probe_persona_direction.py
python examples/demonstration_suite/05_extracting_internals/simulation/steer_generation.py
```

## Data

- `data/persona_vectors.pt` — raw contrastive activations + per-layer directions
- `data/probe_results.json` — per-layer probe accuracy vs. shuffled baseline
- `data/steering_results.json` — per-statement completions and agreement rates by condition
