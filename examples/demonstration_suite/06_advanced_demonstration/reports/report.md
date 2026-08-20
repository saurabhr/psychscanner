# Results: ROME-lite — Causal Tracing and Rank-One Editing on GPT-2

We ran `simulation/rome/run_rome_lite.py`, a toy-scale reproduction of Meng
et al. 2022 (arXiv:2202.05262, "Locating and Editing Factual Associations in
GPT") on GPT-2 small (12 layers) via nnsight's `trace()` intervention API —
the same mechanism `src/psychscanner/nnsight_backend/chat_nnsight.py` uses
elsewhere in the framework. Two deliberate simplifications vs. the original
paper are stated up front in the script itself: causal tracing patches the
full residual stream per (layer, token position) rather than separating
attention/MLP contributions, and the rank-one edit uses the identity matrix
in place of ROME's covariance-based regularizer C — so collateral effects on
unrelated prompts are expected to be larger than the original paper reports,
and are measured directly rather than assumed away.

**Fact verification.** Of 15 candidate capital-city / founder facts, GPT-2
small answered top-1 correctly on 4: "The capital of Italy is [Rome]"
(p=0.157), "The Eiffel Tower is located in the city of [Paris]" (p=0.064),
"The capital of Spain is [Madrid]" (p=0.105), and "Microsoft was founded by
Bill [Gates]" (p=0.863, by far the most confident). The other 11 — Japan,
Germany, Russia, China, England, Egypt, Canada, Portugal, the Great Wall,
and the Colosseum — all failed top-1 (mostly defaulting to "the"), so
GPT-2-small's factual recall for this fact category is closer to 25% than
the near-ceiling behavior larger models show.

**Causal tracing localizes the fact to the subject token, as expected.**
Across the 4 traced facts, restoring clean activations at the last subject
token had by far the largest effect at layers 5, 6, 3, and 0 respectively
(`best_layer_at_last_subject_token_across_facts`) — a spread rather than one
universal "fact layer," consistent with GPT-2-small's shallow 12-layer
depth giving less room for the layer-15-ish mid-depth peak the original
paper reports on GPT-2-XL/GPT-J. 3 of 4 facts also showed the expected
trivial effect of the global argmax concentrating at the final token
position (noted directly in the output rather than treated as the
meaningful signal — see `causal_trace_summary.note`).

**Rank-one edits: 3/3 attempted succeeded, with real collateral damage.**
The script ran 3 hardcoded edit attempts (a subset of the 4 verified facts —
Microsoft/Gates wasn't included in the edit list): Italy→Paris (edit at
layer 5, target Rome→Paris), Eiffel Tower→Rome (layer 6, Paris→Rome), and
Spain→Berlin (layer 3, Madrid→Berlin). All 3 top-1 outputs flipped to the
new target exactly as intended. But every edit also changed the model's
answer on some of the 6 unrelated control prompts — 3/6, 2/6, and 3/6
respectively, a mean collateral rate of 44.4%. This is the direct,
predicted consequence of skipping ROME's covariance regularizer: the
identity-matrix edit is not surgical, and the script's own docstring says
so before a single edit runs. This collateral rate is the headline number
this "lite" reproduction is built to measure, not an incidental side
finding.

**Takeaways.** (1) The mechanical claim holds: nnsight's `trace()` API
supports both read-only causal tracing and a real (if simplified) weight
edit, entirely from Python, on a genuinely small local model with no GPU.
(2) GPT-2-small's shallow depth means "the fact layer" isn't a single
consistent number across facts the way it is in larger-model ROME
demonstrations — layer localization here should be read per-fact, not
pooled. (3) The ~44% collateral-change rate is the direct, expected cost of
using identity in place of ROME's covariance statistic, and is reported
as the finding rather than a caveat to explain away.

## Reproduce

```bash
source .venv311/bin/activate
python examples/demonstration_suite/06_advanced_demonstration/simulation/rome/run_rome_lite.py
```

## Data

- `data/rome/rome_lite_results.json` — verified facts, full causal-trace
  grids, and per-edit pre/post top-1 + collateral counts
