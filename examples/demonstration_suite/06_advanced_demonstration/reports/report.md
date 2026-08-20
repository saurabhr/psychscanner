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

---

# Results: CCS — Contrast-Consistent Search on GPT-2

We ran `simulation/ccs/run_ccs.py`, a toy-scale reproduction of Burns et al.
2022/2023 ("Discovering Latent Knowledge in Language Models Without
Supervision") on GPT-2 small (12 layers), the third and last of Demo 06's
three interpretability reproductions. 20 declarative statements (10 true,
10 false; capitals, astronomy, physics, geometry, arithmetic — deliberately
spanning topics rather than negating one fact family, unlike ROME's
capital-city set, so truth value isn't confounded with subject matter) were
each templated as `"{statement}\nTrue or False? True"` and `"...False"`,
with the residual-stream activation captured at each prompt's last token,
every layer. A per-layer linear probe was trained with Burns et al.'s
unsupervised consistency+confidence loss (no truth label ever enters the
loss — see the script's own docstring for the exact objective) and,
separately, a supervised logistic-regression probe was trained directly on
the same activations with real labels, as an accuracy-ceiling reference.

**CCS collapsed to a trivial, non-truth-tracking solution — robustly, not
as a code bug.** With no regularization, the probe reached zero loss by
learning p(True-completion)=1.0 and p(False-completion)=0.0 for literally
every one of the 20 statements, true or false alike — it had simply learned
to detect which literal token ("True" vs "False") was present, the easiest
possible way to satisfy a loss built only from consistency and confidence,
with nothing in the objective preventing it. This is a real, previously
published critique of CCS in exactly this form (surface answer-token
identity is itself perfectly "consistent and confident," so the loss alone
can't distinguish it from genuine truth-tracking — see e.g. Levinstein &
Herrmann 2023, "Still No Lie Detector for Language Models"), now
empirically reproduced rather than assumed. We tested the standard
mitigation, weight decay, sweeping 0/0.01/0.1/1/10 on one layer before
settling on 1.0 as the script's default: even at decay=10, p(True) stayed
uniformly high (0.83–0.88) and p(False) uniformly low (0.12–0.14) across
*all* 20 statements regardless of truth value — accuracy resolved to
exactly 0.500 at every one of the 12 layers, with or without regularization.
The collapse is structural, not a training-instability fluke.

**The supervised "ceiling" baseline also failed to generalize — a second,
distinct finding.** With only 20 examples in 768 GPT-2-small dimensions,
even a single linear layer is wildly overparameterized; a fixed 80/20 split
would put only 4 examples in the test set (quantized to steps of 0.25 and
noisy enough to flip on one example), so this was run as leave-one-out
cross-validation (train on 19, test on 1, averaged over all 20 folds)
instead. Real-label LOO accuracy ranged 0.0–0.3 across layers — at or
*below* its own shuffled-label control (0.40–0.60, correctly centered near
chance, confirming the harness itself isn't biased) at every single layer.
This is a distinct failure mode from CCS's: not a degenerate-solution
collapse, but a sample-size/dimensionality problem — 19 training points in
a 768-dimensional space is too underdetermined for a linear probe to
generalize at all, whether the signal it's chasing is real or spurious.

**Takeaways.** (1) The mechanical claim holds: nnsight's `trace()` API and
a torch-only linear probe (same no-scikit-learn pattern as
`run_othello_probe.py` and `probe_persona_direction.py`) support CCS's
exact unsupervised objective, entirely on a small local model with no GPU.
(2) The demo reproduces a genuine, previously-documented weakness of CCS —
its loss alone doesn't rule out latching onto surface answer-token identity
— rather than a successful "truth direction," and does so with an explicit,
disclosed regularization sweep rather than reporting the first result that
compiled. (3) The supervised control meant to sanity-check CCS revealed its
own, independent problem (N=20 is too small for this probe/dimensionality
to generalize at all), which matters for reading the CCS null correctly:
the failure to beat chance isn't strong evidence GPT-2-small lacks any
truth-tracking representation, since the supervised ceiling couldn't detect
one either, at this sample size, even with real labels.

## Reproduce (CCS)

```bash
source .venv311/bin/activate
python examples/demonstration_suite/06_advanced_demonstration/simulation/ccs/run_ccs.py
```

## Data (CCS)

- `data/ccs/ccs_results.json` — per-layer CCS accuracy, supervised LOO-CV
  accuracy, and shuffled-label control
