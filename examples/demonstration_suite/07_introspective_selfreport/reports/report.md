# Results: Introspective Self-Report, Adapted from Plunkett et al. (2025)

We ran 10 simulated agents (fictional characters, 5-attribute choice domains --
condos, vacations, laptops, used cars) through a two-stage protocol against
`smollm2:360m-instruct-fp16` on local Ollama, real calls throughout, no
mocking: (1) an in-context few-shot preamble (8 worked example choices,
computed from a randomly-generated target attribute-weight vector never shown
to the model) standing in for the paper's fine-tuning step, followed by 6
decision trials (2AFC + a 1-4 confidence rating) and 4 introspection-report
repeats per agent; (2) for 5 of the 10 agents ("trained"), the introspection
prompt was additionally preceded by a worked-example block of two *other*
agents' accurate self-reports -- the in-context analog of the paper's
Experiment 2 fine-tuning-to-improve-introspection. See `readme.md` for why
these two substitutions (in-context vs. gradient fine-tuning) were made.

## Self-report accuracy (the paper's own headline measure)

Pearson r between each agent's averaged reported attribute weight and its
true target weight, computed separately for the two conditions (n=25
agent-attribute points each, 5 attributes x 5 agents per condition):

| Condition | r | p | paper's own r (GPT-4o / 4o-mini) |
|---|---|---|---|
| control (no worked-example preamble) | 0.031 | 0.885 | 0.54 / 0.50 (Exp 1, before training) |
| trained (worked-example preamble) | 0.223 | 0.283 | 0.74 / 0.75 (Exp 2, after training) |

Both correlations are statistically indistinguishable from zero at this
sample size, and the "trained" condition's r is *higher* than control's but
well within noise (p=0.283) -- so this run does not reproduce the paper's
own-preference-tracking finding at any model scale, let alone the training
improvement it reports. Two things this demo already flagged as expected
going in: (a) the paper's effect was found in GPT-4o/4o-mini, not a 360M
local model, and (b) the "trained" condition here is a same-context
worked-example prompt, not the gradient fine-tuning the paper's Experiment 2
actually used, so even a much stronger model wouldn't be expected to fully
reproduce the paper's post-training r.

A concrete failure mode visible in `analysis/reported_vs_target_weights.png`:
7 of the 50 reported-weight points fall outside the requested [-100, 100]
scale (as far out as 4060) -- `smollm2:360m` doesn't reliably respect the
numeric range it's asked to report in, on top of not tracking the target
weight's sign/magnitude. The plot clips to ±100 (matching the paper's own
Fig. 2/3 axes) for readability; the clipping itself is the finding.

## Decision trials + SDT/metacognition extension (beyond the paper)

Choice accuracy (model's 2AFC pick matches the option the agent's target
weights actually favor): **0.367** (22/60), below the 63.3% base rate of
"B" being correct across these 60 trials -- because the model chose **"A"
on all 60 trials**, regardless of content. It reported "very confident"
(mapped to confidence=4) on all 60 as well.

This constant-response pattern makes hit-rate and false-alarm-rate
degenerate (P(resp=A | stim=A) = P(resp=A | stim=B) = 1), so
`metasignal.stdpy.compute_all_measures` correctly returns undefined
(NaN) values for d', meta-d', and M-ratio -- not a bug in the analysis, a
real and fairly common failure mode for very small instruction-tuned
models on forced-choice tasks embedded in a long (~2500-character) system
prompt: a first-option bias so strong it overrides the actual attribute
content, paired with uniformly high stated confidence regardless of
correctness (i.e., the opposite of good metacognition -- confident and
wrong). This is exactly the kind of pattern the paper's own SDT-adjacent
discussion motivates studying, even though the paper itself never
elicited confidence or computed these measures.

## Takeaways

1. **The pipeline works end-to-end and is honest about its results.** Every
   number above came from a real local model run, not synthetic/mocked data
   -- including the null/degenerate ones. Swapping `MODEL_NAME`/`FAMILY` in
   `simulation/run_introspection_task.py` for a frontier model (e.g. Groq's
   `llama-3.1-8b-instant` like `01_reward_task`, or `gpt-4o-mini` with an
   API key) is the natural next step to see whether a more capable model
   both tracks its instilled preferences (r above ~0) and shows non-
   degenerate SDT measures.
2. **In-context preference instillation is a much weaker manipulation than
   fine-tuning**, at least at this model scale -- worth keeping in mind
   before treating this demo's numbers as a real test of the paper's claims.
3. **The metasignal SDT layer is doing its job correctly by returning NaN
   here** -- a model with a constant response has undefined discrimination
   by definition, and silently reporting a fake d'=0 would have been the
   actual bug. `analysis/analyze.py` now surfaces this as an explicit,
   readable finding instead of a bare NaN in `summary.txt`.

## Caveats / scope

Small demo run (10 agents, 6 decision trials + 4 introspection reps each,
`N_FEWSHOT_EXAMPLES=8`), not a statistically powered study. `smollm2:360m`
was chosen for consistency with the rest of the repo's free/local/no-API-key
examples (README quickstart, VVIQ-16 study), not for its suitability to this
particular task -- see Takeaway 1. The "trained" vs. "control" split is
between-agent (5 agents each), not within-agent before/after, to avoid
prompt-order contamination; a larger agent count would give this comparison
more power.
