# Results: Memory Architecture and Feedback in Paired-Associate Recall

We ran the PAL-50 paired-associate-learning task (`examples/tasks/pal50.json`,
tasktype "episodic" — 50 word pairs across 5 semantic-similarity levels,
study phase then test phase) through psychscanner's `ExpCard`/`ScannerModel`
pipeline across all 5 features this demo exists to show: single-turn
(`memory=SingleTurn`), trial-chain (`memory=Convo, chain_type="trial"`),
episodic-chain (`memory=Convo, chain_type="task"`), and episodic-chain with
two `FeedbackBase` handlers (`CorrectIncorrectFeedback`,
`RewardFeedback`) added after each test trial. A balanced 3-pairs-per-level
subset (15 pairs, 30 trials/run) ran on two providers — Groq
(`openai/gpt-oss-120b`) and OpenRouter (`meta-llama/llama-3.1-8b-instruct`)
— real API calls throughout, no mocking. 9 of the 10 planned condition×model
runs completed; `feedback_reward` on the Groq model failed all 3 retry
attempts with a persistent `output_parse_failed` 400 from Groq's structured-
output endpoint (empty `failed_generation`, so no diagnosable payload was
returned) — a Groq-side flakiness specific to that condition, not reproduced
on any of that same model's other 4 conditions. That cell is missing from
the results below rather than papered over.

**Memory architecture is the dominant effect, by a wide margin.** Recall
accuracy on trials with *any* form of memory carry-over — trial-chain,
episodic-chain, and both feedback conditions — was a clean 100% (30/30,
30/30, 15/15, 30/30 test trials respectively, pooled across both models).
Single-turn accuracy, where each trial is a stateless call with zero
carry-over from the study phase, was 12.5% (3/24 pooled) — effectively
floor, since a model with no memory of the study phase has no principled
way to answer. The two models even failed in qualitatively different ways
at floor: Groq's `gpt-oss-120b` answered literally `"unknown"` on every one
of the 9 single-turn test trials it managed to parse at all (0% correct,
21/30 rows had no parseable response), while OpenRouter's
`llama-3.1-8b-instruct` guessed real words and got 3/15 right by chance
(20%). The `"unknown"` pattern is arguably the more honest failure mode —
the model correctly recognized it had no basis for an answer rather than
confabulating one — but it does mean the two models aren't directly
comparable at floor.

**Feedback type does not move the needle here, because there's no headroom
left to move.** `feedback_ci` and `feedback_reward` matched episodic-chain's
100% exactly. This demo's task design can't separate "does feedback help"
from "does memory help," because every feedback condition already inherits
full episodic memory (`chain_type="task"`) before feedback is layered on —
the ceiling was already reached by memory alone. Distinguishing a feedback
effect from a memory effect needs a design with headroom below ceiling
(e.g. the harder, semantically-unrelated-pair-only version of this task, or
the `advanced/exp2_1_with_feedback` / `exp2_2_no_feedback` sub-experiments
already scaffolded alongside this demo, which use a smaller, more
error-prone local model specifically so accuracy isn't saturated).

**Similarity level shows a small, likely confounded trend.** Pooled across
all 5 conditions, accuracy rose roughly monotonically from 77.8% (unrelated
pairs, similarity 0.0) to 88.5% (similarity 0.75), dipping slightly at 1.0
(84.0%). Because this pooled number mixes a floor condition (single-turn)
with four ceiling conditions, most of this "trend" is single-turn's own
similarity-level accuracy climbing from 0% to 40% before falling back to 0%
at similarity 1.0 (see `accuracy_by_condition_x_similarity.csv`) — every
memory-carrying condition is flat at 1.0 regardless of similarity. With
only 24 single-turn trials this is far too small a sample to read as a real
similarity effect; it's noted here as a candidate hypothesis for a
larger-n follow-up, not a finding.

**Confidence is uninformative as a calibration signal at these sample
sizes.** Every memory-carrying condition reports a single confidence value
for all-correct trials (episodic-chain and feedback_ci both report a flat
mean of 3/6, feedback_reward 5/6, trial-chain 3.43/6) — there's no
incorrect-trial confidence to contrast against, since those conditions have
no errors. Single-turn's confidence (1/6, correct and incorrect alike) is
the only case with variation, but with only 3 correct trials in that
condition, this isn't enough to say anything about calibration one way or
the other.

**Takeaways.** (1) The demo confirms the mechanical claim: the same
`ExpCard`/`ScannerModel`/`FeedbackBase` pipeline that ran Demo 01's agent
comparison runs unchanged across memory type and feedback handler — only
`card.memory`/`card.chain_type`/`card.feedback_fn` differ between
conditions. (2) At this task's default difficulty (real study-then-test
pairs, no distractors), memory architecture alone saturates accuracy the
moment any conversational carry-over exists, leaving no room to observe a
feedback effect — the harder `advanced/` sub-experiments exist precisely to
open up that headroom. (3) Groq's `gpt-oss-120b` produced a real, persistent
structured-output failure specific to the `feedback_reward` condition
(3/3 retries), worth tracking if this task shape is reused for a Groq-only
run in the future.

## Reproduce

```bash
source .venv311/bin/activate
python examples/demonstration_suite/02_association_memory/simulation/run_simulation.py
python examples/demonstration_suite/02_association_memory/analysis/analyze.py
```

## Data

- `data/{condition}__{family}-{model}.csv` — 9 raw per-condition-per-model
  CSVs (`feedback_reward__groq-openai-gpt-oss-120b.csv` missing, see above)
- `analysis/scored_trials.csv` — 129 scored test trials, all conditions/models
- `analysis/accuracy_by_condition.csv`, `analysis/accuracy_by_condition_x_similarity.csv`
- `analysis/analysis_output.txt` — full text summary
