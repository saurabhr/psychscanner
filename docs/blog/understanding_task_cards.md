# Understanding Task Cards in PsychScanner

*Draft for Substack. Runnable companion files live in
[`examples/tasks/tutorial/`](../../examples/tasks/tutorial/).*

A **task card** is the file that describes an experiment to PsychScanner: the
instructions, the trials, and how a trial should be scored. Everything else -
which model to run it on, how many simulated participants, whether it
remembers past trials - lives outside the card, on the run config
(`ExpCardInit`). This post starts from the simplest possible card and adds one
idea at a time.

## 1. The simplest card: a survey

[`examples/tasks/example_survey.json`](../../examples/tasks/example_survey.json)
rates two personality dimensions:

```json
{
    "tasktype": "survey",
    "taskname": "example_survey",
    "instructions": {"definition": ["You will rate your agreement..."]},
    "contexts": ["Openness to Experience", "Conscientiousness"],
    "contexts_id": ["O", "C"],
    "context_present": true,
    "items": {
        "O_1": [{"trcode": "O_1", "stimulus": "I enjoy trying new things..."}],
        "C_1": [{"trcode": "C_1", "stimulus": "I am organized..."}]
    },
    "parser": "DefaultLiteralAgree",
    "chain_type": "item"
}
```

`items` groups trials; each trial needs a unique `trcode` and a `stimulus`.
That's the whole schema a beginner needs.

## 2. Context item and memory - for free, and by choice

**Context item** isn't something you write - it's derived. The text before
the first `_` in a trial's `trcode` (`"O"` in `"O_1"`) is looked up in
`contexts_id`, and the matching string from `contexts` (`"Openness to
Experience"`) is attached to that trial automatically as `context_item`. It
gets prepended to the prompt because `context_present: true`, and it's saved
back onto every result row - useful for grouping analysis by dimension
without re-deriving it later.

**Memory**, by contrast, is a run-time choice, not a card field:

```python
card = ExpCardInit(..., memory="SingleTurn")  # each trial: a fresh conversation
card = ExpCardInit(..., memory="Convo")       # trials share one conversation
```

`SingleTurn` is the right default for a questionnaire - you don't want trial 5
influenced by however the model answered trial 2. Reach for `Convo` when the
task itself is about continuity (see step 4).

## 3. Three ways trials can be chained

The card field that actually changes structure is `chain_type`. It picks
which trials share a LangGraph conversation thread. PsychScanner's demos call
the three modes **Single-Turn**, **Trial-Chain**, and **Episodic-Chain**; here
they are on the same O/C survey so only the chaining changes.

**Single-Turn** (`chain_type: "item"`) - what step 1 already is. One stimulus,
one call, fully independent trials.

**Trial-Chain** (`chain_type: "trial"`,
[`02_trial_chain_survey.json`](../../examples/tasks/tutorial/02_trial_chain_survey.json)) -
sub-stimuli that share a `trcode` run as one mini-conversation, then the next
`trcode` starts fresh. Here each item is followed by a "justify your rating"
turn that can see the rating it's justifying, but item `O_1`'s conversation
never leaks into item `C_1`'s:

```json
"O_1": [
    {"trcode": "O_1", "stimulus": "I enjoy trying new things and exploring new ideas."},
    {"trcode": "O_1", "stimulus": "Now justify the rating you just gave in one sentence."}
]
```

**Episodic-Chain** (`chain_type: "task"`,
[`03_episodic_chain_survey.json`](../../examples/tasks/tutorial/03_episodic_chain_survey.json)) -
every trial in the whole run shares one conversation. The last item can
reference the first:

```json
"C_2": [{"trcode": "C_2", "stimulus": "Given your earlier answers, do you follow through on commitments?"}]
```

`chain_type` alone doesn't create memory - pair it with `memory="Convo"` on
the run config, or the chaining has nothing to carry over. For tasks that also
need to inject new instructions mid-conversation (e.g. switching from an
encoding phase to a test phase), see `tasktype: "episodic_system"` and the
per-trial `system_message` key in
[`rm_episodic_demo.json`](../../examples/tasks/rm_episodic_demo.json) - the
same episodic-chain idea, with the system prompt evolving alongside it.

## 4. Advanced: multimedia, tools, and generating a card from a script

Everything above is easy to hand-write as JSON. Once a card needs generated
paths, per-trial tool subsets, or looping logic, a Python script that returns
the same dict is less error-prone than hand-editing JSON. See
[`04_multimodal_tools_survey.py`](../../examples/tasks/tutorial/04_multimodal_tools_survey.py):

```python
"O_2": [{
    "trcode": "O_2",
    "stimulus": [
        image_block(STIMULI_DIR / "mood_board.png"),
        {"type": "text", "text": "Rate how much this image reflects your openness to new experiences."},
    ],
    "tools": ["image_zoom"],
}]
```

`image_block()` reads a local file and returns a standard content block -
useful in a script. If you're staying in pure JSON, the equivalent is a
`{"type": "image", "path": "stimuli/mood_board.png"}` block, resolved the same
way at run time. `tools` on a trial narrows which of the run's tools
(`ExpCardInit(tools=[...])`) that specific trial can call - `image_zoom` here
is only bound on the image trial, not the plain-text ones.

**Rule of thumb:** small, static cards → JSON. Cards with computed stimuli,
many near-duplicate trials, or conditional structure → a script that builds
the same dict.

## 5. A parser for the response you actually want

`DefaultLiteralAgree` only captures a bare rating. Once trial 2 asks for a
justification too (step 3), the default parser can't see it. Custom parsers
are just Pydantic models -
[`05_custom_parser.py`](../../examples/tasks/tutorial/05_custom_parser.py):

```python
class AgreeWithReason(BaseModel):
    """Rate your agreement with the statement and justify it briefly."""
    rating: Literal[1, 2, 3, 4, 5] = Field(..., description="1 = strongly disagree, 5 = strongly agree.")
    reason: str = Field(..., description="One sentence justifying the rating.")
```

Pass the class directly as `parser=AgreeWithReason` on `ExpCardInit` - no
registration needed unless you want to reference it by name string from JSON.
See [`docs/guides/custom_parsers.md`](../guides/custom_parsers.md) for
per-trial parser routing when different trials need different schemas.

## Where to go next

- [`docs/guides/cognitive_tasks.md`](../guides/cognitive_tasks.md) - full task
  JSON schema and a worked Reality Monitoring example.
- [`docs/guides/memory_types.md`](../guides/memory_types.md) - `memory` /
  `chain_type` combinations, with a table of which pairing fits which
  experiment.
- [`docs/guides/task_library.md`](../guides/task_library.md) - share a task
  card with `task_library()` instead of a hardcoded path.
