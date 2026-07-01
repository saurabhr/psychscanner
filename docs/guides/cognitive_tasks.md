# Cognitive Tasks

PsychScanner is designed for validated cognitive paradigms.
This guide covers the Reality Monitoring (RM) paradigm in detail and provides
a template for building other multi-phase cognitive experiments.

---

## Task JSON schema

Every task is defined in a JSON file (or inline dict). The top-level keys are:

```json
{
  "tasktype"        : "survey",
  "taskname"        : "my_task",
  "instructions"    : { "definition": ["…"] },
  "contexts"        : ["Scene 1 description", "Scene 2 description"],
  "contexts_id"     : ["encode", "test"],
  "context_present" : true,
  "chain_type"      : "item",
  "parser"          : "Response_part_1_rm",
  "items": {
    "encode": [
      { "trcode": "encode_1", "stimulus": { "Word_Pair": { "word_1": "APPLE", "word_2": "FRUIT"  } } },
      { "trcode": "encode_2", "stimulus": { "Word_Pair": { "word_1": "TABLE", "word_2": "____"   } } }
    ],
    "test": [
      { "trcode": "test_1",   "stimulus": "APPLE — was the second word internally or externally generated?" },
      { "trcode": "test_2",   "stimulus": "TABLE — was the second word internally or externally generated?" }
    ]
  }
}
```

### Key fields

| Field | Description |
|-------|-------------|
| `tasktype` | Arbitrary label (`"survey"`, `"imagery"`, etc.) |
| `taskname` | Used as a sub-folder name in output paths |
| `instructions` | String, list of strings, or `{"definition": [...]}` dict shown in the system message |
| `contexts` | Full-text descriptions of task contexts (shown to the agent) |
| `contexts_id` | Short IDs — must match the `trcode` prefix (text before the first `_`) of each trial, not the `items` dict keys |
| `context_present` | Whether to include context text in the prompt |
| `chain_type` | `"item"`, `"trial"`, or `"task"` — see [Memory Types](memory_types.md#chain_type) |
| `parser` | Registered parser class name used when `card.parser = "1"` |
| `items` | Dict of trial-group labels (keys are arbitrary, not used for context lookup); each value is a list of trial dicts |

### Trial dict fields

| Field | Required | Description |
|-------|----------|-------------|
| `trcode` | Yes | Unique trial identifier. The text before the first `_` is used to look up the trial's context in `contexts_id` |
| `stimulus` | Yes | The prompt shown to the model (string or structured dict) |
| `fb` | No | Whether this trial receives feedback (default: `True`) |
| `corrAns` | No | Correct answer for validation |
| `parser` | No | Per-trial parser name override |

---

## Reality Monitoring task

The Reality Monitoring (RM) paradigm tests whether an agent can distinguish
between self-generated (imagined) and externally provided information.

### Encoding phase

The agent is shown word pairs. For **perceived** trials, both words are given.
For **imagined** trials, only the first word is given and the agent must generate the second.

```json
{ "trcode": "encode_perceived_1",
  "stimulus": { "Word_Pair": { "word_1": "APPLE", "word_2": "FRUIT"  } } }

{ "trcode": "encode_imagined_1",
  "stimulus": { "Word_Pair": { "word_1": "TABLE", "word_2": "____"   } } }
```

Parser: `Response_part_1_rm` — returns `Word_2` (str) + `Rating` (0–100 relatedness).

### Test phase

The agent is shown only the first word and must judge whether the second word
was internally or externally generated, and rate confidence.

```json
{ "trcode": "test_1",
  "stimulus": "APPLE — was the second word internally or externally generated?" }
```

Parser: `Response_part_2_rm` — returns `Judgment` (internal/external) + `Confidence` (1–6).

### Multi-phase setup

Run encoding and test phases in a single `Convo` experiment:

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import Response_part_1_rm, Response_part_2_rm

def rm_parser(trcode: str):
    """Route parser by trial type."""
    return Response_part_2_rm if "test" in trcode else Response_part_1_rm

card = ExpCardInit(
    model      = "llama3.1:8b",
    family     = "ollama",
    parameters = {"temperature": 0},
    task_file  = Path("tasks/rm_task.json"),
    memory     = "Convo",
    chain_type = "item",
    parser     = rm_parser,
    cogtype    = "no",
    nsim       = 20,
    proj_dir   = Path("./results"),
    projectname = "rm_study",
    tunnel_status = "1",
)

scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

### RM with feedback

Add trial-level correctness feedback using `FeedbackBase`:

```python
from psychscanner import FeedbackBase
import json

class RMFeedback(FeedbackBase):
    def on_response(self, trial: dict, response: dict) -> str:
        word2 = trial["stimulus"]["Word_Pair"]["word_2"]
        given = response.get("Word_2", "")
        if "__" in word2:   # imagined trial
            fb = "CORRECT — novel word generated." if given else "INCORRECT — no word provided."
        else:               # perceived trial
            fb = "CORRECT." if given.lower() == word2.lower() else f"INCORRECT — expected '{word2}', got '{given}'."
        return json.dumps({"feedback": fb})

fb_card = ExpCardInit(
    model       = "llama3.1:8b",
    family      = "ollama",
    task_file   = Path("tasks/rm_task.json"),
    memory      = "Convo",
    chain_type  = "task",
    parser      = rm_parser,
    feedback    = True,
    feedback_fn = RMFeedback,
    cogtype     = "no",
    nsim        = 20,
    proj_dir    = Path("./results"),
    projectname = "rm_feedback_study",
)

scanner = ScannerModel(expcard=ExpCard(fb_card))
results = scanner.run(progress_bar=True)
```

Feedback settings must be passed to a fresh `ExpCardInit`/`ExpCard` — mutating
a `card` object after it's already been passed to `ExpCard()` has no effect on
that (or any already-run) `ScannerModel`.

---

## VVIQ-16 (built-in)

The Vividness of Visual Imagery Questionnaire (VVIQ-16) is included as the
default task. It runs automatically when no `task_file` is set.

```python
card = ExpCardInit(
    model   = "gpt-4o-mini",
    family  = "openai",
    cogtype = "no",
    nsim    = 50,
    parser  = "1",   # resolves DefaultLiteralVivid15 from task JSON
)
```

---

## Building a custom cognitive task

1. Create a task JSON with `contexts`, `contexts_id`, and `items`.
2. Choose a `parser` that matches your response format.
3. Set `memory` and `chain_type` to match your paradigm.
4. If feedback is needed, implement `FeedbackBase.on_response()`.

```python
task = {
    "tasktype": "imagery",
    "taskname": "mental_rotation",
    "instructions": {"definition": ["Imagine rotating the object and describe what you see."]},
    "contexts":    ["Cube", "Pyramid"],
    "contexts_id": ["cube", "pyramid"],
    "context_present": True,
    "chain_type": "item",
    "parser": "DefaultResponseRating",
    "items": {
        "cube":    [{"trcode": "cube_0",    "stimulus": "Rotate the cube 90° clockwise."}],
        "pyramid": [{"trcode": "pyramid_0", "stimulus": "Rotate the pyramid upside down."}],
    }
}

card = ExpCardInit(task_file=task, parser="1", ...)
```

---

## See also

- [Survey Tasks](survey_tasks.md) — simple questionnaire tasks
- [Memory Types](memory_types.md) — Convo vs. SingleTurn
- [Parsers API](../api/parsers.md) — RM parser classes
- [Feedback API](../examples/feedback_loop.md) — feedback examples
