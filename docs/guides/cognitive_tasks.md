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
| `stimulus` | Yes | The prompt shown to the model: a string, a structured dict, or a **list of standard content blocks** (e.g. `image_block(...)` + `{"type": "text", ...}`) for multimodal trials — see [Visual Search and Attention](#visual-search-and-attention) below |
| `fb` | No | Whether this trial receives feedback (default: `True`) |
| `corrAns` | No | Correct answer for validation |
| `parser` | No | Per-trial parser name override |
| `tools` | No | Per-trial tool subset: a list of tool names selecting from `card.tools` (see [Tool Binding](#tool-binding) below). Absent → full `card.tools` pool; `[]` → no tools for this trial |

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

## Visual Search and Attention

Visual search / attention paradigms (feature search, conjunction search,
sustained-attention vigilance blocks) need an **image stimulus**, not just
text. The `stimulus` field accepts a list of standard content blocks
(https://docs.langchain.com/oss/python/langchain/messages) — build them with
`psychscanner.datasets.prompts.multimodal.image_block` (also
`audio_block`/`file_block`/`website_block` for other modalities):

```python
from psychscanner.datasets.prompts.multimodal import image_block

def search_trial(trcode, image_path, question):
    return {
        "trcode": trcode,
        "stimulus": [image_block(image_path), {"type": "text", "text": question}],
    }
```

The same multimodal stimulus works under all three memory/`chain_type`
combinations the engine supports — see [Memory Types](memory_types.md#chain_type)
for how `chain_type` picks the LangGraph `thread_id`
(`TaskRunner.execute`, `chain_type == "trial"` → `trace_cfg["trial"] + trcode`;
`chain_type in ("item", "task")` → `trace_cfg["task"]`). `thread_id` only has
an effect under `memory="Convo"` (a checkpointer is attached); under
`SingleTurn` it's silently ignored.

| Configuration | `memory` | `chain_type` | Use for |
|---|---|---|---|
| Single trial | `SingleTurn` | `"item"` | Feature/pop-out search — each display independent |
| Trial-chain | `Convo` | `"trial"` | Serial attention scan within one display |
| Episodic chain | `Convo` | `"task"` | Sustained-attention / oddball vigilance block |

### Single trial (feature search)

Each display is an independent trial with no memory carry-over — standard
for a cross-sectional accuracy/confidence design:

```python
task = {
    "tasktype": "visual_search", "taskname": "feature_search",
    "instructions": {"definition": ["Decide whether the red-circle target is present among the blue-circle distractors."]},
    "contexts": ["Feature search (pop-out)"], "contexts_id": ["feat"],
    "context_present": False, "chain_type": "item",
    "parser": "DefaultResponseRating",
    "items": {"feat": [
        search_trial("feat_1", "stimuli/feature_present_01.png", "Is the target present? Answer yes/no and confidence 1-5."),
        search_trial("feat_2", "stimuli/feature_absent_01.png",  "Is the target present? Answer yes/no and confidence 1-5."),
    ]},
}
card = ExpCardInit(task_file=task, memory="SingleTurn", chain_type="item", parser="1", cogtype="no", nsim=20)
```

### Trial-chain (within-trial attention scan)

Models serial covert attention shifts across quadrants of one
conjunction-search display. Sub-stimuli **share the same `trcode`**
(`conj_1` repeated) so they land on the same LangGraph thread and accumulate
memory only within that trial — the next trial (`conj_2`) starts a fresh
thread:

```python
"items": {"conj": [
    search_trial("conj_1", "stimuli/conjunction_TL.png", "Top-left quadrant — note any target features, then say NEXT."),
    search_trial("conj_1", "stimuli/conjunction_TR.png", "Top-right quadrant — note any target features, then say NEXT."),
    search_trial("conj_1", "stimuli/conjunction_BL.png", "Bottom-left quadrant — note any target features, then say NEXT."),
    search_trial("conj_1", "stimuli/conjunction_BR.png", "Bottom-right quadrant. Now judge: was the target present anywhere? yes/no + confidence."),
]}
```
```python
card = ExpCardInit(task_file=task, memory="Convo", chain_type="trial", parser="1", cogtype="no", nsim=20)
```

### Episodic chain (sustained-attention / oddball block)

A vigilance block of mostly-standard displays with rare oddball targets. The
whole block is one conversation, so expectation/fatigue effects can
accumulate across trials — optionally paired with `FeedbackBase` for
correctness feedback, same pattern as `RMFeedback` above:

```python
"items": {"vig": [
    search_trial("vig_1", "stimuli/standard_01.png", "Oddball present? yes/no."),
    search_trial("vig_2", "stimuli/standard_02.png", "Oddball present? yes/no."),
    search_trial("vig_3", "stimuli/oddball_01.png",  "Oddball present? yes/no."),
    # ... continues for the full block
]}
```
```python
card = ExpCardInit(task_file=task, memory="Convo", chain_type="task", parser="1",
                    feedback=True, feedback_fn=VigilanceFeedback, cogtype="no", nsim=20)
```

### Media storage

Large stimuli are stored once, content-addressed, under
`<data_root_dir>/media/<sha256>.<ext>`, and referenced by path in the
persisted `.psyscan` record — the same image reused across many trials or
`nsim` participants is written only once, not duplicated into every
checkpoint.

---

## Tool Binding

`card.tools` binds LangChain tools to the model for the whole run — it's a
single hyperparameter, like `parameters`, not a swept condition: every
persona and every trial in one `ExpCard` shares the same tool pool.

```python
from langchain_core.tools import tool

@tool
def image_zoom(region: str) -> str:
    """Return a zoomed-in crop of the display for the named region."""
    ...

card = ExpCardInit(task_file=task, tools=[image_zoom], cogtype="no", nsim=20)
```

To vary which tools are *available* trial-by-trial within that same run —
e.g. only expose `image_zoom` on trials where zooming is part of the
paradigm — add a `"tools"` key to individual trial dicts, naming a subset of
`card.tools` by their `BaseTool.name`:

```python
"items": {"feat": [
    {"trcode": "feat_1", "stimulus": [...], "tools": ["image_zoom"]},  # can zoom
    {"trcode": "feat_2", "stimulus": [...], "tools": []},              # no tools this trial
    {"trcode": "feat_3", "stimulus": [...]},                            # falls back to card.tools
]}
```

An unrecognized name in a trial's `"tools"` list raises `ValueError` rather
than silently binding nothing — it's almost always a typo against
`card.tools`. To compare *whether tools exist at all* as an experimental
condition (rather than which trials can use them), run two `ExpCard`s — one
with `tools=[...]`, one with `tools=None` — the same way you'd vary
`model` or `memory`.

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

- [PsychScanner Workflow](psychscanner_workflow.md) — the experiment card
  (`ExpCard`) that wraps a task card with model, memory, and persona config
- [Survey Tasks](survey_tasks.md) — simple questionnaire tasks
- [Cognitive Tasks with SweetPea](sweetpea_task_cards.md#3-convert-the-sampled-sequence-into-items) —
  generate a counterbalanced `items` sequence instead of hand-writing one,
  then export it into a task card exactly like the ones on this page
- [Task Library](task_library.md) — fetch a saved task card by name once
  it's written to disk, instead of hardcoding a path
- [Running a Survey with Persona Levels](../examples/multi_persona.md) —
  crossing a task card with multiple simulated personas
- [Memory Types](memory_types.md) — Convo vs. SingleTurn
- [Parsers API](../api/parsers.md) — RM parser classes
- [Feedback API](../examples/feedback_loop.md) — feedback examples
