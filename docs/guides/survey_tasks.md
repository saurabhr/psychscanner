# Survey Tasks

A **survey task** presents a series of items to simulated participants and collects
structured responses. This is the most common use case for PsychScanner.

---

## Task JSON structure

```json
{
  "tasktype"        : "survey",
  "taskname"        : "agreement_scale",
  "instructions"    : {
    "definition": [
      "You will read a series of statements.",
      "For each statement, rate your agreement on a scale from 1 to 5.",
      "1 = strongly disagree, 5 = strongly agree."
    ]
  },
  "contexts"        : ["Block 1"],
  "contexts_id"     : ["item"],
  "context_present" : false,
  "chain_type"      : "item",
  "parser"          : "DefaultLiteralAgree",
  "items": {
    "items": [
      { "trcode": "item_1", "stimulus": "I enjoy thinking deeply about abstract concepts." },
      { "trcode": "item_2", "stimulus": "I prefer concrete, practical problems." },
      { "trcode": "item_3", "stimulus": "I often consider multiple perspectives on an issue." }
    ]
  }
}
```

### Minimal survey checklist

| Field | Value for surveys |
|-------|------------------|
| `tasktype` | `"survey"` |
| `chain_type` | `"item"` (usually) |
| `context_present` | `false` unless contexts should appear in the prompt |
| `parser` | A Likert or rating parser from the registry |
| Each trial's `trcode` prefix | Text before the first `_` must appear in `contexts_id` (the `items` dict key itself is not used for lookup) |

---

## Running a survey

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv

card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    parameters  = {"temperature": 0},
    task_file   = Path("tasks/agreement_scale.json"),
    memory      = "SingleTurn",
    parser      = "1",            # resolve from task JSON's "parser" field
    cogtype     = "no",
    nsim        = 30,
    proj_dir    = Path("./results"),
    projectname = "agreement_pilot",
    enabletqdm  = True,
)

scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## Multi-context surveys

When items come from different contexts (e.g. different vignettes), define
multiple context groups:

```json
{
  "contexts"        : ["Vignette A — workplace scenario", "Vignette B — family scenario"],
  "contexts_id"     : ["work", "family"],
  "context_present" : true,
  "items": {
    "work":   [
      {"trcode": "work_1", "stimulus": "The manager acted appropriately."},
      {"trcode": "work_2", "stimulus": "The employee was treated fairly."}
    ],
    "family": [
      {"trcode": "family_1", "stimulus": "The parent made the right decision."}
    ]
  }
}
```

Setting `context_present: true` prepends the context description to each item prompt.

---

## Choosing a parser for surveys

| Survey type | Recommended parser |
|-------------|-------------------|
| 1–5 agreement (Likert) | `DefaultLiteralAgree` |
| 1–5 vividness (VVIQ) | `DefaultLiteralVivid15` |
| 0–10 vividness | `DefaultLiteralVivid010` |
| Free text + rating | `DefaultResponseRating` |
| Binary (yes / no) | Custom `Literal["yes","no"]` parser |
| Multiple choice | Custom `Literal[...]` parser |

See [Custom Parsers](custom_parsers.md) for how to write your own.

---

## Inline task (no JSON file)

Pass the task as a dict directly on the card:

```python
task = {
    "tasktype":         "survey",
    "taskname":         "openness_items",
    "instructions":     {"definition": ["Rate each statement 1 (disagree) to 5 (agree)."]},
    "contexts":         ["Openness"],
    "contexts_id":      ["open"],
    "context_present":  False,
    "chain_type":       "item",
    "parser":           "DefaultLiteralAgree",
    "items": {
        "items": [
            {"trcode": "open_1", "stimulus": "I have a vivid imagination."},
            {"trcode": "open_2", "stimulus": "I enjoy artistic experiences."},
        ]
    }
}

card = ExpCardInit(task_file=task, parser="1", cogtype="no", nsim=10, ...)
```

---

## Task template

Generate an empty skeleton with `get_task_template`:

```python
from psychscanner import get_task_template
import json

template = get_task_template()
print(json.dumps(template, indent=2))
# Edit and save to a .json file
```

---

## See also

- [Cognitive Tasks](cognitive_tasks.md) — multi-phase paradigms; also
  covers the task JSON schema this page's surveys are an instance of
- [PsychScanner Workflow](psychscanner_workflow.md) — the experiment card
  (`ExpCard`) that wraps a survey task with model, memory, and persona config
- [Running a Survey with Persona Levels](../examples/demonstration_suite/03_personality_survey.md) —
  crossing a survey task card with multiple simulated personas
- [Cognitive Tasks with SweetPea](sweetpea_task_cards.md) — generate a
  counterbalanced item sequence instead of hand-writing one
- [Task Library](task_library.md) — fetch a saved survey task by name
  once it's written to disk
- [Memory Types](memory_types.md) — SingleTurn vs. Convo
- [Parsers API](../api/parsers.md) — all bundled parser classes
- [Configuration Reference](../configuration.md) — full parameter list
