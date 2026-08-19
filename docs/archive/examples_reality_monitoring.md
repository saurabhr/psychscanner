# Reality Monitoring Example

The Reality Monitoring (RM) paradigm tests whether a model can distinguish between
self-generated (imagined) and externally provided information.

This matches notebook `05_rm_task.ipynb` in the examples directory.

---

## Overview

The task runs in two phases within a single conversation:

1. **Encoding** — the agent sees word pairs; for *imagined* trials the second word is blank and must be generated
2. **Test** — the agent is shown only the first word and must judge whether the second word was internal or external, with a confidence rating

---

## Task JSON

```json
{
  "tasktype"        : "survey",
  "taskname"        : "rm_task",
  "instructions"    : {
    "definition": [
      "You will complete a memory task in two phases.",
      "Phase 1: You will see word pairs. For some pairs the second word is blank — generate the first word that comes to mind.",
      "Phase 2: You will see only the first word and must judge whether the second word was given to you or you imagined it."
    ]
  },
  "contexts"        : ["Encoding Phase", "Test Phase"],
  "contexts_id"     : ["encode", "test"],
  "context_present" : true,
  "chain_type"      : "item",
  "parser"          : "Response_part_1_rm",
  "items": {
    "encode": [
      { "trcode": "encode_perceived_1",
        "stimulus": { "Word_Pair": { "word_1": "APPLE",  "word_2": "FRUIT"  } } },
      { "trcode": "encode_imagined_1",
        "stimulus": { "Word_Pair": { "word_1": "TABLE",  "word_2": "____"   } } },
      { "trcode": "encode_perceived_2",
        "stimulus": { "Word_Pair": { "word_1": "OCEAN",  "word_2": "WAVE"   } } },
      { "trcode": "encode_imagined_2",
        "stimulus": { "Word_Pair": { "word_1": "FOREST", "word_2": "____"   } } }
    ],
    "test": [
      { "trcode": "test_1", "stimulus": "APPLE — was the second word given to you (external) or did you imagine it (internal)?" },
      { "trcode": "test_2", "stimulus": "TABLE — was the second word given to you (external) or did you imagine it (internal)?" },
      { "trcode": "test_3", "stimulus": "OCEAN — was the second word given to you (external) or did you imagine it (internal)?" },
      { "trcode": "test_4", "stimulus": "FOREST — was the second word given to you (external) or did you imagine it (internal)?" }
    ]
  }
}
```

---

## Running the task

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import Response_part_1_rm, Response_part_2_rm

def rm_parser(trcode: str):
    """Route to the correct parser based on trial phase."""
    return Response_part_2_rm if "test" in trcode else Response_part_1_rm

card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    parameters  = {"temperature": 0},
    task_file   = Path("tasks/rm_task.json"),
    memory      = "Convo",          # required: carry encoding context into test
    chain_type  = "item",
    parser      = rm_parser,        # callable: dispatches per trcode
    cogtype     = "no",
    nsim        = 30,
    proj_dir    = Path("./results"),
    projectname = "rm_study",
    tunnel_status = "1",            # checkpoint in case run is interrupted
    enabletqdm  = True,
)

scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## Parser details

### Encoding phase — `Response_part_1_rm`

Returns two fields:

| Field | Type | Description |
|-------|------|-------------|
| `Word_2` | `str` | The second word (given or generated) |
| `Rating` | `float` (0–100) | Relatedness rating |

### Test phase — `Response_part_2_rm`

Returns two fields:

| Field | Type | Description |
|-------|------|-------------|
| `Judgment` | `"internal"` / `"external"` | Source judgment |
| `Confidence` | `int` (1–6) | Confidence rating |

---

## Analysing results

```python
import pandas as pd

df = pd.read_csv("results/rm_study/rm_task/openai_gpt-4o-mini_Convo/rm_study.csv")

# Encoding accuracy (perceived trials)
encode = df[df["trcode"].str.contains("perceived")]
print("Encoding accuracy:", (encode["Word_2"].str.lower() == encode["corrAns"].str.lower()).mean())

# Reality monitoring performance
test = df[df["trcode"].str.startswith("test")]
# Imagined items should get "internal" judgments
imagined_trcodes = ["test_2", "test_4"]
imagined = test[test["trcode"].isin(imagined_trcodes)]
print("RM hit rate:", (imagined["Judgment"] == "internal").mean())
```

---

## See also

- [Cognitive Tasks guide](../guides/cognitive_tasks.md) — full RM schema + feedback variant
- [Memory Types](../guides/memory_types.md) — why `Convo` is required
- [Session Recovery](../guides/session_recovery.md) — for large-scale runs
