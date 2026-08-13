# Multi-Persona Simulation

PsychScanner can simulate participants with distinct personas by loading persona
JSON files. Each persona statement becomes the system prompt for one simulated
participant.

---

## Persona file format

A persona file is a `.json` file with a `persona_statements` key:

```json
{
  "persona_statements": [
    "You are a 25-year-old college student who is highly imaginative and often daydreams.",
    "You are a 45-year-old accountant who is practical, detail-oriented, and rarely daydreams.",
    "You are a 30-year-old artist who experiences vivid mental imagery.",
    "You are a 60-year-old retired engineer who thinks concretely and systematically."
  ]
}
```

Save this as `personas/my_personas.json`.

---

## Running with custom personas

Set `cogtype="custom"` and pass the path(s) to your persona files:

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv

card = ExpCardInit(
    model          = "gpt-4o-mini",
    family         = "openai",
    task_file      = Path("tasks/openness.json"),
    cogtype        = "custom",       # use persona files
    persona_files  = [Path("personas/my_personas.json")],
    memory         = "SingleTurn",
    parser         = "1",
    proj_dir       = Path("./results"),
    projectname    = "persona_study",
    enabletqdm     = True,
)

# One run per persona statement (4 simulated participants)
scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

The number of simulated participants equals the total number of statements across
all persona files. `nsim` is not used when `cogtype="custom"`.

---

## Multiple persona files (crossed design)

If you pass multiple files, PsychScanner takes the **Cartesian product** — every
combination of one statement from each file becomes a participant:

```python
card.persona_files = [
    Path("personas/age_groups.json"),    # 3 statements → age: young/middle/old
    Path("personas/traits.json"),        # 2 statements → trait: high/low openness
]
# → 3 × 2 = 6 simulated participants
```

Use this for factorial designs where you want every combination of demographic
and personality variables.

---

## Assistant mode (no persona)

`cogtype="assistant"` runs with a single generic system prompt (no persona file
needed). Set `nsim` to control how many replications:

```python
card = ExpCardInit(
    ...
    cogtype  = "assistant",
    nsim     = 20,           # 20 replications of the same generic assistant
)
```

---

## No-persona mode

`cogtype="no"` removes the system prompt entirely. The model responds with
minimal role framing. This is the fastest option for baseline measurements:

```python
card = ExpCardInit(
    ...
    cogtype = "no",
    nsim    = 100,
)
```

---

## Analysing persona differences

```python
import pandas as pd

df = pd.read_csv("results/persona_study/openness_items/openai_gpt-4o-mini_SingleTurn/persona_study.csv")

# Each sim_id corresponds to one persona
print(df.groupby("sim_id")["rating"].mean())

# Merge with persona labels
personas = [
    "Imaginative student",
    "Practical accountant",
    "Vivid artist",
    "Concrete engineer",
]
df["persona"] = df["sim_id"].map(dict(enumerate(personas)))
print(df.groupby("persona")["rating"].agg(["mean", "std"]))
```

---

## See also

- [Configuration Reference](../configuration.md) — `cogtype`, `persona_files`, `nsim`
- [Cognitive Tasks](../guides/cognitive_tasks.md) — the task card (task
  JSON schema) this page crosses with personas
- [PsychScanner Workflow](../guides/psychscanner_workflow.md) — the
  experiment card (`ExpCard`) that `cogtype`/`persona_files` are fields on
- [Survey Tasks](../guides/survey_tasks.md) — task JSON structure
- [Task Library](../guides/task_library.md) — fetch a saved task card by
  name instead of hardcoding a path, as this page's examples do
- [Session Recovery](../guides/session_recovery.md) — for large persona × item designs
