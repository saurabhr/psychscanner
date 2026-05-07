# Basic Survey Example

This example runs a simple Likert-scale questionnaire using `gpt-4o-mini`.
The built-in VVIQ-16 imagery questionnaire is used as the default task.

---

## Minimal working example

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    projectname = "vviq_pilot",
    proj_dir    = Path("./results"),
    cogtype     = "no",
    nsim        = 20,
    memory      = "SingleTurn",
    parser      = DefaultLiteralVivid15,
    enabletqdm  = True,
)

scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

Output files are written to:
```
results/vviq_pilot/<taskname>/openai_gpt-4o-mini_SingleTurn/
```

---

## Custom Likert survey

Define your own items in a JSON file:

```json
{
  "tasktype"        : "survey",
  "taskname"        : "openness_items",
  "instructions"    : {
    "definition": [
      "You will read a series of statements.",
      "For each statement, rate your agreement on a scale from 1 to 5.",
      "1 = strongly disagree, 5 = strongly agree."
    ]
  },
  "contexts"        : ["Openness to Experience"],
  "contexts_id"     : ["items"],
  "context_present" : false,
  "chain_type"      : "item",
  "parser"          : "DefaultLiteralAgree",
  "items": {
    "items": [
      { "trcode": "open_1", "stimulus": "I have a vivid imagination." },
      { "trcode": "open_2", "stimulus": "I enjoy artistic experiences." },
      { "trcode": "open_3", "stimulus": "I prefer variety to routine." },
      { "trcode": "open_4", "stimulus": "I am quick to understand new ideas." }
    ]
  }
}
```

Save this as `tasks/openness.json`, then:

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv

card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    task_file   = Path("tasks/openness.json"),
    projectname = "openness_study",
    proj_dir    = Path("./results"),
    cogtype     = "no",
    nsim        = 50,
    memory      = "SingleTurn",
    parser      = "1",           # resolve from task JSON's "parser" field
    enabletqdm  = True,
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## Choosing a parser

| Scale | Parser class |
|-------|-------------|
| 1–5 agreement (Likert) | `DefaultLiteralAgree` |
| 1–5 vividness | `DefaultLiteralVivid15` |
| 0–10 vividness | `DefaultLiteralVivid010` |
| Free response + rating | `DefaultResponseRating` |

Pass the class directly or use `parser="1"` (resolves from the task JSON `"parser"` field).

---

## Reading the results

```python
import pandas as pd

df = pd.read_csv("results/openness_study/openness_items/openai_gpt-4o-mini_SingleTurn/openness_study.csv")
print(df.columns.tolist())
# ['sim_id', 'trcode', 'stimulus', 'pred_resp', 'rating', ...]

print(df.groupby("trcode")["rating"].mean())
```

---

## See also

- [Survey Tasks guide](../guides/survey_tasks.md) — full JSON schema reference
- [Parsers API](../api/parsers.md) — all bundled parser classes
- [Configuration Reference](../configuration.md) — full parameter list
