# Quick Start

## Minimal example

```python
from pathlib import Path
from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

# 1. Configure the experiment
card = ExpCardInit()
card.model         = "gpt-4o-mini"
card.family        = "openai"
card.projectname   = "my_experiment"
card.proj_dir      = Path("./results")
card.system_prompt = "You are a human participant in a psychology study."
card.task_prompt   = "Rate how vividly you can imagine the following scene (1–5): {scene}"
card.response_format = {"Vividness": "int (1–5)"}
card.nsim          = 10   # number of simulated participants

# 2. Run
scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()

# 3. Export to CSV
to_csv(scanner, path=card.proj_dir)
```

## Memory modes

Set `card.memory` to control whether the model sees prior trials:

| Value | Behaviour |
|---|---|
| `"SingleTurn"` | Each trial is an independent conversation — no memory of prior responses |
| `"Convo"` | Trials are chained in one conversation — model remembers previous turns |

## Feedback

Pass a `FeedbackBase` subclass as `feedback_fn` to inject per-trial corrective feedback:

```python
from psychscanner import FeedbackBase
import json

class MyFeedback(FeedbackBase):
    def on_response(self, trial, response):
        if int(response.get("Vividness", 3)) < 2:
            return json.dumps({"hint": "Try to engage more deeply with the imagery."})
        return None

card.feedback    = True
card.feedback_fn = MyFeedback
```

## Session recovery

For long runs, enable checkpointing so the experiment can resume after interruption:

```python
card.tunnel_status = "1"   # enable session tunnel
```

If the process is killed, re-run the same script — `ScannerModel` will skip already-completed participants and resume from where it left off.

## Exporting results

```python
from psychscanner import to_csv, concat_csv

# Write one CSV per participant
to_csv(scanner, path=card.proj_dir)

# Concatenate all CSVs into a single DataFrame
df = concat_csv(card.proj_dir)
```
