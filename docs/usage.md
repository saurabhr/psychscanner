# Quick Start

## Installation

```bash
conda install -c conda-forge uv                    # if you use conda
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
uv venv psyscan --python 3.11
source psyscan/bin/activate

git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv pip install -e .
```

See [Installation](installation.md) for API key setup and full details.

## Minimal example

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

# 1. Configure the experiment
card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    projectname = "my_experiment",
    proj_dir    = Path("./results"),
    cogtype     = "no",         # no persona files — set participant count directly
    nsim        = 10,           # number of simulated participants
    memory      = "SingleTurn",
    parser      = DefaultLiteralVivid15,
)

# 2. Run (uses built-in VVIQ-16 imagery questionnaire by default)
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
