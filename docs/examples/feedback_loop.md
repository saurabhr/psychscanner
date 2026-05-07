# Feedback Loop Example

PsychScanner can inject trial-level feedback back into the conversation after each
response. This simulates adaptive learning paradigms where participants are corrected
or informed of their performance.

This matches notebooks `06_feedback_api.ipynb` and `07_rm_feedback_task.ipynb`.

---

## How feedback works

1. Set `feedback=True` and provide a `feedback_fn` class on the card.
2. After each trial, the scanner calls `feedback_fn.on_response(trial, response)`.
3. The returned string is injected as an assistant message into the conversation.
4. Requires `memory="Convo"` (feedback is part of the conversation history).

---

## FeedbackBase interface

```python
from psychscanner import FeedbackBase
import json

class MyFeedback(FeedbackBase):
    def on_response(self, trial: dict, response: dict) -> str | None:
        """
        trial    — the raw trial dict from the task JSON
        response — the parsed response dict (keys match your parser fields)

        Return a JSON string to inject as feedback, or None to skip.
        """
        ...
```

---

## Simple correctness feedback

For a questionnaire where each item has a `corrAns` field:

```python
from psychscanner import FeedbackBase, ExpCardInit, ExpCard, ScannerModel, to_csv
from pathlib import Path
import json

class CorrectnessFeedback(FeedbackBase):
    def on_response(self, trial: dict, response: dict) -> str | None:
        correct = trial.get("corrAns")
        given   = response.get("rating")
        if correct is None:
            return None
        if str(given) == str(correct):
            msg = f"Correct! The answer was {correct}."
        else:
            msg = f"Incorrect. The correct answer was {correct}, you gave {given}."
        return json.dumps({"feedback": msg})

card = ExpCardInit(
    model        = "gpt-4o-mini",
    family       = "openai",
    task_file    = Path("tasks/my_task.json"),
    memory       = "Convo",
    chain_type   = "task",      # required for feedback
    parser       = "1",
    cogtype      = "no",
    nsim         = 20,
    feedback     = True,
    feedback_fn  = CorrectnessFeedback,
    proj_dir     = Path("./results"),
    projectname  = "feedback_study",
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## Reality monitoring with feedback

Track used words across trials and give detailed encoding feedback:

```python
from psychscanner import FeedbackBase
import json

_used_words: list[str] = []   # shared across trials within a participant

class RMEncodingFeedback(FeedbackBase):
    def on_response(self, trial: dict, response: dict) -> str | None:
        stim  = trial["stimulus"]
        if not isinstance(stim, dict):
            return None   # skip test phase trials

        word1 = stim["Word_Pair"]["word_1"].strip()
        word2 = stim["Word_Pair"]["word_2"].strip()
        given = str(response.get("Word_2", "")).strip()

        _used_words.append(word1)

        if "____" in word2:           # imagined trial
            if given.lower() in [w.lower() for w in _used_words]:
                fb = f"INCORRECT — '{given}' was already used. Try a novel word."
            else:
                _used_words.append(given)
                fb = f"CORRECT — '{given}' is a novel word."
        else:                         # perceived trial
            _used_words.append(word2)
            if word2.lower() == given.lower():
                fb = f"CORRECT — '{word2}' was the provided word."
            else:
                fb = f"INCORRECT — expected '{word2}', you gave '{given}'."

        return json.dumps({"feedback": fb})
```

---

## Stateful feedback with `__init__`

The feedback class is instantiated once per participant run. Use `__init__` for
per-participant state:

```python
class StatefulFeedback(FeedbackBase):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total   = 0

    def on_response(self, trial: dict, response: dict) -> str | None:
        self.total += 1
        correct = trial.get("corrAns")
        given   = response.get("rating")
        if str(given) == str(correct):
            self.correct += 1
        acc = self.correct / self.total
        return json.dumps({
            "feedback": f"Running accuracy: {acc:.0%} ({self.correct}/{self.total})"
        })
```

---

## Required card settings for feedback

| Parameter | Required value |
|-----------|---------------|
| `memory` | `"Convo"` |
| `chain_type` | `"task"` |
| `feedback` | `True` |
| `feedback_fn` | Your `FeedbackBase` subclass (class, not instance) |

---

## See also

- [Cognitive Tasks guide](../guides/cognitive_tasks.md) — RM feedback walkthrough
- [FeedbackBase API](../api/expcard.md#feedback) — method reference
- [Memory Types](../guides/memory_types.md) — chain_type explanation
