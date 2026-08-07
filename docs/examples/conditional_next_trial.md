# Conditional Next-Trial Example

PsychScanner can insert a new trial into the run based on the model's response to the
trial that just executed, before moving on to the task card's next trial. This
supports adaptive/staircase designs and retry-until-valid paradigms — the trial
sequence isn't fixed at authoring time, it can branch at run time.

This mirrors the [Feedback Loop](feedback_loop.md) mechanism (same lifecycle, same
`trial`/`response` signature), but instead of returning text to inject, the handler
returns a whole new trial to run.

---

## How it works

1. Set `next_trial=True` and provide a `next_trial_fn` class on the card.
2. After each trial (original or inserted), the scanner calls
   `next_trial_fn.next_trial(trial, response)`.
3. Returning a trial dict (same shape as a task JSON item: `trcode`/`stimulus`,
   optional `fb`/`tools`/`parser`) runs it immediately, before the task card's next
   trial. Returning `None` moves on to that next trial.
4. **Recursive-break guard:** if the handler proposes the exact same `stimulus`
   more than `max_repeat` times in a row (default 3), the runner stops asking and
   resumes the task card's own trial sequence — this bounds a handler that would
   otherwise loop forever on an unmet condition.

---

## NextTrialBase interface

```python
from psychscanner import NextTrialBase

class MyNextTrial(NextTrialBase):
    max_repeat = 3  # override to change the repeat-guard threshold

    def next_trial(self, trial: dict, response: dict) -> dict | None:
        """
        trial    — the trial dict that was just executed
        response — the parsed response dict (keys match your parser fields)

        Return a new trial dict to run next, or None to continue with the
        task card's own next trial.
        """
        ...
```

---

## Retry until a valid rating is given

```python
from psychscanner import NextTrialBase, ExpCardInit, ExpCard, ScannerModel

class RetryUntilRated(NextTrialBase):
    def next_trial(self, trial: dict, response: dict) -> dict | None:
        if response.get("rating") is None:
            return {
                "trcode": trial["trcode"] + "_retry",
                "stimulus": "Please answer with a single number from 1 to 5.",
            }
        return None

card = ExpCardInit(
    model         = "gpt-4o-mini",
    family        = "openai",
    task_file     = "tasks/my_survey.json",
    memory        = "Convo",       # the retry needs to see the original question
    chain_type    = "task",
    parser        = "1",
    cogtype       = "no",
    nsim          = 20,
    next_trial    = True,
    next_trial_fn = RetryUntilRated,
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
```

`RetryUntilRated.max_repeat` (default 3) caps how many times the exact same retry
prompt can fire in a row — after that the runner gives up and moves to the task
card's next trial rather than retrying forever.

---

## Adaptive staircase (raise/lower difficulty)

```python
class Staircase(NextTrialBase):
    def __init__(self):
        super().__init__()
        self.level = 1

    def next_trial(self, trial, response):
        correct = str(response.get("answer")) == str(trial.get("corrAns"))
        self.level = min(self.level + 1, 5) if correct else max(self.level - 1, 1)
        if self.level == trial.get("_level"):
            return None  # converged, let the task card continue
        return {
            "trcode": f"stair_{self.level}",
            "stimulus": f"Difficulty level {self.level} item",
            "_level": self.level,
        }
```

Like `FeedbackBase`, the handler is instantiated once per participant simulation, so
`__init__` state (here, `self.level`) is safe to keep across trials.

---

## Required card settings

| Parameter | Required value |
|-----------|---------------|
| `next_trial` | `True` |
| `next_trial_fn` | Your `NextTrialBase` subclass (class, not instance) |

Inserted trials inherit `context_present`/`context_item`/`tasktype` from the trial
they follow unless you override them explicitly in the returned dict.

---

## See also

- [Feedback Loop](feedback_loop.md) — the equivalent mechanism for injecting text
  feedback rather than branching the trial sequence
- [Memory Types](../guides/memory_types.md) — `chain_type` explanation
- [Cognitive Tasks guide](../guides/cognitive_tasks.md) — task JSON trial schema
