# Memory Types

PsychScanner supports two memory modes that control whether a simulated participant
"remembers" previous trials within an experiment run.

> The snippets below (`card.memory = ...`, `card.memory_k = ...`, etc.) illustrate
> `ExpCardInit` fields. `ExpCard` reads these values once at construction time
> (`ExpCard(card)`), so set them on the `ExpCardInit` instance *before* building
> the `ExpCard`/`ScannerModel` — mutating `card` afterwards has no effect on an
> already-constructed `ExpCard`.

---

## SingleTurn (stateless)

```python
card.memory = "SingleTurn"
```

Each trial is sent as a **fresh, independent conversation**. The model has no knowledge
of what it said in previous trials.

**When to use:**
- Standard questionnaires where trial order should not matter
- Designs that require statistical independence between responses
- Any task where you want to eliminate within-session carry-over effects

**What the model sees per trial:**
```
[system message]
[trial stimulus]
```

---

## Convo (conversational)

```python
card.memory = "Convo"
```

All trials for one participant run within **a single continuous conversation**.
The model sees every prior exchange and builds up context across trials.

**When to use:**
- Multi-phase tasks (encoding → retrieval)
- Feedback experiments where the model is corrected after each trial
- Tasks that simulate a real cognitive session

**What the model sees by trial 3:**
```
[system message]
[trial 1 stimulus] → [trial 1 response]
[trial 2 stimulus] → [trial 2 response]
[trial 3 stimulus]
```

---

## Context window management

For long experiments in Convo mode the conversation grows large.
Two parameters control what stays in context.

### `memory_k`

```python
card.memory_k = -1   # default: keep all messages
card.memory_k = 10   # keep only the last 10 messages
```

When set to `N`, the conversation is truncated to the **N most recent messages**
before each new trial. Older turns are dropped silently.
Use this to cap token costs on large-scale Convo experiments.

### `summary_k`

```python
card.summary_k = 0    # default: no summarisation
card.summary_k = 20   # summarise when context overflows
```

`summary_k` is a threshold, not a batch size: once the overflow (all messages
beyond `memory_k`) reaches `summary_k` messages, the **entire overflow** is
condensed into a single summary message by the model before continuing.
This preserves thematic context without the cost of the full history.

---

## chain_type

`chain_type` controls how individual task trials are structured,
and interacts with the memory mode.

| Value | What happens each turn |
|-------|----------------------|
| `"item"` | One stimulus → one model call. The most common mode. |
| `"trial"` | Multiple stimuli shown in sequence within a single turn. |
| `"task"` | Trials carry over context from previous trials (task-level chaining, commonly used with feedback). |

```python
card.chain_type = "item"   # default for most tasks
card.chain_type = "task"   # commonly used for feedback experiments
```

Feedback injection itself is controlled by `feedback`/`feedback_fn` and works
regardless of `chain_type` — `chain_type` only changes which LangGraph
`thread_id` trials share, not whether feedback is applied.

`chain_type` can be set on the card (overrides the task JSON) or defined inside the task JSON.

---

## Choosing the right combination

| Experiment type | `memory` | `chain_type` |
|---|---|---|
| Standard questionnaire | `SingleTurn` | `"item"` |
| Imagery vividness (VVIQ) | `SingleTurn` | `"item"` |
| Reality monitoring — encoding only | `SingleTurn` | `"item"` |
| Reality monitoring — encode + test | `Convo` | `"item"` |
| RM with trial feedback | `Convo` | `"task"` |
| Multi-phase cognitive task | `Convo` | `"trial"` |

---

## Example: SingleTurn vividness rating

```python
from psychscanner import ExpCardInit, ExpCard, ScannerModel
from psychscanner.parsers import DefaultLiteralVivid15

card = ExpCardInit(
    model   = "gpt-4o-mini",
    family  = "openai",
    memory  = "SingleTurn",
    parser  = DefaultLiteralVivid15,
    cogtype = "no",
    nsim    = 20,
)
results = ScannerModel(expcard=ExpCard(card)).run()
```

## Example: Convo with context trimming

```python
card = ExpCardInit(
    model    = "llama3.1:8b",
    family   = "ollama",
    memory   = "Convo",
    memory_k = 20,       # keep last 20 messages only
    cogtype  = "no",
    nsim     = 10,
)
```

---

## See also

- [Survey Tasks](survey_tasks.md) — task JSON structure
- [Cognitive Tasks](cognitive_tasks.md) — multi-phase paradigms
- [Custom Parsers](custom_parsers.md) — structuring responses
- [Configuration Reference](../configuration.md) — all memory parameters
