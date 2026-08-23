# Custom Parsers

A **parser** in PsychScanner is a Pydantic `BaseModel` subclass.  
It tells the LLM what fields to return and enforces type constraints on every value.

This guide covers:
1. [Why use a parser?](#1-why-use-a-parser)
2. [Anatomy of a parser](#2-anatomy-of-a-parser)
3. [Writing your first parser](#3-writing-your-first-parser)
4. [Field types and constraints](#4-field-types-and-constraints)
5. [Registering a parser (optional)](#5-registering-a-parser-optional)
6. [Per-trial routing with multiple parsers](#6-per-trial-routing-with-multiple-parsers)
7. [Inspecting and debugging parsers](#7-inspecting-and-debugging-parsers)
8. [Common patterns](#8-common-patterns)

---

## 1. Why use a parser?

Without a parser, the model returns a raw `AIMessage` whose `.content` is an
unstructured string — anything from "4" to a paragraph of reasoning.

With a parser, `ExpCard` calls LangChain's `with_structured_output()`.  
The model is forced to emit valid JSON that matches the schema, and the result
arrives as a Python dict ready for analysis:

```python
# Without parser
{"pred_resp": AIMessage(content="I'd say about 4 on that scale.")}

# With DefaultLiteralAgree parser
{"pred_resp": {"rating": 4}}
```

---

## 2. Anatomy of a parser

```python
from pydantic import BaseModel, Field
from typing import Literal

class MyParser(BaseModel):
    """<-- class docstring becomes the top-level schema description shown to the model -->"""

    field_name: SomeType = Field(
        ...,                          # required (no default)
        description="<-- shown to the model as the field's instruction -->",
    )
```

Key decisions:

| Decision | Effect |
|---|---|
| **Class docstring** | Sent as the overall task instruction for this parser |
| **Field name** | The key in the returned dict |
| **Field type** | Enforced by Pydantic — model output is coerced or rejected |
| **`description`** | Sent verbatim in the JSON schema — guide the model clearly |

---

## 3. Writing your first parser

A simple agreement rating on a 1–7 scale:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Agreement7(BaseModel):
    """Rate your agreement with the statement on a scale from 1 to 7."""

    rating: Literal[1, 2, 3, 4, 5, 6, 7] = Field(
        ...,
        description=(
            "Agreement rating: "
            "1 = strongly disagree, 7 = strongly agree. "
            "Always give a single integer."
        ),
    )
```

Wire it into an experiment card:

```python
from psychscanner import ExpCardInit, ExpCard, ScannerModel

card = ExpCardInit(
    model="gemma3:12b",
    family="ollama",
    parameters={"temperature": 0},
    parser=Agreement7,          # pass the class directly
    task_file="my_survey.json",
    cogtype="no",
    nsim=1,
)
results = ScannerModel(expcard=ExpCard(card)).run()
# results[0][0]["pred_resp"]  →  {"rating": 5}
```

---

## 4. Field types and constraints

### Categorical fields — `Literal`

Use `Literal` when the model must choose from a fixed set of options.  
List the options in order of how you want the model to encounter them — LLMs sometimes
favour the first option in ambiguous cases.

```python
from typing import Literal

source: Literal["internal", "external", "new"] = Field(...)
sentiment: Literal["positive", "neutral", "negative"] = Field(...)
```

### Bounded continuous fields — `float` with `ge`/`le`

```python
rating: float = Field(..., ge=0.0, le=100.0,
    description="Relatedness from 0 (not related) to 100 (highly related).")
```

Or with `confloat` (deprecated in Pydantic v2 — prefer `Field(ge=, le=)` instead):

```python
from pydantic import confloat
rating: confloat(ge=0.0, le=100.0) = Field(...)
```

### Integer Likert scales — `Literal` of ints

```python
confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
    ..., description="1 = not at all confident, 6 = highly confident."
)
```

Using `Literal` instead of `int` prevents out-of-range integers without writing
a custom validator.

### Free-text fields — `str`

```python
second_word: str = Field(
    ..., description="The word you imagined to complete the word pair."
)
```

### Multi-field parsers

Combine fields freely:

```python
class EncodingResponse(BaseModel):
    """Respond to the encoding trial."""

    word: str = Field(..., description="The second word (given or imagined).")
    relatedness: float = Field(..., ge=0.0, le=100.0,
        description="Relatedness of the two words (0–100 %).")
```

---

## 5. Registering a parser (optional)

Bundled parsers are auto-registered in `PARSER_REGISTRY` at import time.  
**Custom parsers do not need to be registered** — you can always pass the class directly.

If you want to reference your parser by name string (e.g. in a task JSON file or via `get_parser()`),
add it to the registry:

```python
from psychscanner.parsers import PARSER_REGISTRY
from my_parsers import Agreement7

PARSER_REGISTRY["Agreement7"] = Agreement7

# Now these all work:
from psychscanner.parsers import get_parser
cls = get_parser("Agreement7")

# And in task JSON:
# { "parser": "Agreement7" }
# with card.parser = "1"
```

Do this registration **before** creating `ExpCard`.

---

## 6. Per-trial routing with multiple parsers

When a task has different trial types that need different parsers, you have two options.

### Option A — callable dispatch (recommended for code)

```python
from psychscanner.parsers import PairedAssociateRecall

def phase_dispatch(trcode: str):
    if "test" in trcode:
        return PairedAssociateRecall   # recalled_word + confidence
    return None                        # no structured output needed during study

card = ExpCardInit(parser=phase_dispatch, ...)
```

The callable is called once per trial with the trial's `trcode` string and must
return a `BaseModel` subclass (or `None` for no structured output on that trial).

### Option B — `"parser"` key in the task JSON (Form A)

Add a `"parser"` field to each trial dict in your task JSON file.
The value must be a registered parser name string.
Set `card.parser = "0"` as the card-level fallback:

```json
{
  "chain_type": "item",
  "parser": "0",
  "items": {
    "study_1": [{"trcode": "study_1", "parser": null, "stimulus": {...}}],
    "test_1":  [{"trcode": "test_1", "parser": "PairedAssociateRecall", "stimulus": {...}}]
  }
}
```

**Priority:** per-trial JSON `"parser"` (Option B) takes precedence over the card-level
callable (Option A). You can mix them: use the callable as a fallback for trials that
don't specify a parser in the JSON.

---

## 7. Inspecting and debugging parsers

### View the JSON schema

The schema is what gets sent to the model as the structured-output spec:

```python
import json
print(json.dumps(Agreement7.model_json_schema(), indent=2))
```

### Keep the raw AIMessage alongside parsing

Set `parser_raw=True` to get both the parsed dict and the original model output:

```python
card = ExpCardInit(parser=Agreement7, parser_raw=True, ...)
# trial["pred_resp"] now contains {"rating": 5, "_raw": AIMessage(...)}
```

### Change the structured-output method

`parser_config` is forwarded to `model.with_structured_output(**parser_config)`:

```python
card = ExpCardInit(
    parser=Agreement7,
    parser_config={"method": "function_calling"},  # try if json_schema fails
    ...
)
```

Default is `{"method": "json_schema"}`. Use `"function_calling"` or `"json_mode"`
if your provider doesn't support JSON schema mode.

---

## 8. Common patterns

### Confidence + rating pair

```python
class RatingConf(BaseModel):
    """Rate and express confidence."""
    rating: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="Rating: 1 (low) to 5 (high).")
    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ..., description="Confidence: 1 (guessing) to 6 (certain).")
```

### Free recall with confidence

```python
class RecallResponse(BaseModel):
    """Recall the studied item and rate confidence."""
    recalled_word: str = Field(
        ..., description="The word you believe was paired with the probe during study.")
    confidence: Literal[1, 2, 3, 4, 5, 6] = Field(
        ..., description="Confidence in the recall: 1–6.")
```

### Binary yes/no with explanation

```python
class YesNoExplain(BaseModel):
    """Answer yes or no and briefly explain."""
    answer: Literal["yes", "no"] = Field(...)
    reason: str = Field(..., description="One sentence explaining the answer.")
```

### Multi-step trial (per-step parsers)

Split complex trials into separate steps using a callable dispatch:

```python
def step_dispatch(trcode: str):
    if "step2" in trcode:
        return StepTwoParser
    return StepOneParser

card = ExpCardInit(parser=step_dispatch, ...)
```

---

## See also

- [Parsers API Reference](../api/parsers.md) — full class listing
- Example notebooks — [Parsers](../tutorials/03_parsers.ipynb), [Parser System Guide](../tutorials/08_ps_parser_guide.ipynb)
- [Paired-Associate Learning task guide](cognitive_tasks.md#paired-associate-learning-pal-task)
