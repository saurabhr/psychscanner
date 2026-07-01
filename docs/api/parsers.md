# Parsers API Reference

Structured-output parsers are Pydantic `BaseModel` subclasses.  
They tell the LLM exactly what fields to return and enforce type constraints on each value.

---

## Top-level namespace — `psychscanner.parsers`

The recommended import path for all bundled parsers plus the registry helpers.

```python
from psychscanner.parsers import (
    list_parsers,
    get_parser,
    resolve_parser,
    PARSER_REGISTRY,
    DefaultLiteralVivid15,   # any parser class by name
    ...
)
```

### `list_parsers() → list[str]`

Return a sorted list of every registered parser class name.

```python
from psychscanner.parsers import list_parsers
print(list_parsers())
# ['AllResponseRMEI', 'AllResponseRMEIN', ..., 'WordNonWord']
```

### `get_parser(name: str) → type[BaseModel]`

Look up a parser class by name.  
Raises `KeyError` with a full list of available names on a typo — no guessing.

```python
from psychscanner.parsers import get_parser
cls = get_parser("DefaultLiteralVivid15")
print(cls.model_json_schema())
```

### `resolve_parser(value, task_parser_name=None) → type[BaseModel] | Callable | None`

Single entry point that accepts any parser specification and returns the concrete form.

| `value` | Result |
|---|---|
| `None` or `"0"` | `None` — no structured output |
| `"1"` | Registry lookup using `task_parser_name` (from task JSON `"parser"` field) |
| Any other string | `get_parser(value)` |
| `BaseModel` subclass | Returned as-is |
| Callable (not a class) | Returned as-is (callable dispatch) |

`ExpCard` calls `resolve_parser` internally — you rarely need to call it directly.

### `PARSER_REGISTRY`

`dict[str, type[BaseModel]]` — the backing store for all bundled parsers.

---

## Module layout

| Module | Contents |
|---|---|
| `psychscanner.parsers` | Top-level namespace + registry helpers |
| `psychscanner.datasets.prompts.parser_tasks` | Vividness + RM parsers (29 classes) |
| `psychscanner.datasets.prompts.parser_general` | Likert, generic, word-class, readiness (12 classes) |
| `psychscanner.datasets.prompts.parser` | Backward-compat shim (old name) |
| `psychscanner.datasets.prompts.parser_extra` | Backward-compat shim (old name) |

All paths for the same class return the **same object** — no duplication.

---

## Bundled parsers

### Vividness

| Class | Fields | Scale |
|---|---|---|
| `DefaultLiteralVivid15` | `Vividness` | 1–5 (1 = no image, 5 = perfectly vivid) |
| `DefaultLiteralVivid010` | `Vividness` | 0–10 |
| `DefaultLiteralVivid15Pol` | `Vividness` | 1–5 (Polish labels) |

### General / Likert

| Class | Fields | Notes |
|---|---|---|
| `DefaultLiteralAgree` | `rating` | 1–5 Likert agreement |
| `DefaultParser` | `response` | Generic list response |
| `DefaultRatingParser` | `response` | Single integer rating |
| `DefaultResponseRating` | `response`, `rating` | String + float |
| `SimpleResponseRating` | `response`, `rating` | Single word + float |
| `TwoResponses` | `Response_1`, `Response_2` | Two-part answers |

### Word classification

| Class | Fields | Options |
|---|---|---|
| `DefaultWordCaseNonWord` | `response` | `Lower-case Word` / `Upper-case Word` / `Non-Word` |
| `DefaultWordCaseNonWordConf16` | `response`, `confidence` | Same + confidence 1–6 |
| `WordNonWord` | `response` | `Upper-Case Word` / `Lower-Case Word` / `Non-Word` |

### Task readiness

| Class | Fields | Notes |
|---|---|---|
| `Ready` | `response` | Literal `"Ready"` |
| `TaskReadyConfidence` | `response`, `rating` | `READY`/`NO STOP` + confidence 1–6 |

### Reality Monitoring (RM) — encoding phase

| Class | Fields | Notes |
|---|---|---|
| `DefaultRMEncodingPhase` | `second_word`, `Relatedness` | Word + 0–100 % relatedness |
| `Response_part_1_rm` | `Word_2`, `Rating` | Encoding word + 0–100 |
| `ResponseRmScSt` | `Word_2`, `Rating`, `Judgment`, `Confidence` | All-in-one |
| `ResponseRmStIE` | `Word_2`, `Rating`, `Judgment`, `Confidence` | Judgment order: internal/external |
| `ResponseRmStEI` | `Word_2`, `Rating`, `Judgment`, `Confidence` | Judgment order: external/internal |
| `ResponseRmStIEN` | `Word_2`, `Rating`, `Judgment`, `Confidence` | + `"new"` option; rich JSON descriptions |
| `ResponseRmStEIN` | `Word_2`, `Rating`, `Judgment`, `Confidence` | Reversed order + `"new"` option |

### Reality Monitoring — test / retrieval phase

| Class | Fields | Notes |
|---|---|---|
| `Response_part_2_rm` | `Judgment`, `Confidence` | `internal`/`external` + 1–6 |
| `Response_part_2_rmrevo` | `Judgment`, `Confidence` | Reversed literal order |
| `DefaultRmChoiceConf16` | `response`, `confidence` | Longer label strings + 1–6 |
| `Source` | `source` | `Imagined` / `Perceived` / `New word` |

### Reality Monitoring — component / union parsers

These are used when a single trial step returns only one component.

| Class | Fields |
|---|---|
| `Word2` | `Word_2` |
| `RelatednessRating` | `Relatedness_Rating` |
| `JudgmentIE` | `Judgment` (`internal`/`external`) |
| `JudgmentEI` | `Judgment` (`external`/`internal`) |
| `JudgmentIEN` | `Judgment` (`internal`/`external`/`new`) |
| `JudgmentEIN` | `Judgment` (`external`/`internal`/`new`) |
| `Confidence16` | `Confidence` (1–6) |
| `AllResponseRMIE` | Union of the four above |
| `AllResponseRMEI` | Union, reversed judgment order |
| `AllResponseRMIEN` | Union + `new` option |
| `AllResponseRMEIN` | Union + `new`, reversed |

### Multi-phase task chain

| Class | Notes |
|---|---|
| `TaskResponse` | Union over `TaskReadyConfidence`, `Task_1_ResponseRate`, `Task_2_ResponseRate`, `Task_3_ResponseRate` |
| `Task_1_ResponseRate` | Encoding: `response` (str) + `rating` (0–100) |
| `Task_2_ResponseRate` | Word-type: `response` (Upper/Lower/Non-word) + `rating` (confidence 1–6) |
| `Task_3_ResponseRate` | Recognition: `response` (Imagined/Perceived/New) + `rating` (confidence 1–6) |

---

## Using parsers in `ExpCard`

```python
from psychscanner import ExpCardInit, ExpCard, ScannerModel
from psychscanner.parsers import DefaultLiteralVivid15, Response_part_1_rm, Response_part_2_rm

# Mode: direct class (recommended for code)
card = ExpCardInit(
    parser=DefaultLiteralVivid15,
    ...
)

# Mode: string name
card = ExpCardInit(parser="DefaultLiteralVivid15", ...)

# Mode: resolve from task JSON ("parser" field)
card = ExpCardInit(parser="1", ...)

# Mode: callable dispatch (per-trial routing)
card = ExpCardInit(
    parser=lambda trcode: Response_part_2_rm if "test" in trcode else Response_part_1_rm,
    ...
)

# Diagnostic flags
card = ExpCardInit(
    parser=DefaultLiteralVivid15,
    parser_raw=True,                          # also keep raw AIMessage
    parser_config={"method": "json_schema"},  # forwarded to with_structured_output()
    ...
)
```

---

## Writing a custom parser

Any `pydantic.BaseModel` subclass works:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentRating(BaseModel):
    """Rate the sentiment of the given statement."""

    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Overall sentiment of the statement."
    )
    confidence: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="1 = not confident, 5 = very confident."
    )

card = ExpCardInit(parser=SentimentRating, ...)
```

See [Custom Parsers Guide](../guides/custom_parsers.md) for design tips and examples.

---

## See also

- [Custom Parsers Guide](../guides/custom_parsers.md)
- Example notebooks — [Parsers](../tutorials/03_parsers.ipynb), [Parser Modules](../tutorials/04_parser_modules.ipynb), [Parser System Guide](../tutorials/08_ps_parser_guide.ipynb)
