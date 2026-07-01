# Configuration Reference

Every experiment is configured through an `ExpCardInit` instance.
This page is a complete reference for all parameters.

```python
from psychscanner import ExpCardInit, ExpCard, ScannerModel

card = ExpCardInit(
    model      = "gpt-4o-mini",
    family     = "openai",
    nsim       = 20,
    cogtype    = "no",
    ...
)
exp = ExpCard(card)
```

---

## Model & Provider

### `model`
**Type:** `str` | **Default:** `"mock-chat-model"`

The model name passed to the LangChain provider. Examples:

| Provider | Example values |
|----------|---------------|
| OpenAI | `"gpt-4o"`, `"gpt-4o-mini"`, `"o1"` |
| Anthropic | `"claude-3-5-sonnet-20241022"`, `"claude-3-haiku-20240307"` |
| Groq | `"llama-3.1-70b-versatile"`, `"mixtral-8x7b-32768"` |
| Ollama | `"llama3.1:8b"`, `"gemma3:12b"`, `"smollm2:360m-instruct-fp16"` |
| HuggingFace | `"mistralai/Mistral-7B-Instruct-v0.3"` |
| Mock | `"mock-chat-model"` (no API key; for testing) |

### `family`
**Type:** `str` | **Default:** `"mock-llm"`

Provider identifier. Must match a supported family:

| Value | API key env var |
|-------|----------------|
| `"openai"` | `OPENAI_API_KEY` |
| `"anthropic"` | `ANTHROPIC_API_KEY` |
| `"groq"` | `GROQ_API_KEY` |
| `"mistral"` | `MISTRAL_API_KEY` |
| `"google"` / `"gemini"` / `"google-genai"` | `GOOGLE_API_KEY` |
| `"together"` | `TOGETHER_API_KEY` |
| `"fireworks"` | `FIREWORKS_API_KEY` |
| `"azure"` | `AZURE_OPENAI_API_KEY` |
| `"huggingface"` | `HUGGINGFACEHUB_API_TOKEN` |
| `"ollama"` | `OLLAMA_API_KEY` (remote only; local needs none) |
| `"cohere"` | `COHERE_API_KEY` |
| `"openrouter"` | `OPENROUTER_API_KEY` (routed through the OpenAI-compatible endpoint) |
| `"mock-llm"` | None |

API keys are read from environment variables (`os.getenv`) — psychscanner does
**not** load a `.env` file automatically. If you keep keys in a `.env` file, call
`load_dotenv()` (from `python-dotenv`) yourself before constructing the card.
You can also pass `{"api_key": "..."}` directly in `parameters`.

### `parameters`
**Type:** `dict | None` | **Default:** `None`

Extra keyword arguments forwarded directly to the model constructor.

```python
card.parameters = {
    "temperature": 0,
    "max_tokens": 256,
}

# Remote Ollama server:
card.parameters = {
    "base_url": "http://my-server:11434",
    "api_key": "my-key",
}
```

---

## Task & Persona

### `task_file`
**Type:** `dict | Path | None` | **Default:** built-in VVIQ-16

The experimental task. Accepts:

- A **file path** to a `.json` task file
- An **inline dict** following the [task JSON schema](guides/cognitive_tasks.md#task-json-schema)
- `None` — uses the built-in VVIQ-16 imagery questionnaire

### `cogtype`
**Type:** `"assistant"` / `"custom"` / `"no"` | **Default:** `"custom"`

Controls how participants (system messages) are generated:

| Value | Behaviour |
|-------|-----------|
| `"custom"` | Load personas from `persona_files`; one simulation per persona |
| `"assistant"` | Use a single generic assistant system prompt |
| `"no"` | No persona; use `nsim` to set the number of simulations |

### `nsim`
**Type:** `int | None` | **Default:** `None`

Number of simulated agents. **Only used when `cogtype="no"`.**

### `persona_files`
**Type:** `list[Path] | None` | **Default:** `None`

Paths to persona JSON files. Each file should have a `persona_statements` key
containing a list of persona descriptions. When `None`, the default persona set is loaded.

### `task_context`
**Type:** `True | False | None` | **Default:** `None`

When `True`, each survey item is formatted as:
```
context: <context description>
situation: <item text>
```
Only meaningful for task types that define a `"contexts"` field.

---

## Memory

### `memory`
**Type:** `"SingleTurn"` / `"Convo"` | **Default:** `"SingleTurn"`

- **`"SingleTurn"`** — every trial is an independent conversation. The model has no memory of prior trials. Best for within-subjects designs where you don't want order effects.
- **`"Convo"`** — all trials for one participant run in a single conversation. The model sees its previous answers. Required for feedback experiments and multi-phase tasks.

### `memory_k`
**Type:** `int` | **Default:** `-1`

Maximum number of messages kept in the conversation context window.

- `-1` — keep all messages (unlimited)
- `N` — keep only the most recent `N` messages, dropping older ones

### `summary_k`
**Type:** `int` | **Default:** `0`

Summarization trigger for `"Convo"` memory.

- `0` — disabled
- `N` — a threshold, not a batch size: once the overflow (messages beyond `memory_k`) reaches `N` messages, the **entire overflow** is summarized into a single summary message

### `chain_type`
**Type:** `"item"` / `"trial"` / `"task"` / `None` | **Default:** `None`

Overrides the `chain_type` field in the task JSON.

| Value | Meaning |
|-------|---------|
| `"item"` | Each trial is one stimulus (one model call per trial) |
| `"trial"` | One trial contains multiple stimuli shown in sequence within the same turn |
| `"task"` | Trials carry over context from previous trials (task-level chaining) |

---

## Parsing

### `parser`
**Type:** `str | type[BaseModel] | Callable | None` | **Default:** `None`

Controls structured output. See [ExpCard API](api/expcard.md) for parser modes.

### `parser_raw`
**Type:** `bool` | **Default:** `False`

When `True`, the raw `AIMessage` is stored alongside the parsed response:
```python
trial["pred_resp"]  # → {"rating": 4, "_raw": AIMessage(...)}
```
Useful for debugging when the model's literal output matters.

### `parser_config`
**Type:** `dict | None` | **Default:** `None`

Forwarded to `model.with_structured_output(**parser_config)`.
Default is `{"method": "json_schema"}`.

Other options depending on provider support:

```python
card.parser_config = {"method": "function_calling"}
card.parser_config = {"method": "json_mode"}
```

---

## Feedback

### `feedback`
**Type:** `bool` | **Default:** `False`

Enable trial-level feedback injection. Requires `feedback_fn` to be set (raises
`ValueError` at `ExpCard` construction otherwise) — accepts `True`/`False` or
the strings `"0"`/`"1"` for backward compatibility. `memory="Convo"` is not
enforced by the code, but feedback is only meaningful when trials share
context, so it's typically paired with `Convo` memory and `chain_type="task"`.

### `feedback_fn`
**Type:** `type[FeedbackBase] | None` | **Default:** `None`

A **class** (not an instance) that subclasses `FeedbackBase`.
PsychScanner instantiates it once per participant and calls `on_response()` after each trial.

```python
from psychscanner import FeedbackBase
import json

class CorrectnessFeedback(FeedbackBase):
    def on_response(self, trial, response):
        correct = trial.get("corrAns")
        given   = response.get("rating")
        if correct and given != correct:
            return json.dumps({"hint": f"Expected {correct}, you gave {given}."})
        return None

card.feedback    = True
card.feedback_fn = CorrectnessFeedback
```

---

## Output & Checkpointing

### `proj_dir`
**Type:** `Path` | **Default:** `~/psychscanner`

Root directory. Data is written to:
```
proj_dir/projectname/taskname/family_model_memory/
```

### `projectname`
**Type:** `str` | **Default:** `"DEFAULTPROJ"`

Sub-directory label that groups related runs together.

### `tunnel_status`
**Type:** `"0"` / `"1"` | **Default:** `"0"`

- `"0"` — no checkpointing; if interrupted, the run must start over
- `"1"` — write a checkpoint after each participant; re-running resumes where it left off

### `tunnel_k`
**Type:** `int` | **Default:** `-1`

Accepted and stored on the card, but currently **not read anywhere** in the run
loop — checkpointing always happens after every system prompt regardless of
this value. Treat it as reserved for future use.

### `enabletqdm`
**Type:** `bool` | **Default:** `False`

Show a tqdm progress bar. Can also be set at run time via `scanner.run(progress_bar=True)`.

### `tags`
**Type:** `list[str]` | **Default:** `[]`

Arbitrary metadata tags attached to the card (stored in serialised output).

### `login_env`
**Type:** `type[Settings] | None` | **Default:** `None`

A `pydantic_settings.Settings` **class** (not a path string), intended for
authenticating proprietary models via a `.env` file. Currently stored on the
card but never instantiated or read anywhere in the codebase — setting it has
no effect. Call `load_dotenv()` yourself if you need `.env`-based API keys.

---

## See also

- [ExpCard API](api/expcard.md) — class reference
- [ScannerModel API](api/scanner_model.md) — running experiments
- [Session Recovery](guides/session_recovery.md) — checkpointing guide
- [Custom Parsers](guides/custom_parsers.md) — structured output
