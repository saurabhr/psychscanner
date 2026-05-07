# ExpCard & ExpCardInit

`ExpCard` and `ExpCardInit` are the central configuration objects in PsychScanner.
You populate an `ExpCardInit` with all experiment parameters, then pass it to `ExpCard`
which validates, resolves, and finalizes the configuration before handing it to `ScannerModel`.

---

## ExpCardInit

`ExpCardInit` is a Pydantic `BaseModel` holding every experiment parameter.
Set fields directly on an instance, or pass them as keyword arguments.

```python
from psychscanner import ExpCardInit

card = ExpCardInit()
card.model  = "gpt-4o-mini"
card.family = "openai"
card.nsim   = 20

# — or equivalently —
card = ExpCardInit(model="gpt-4o-mini", family="openai", nsim=20)
```

### Model & Provider

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"mock-chat-model"` | LLM model name (e.g. `"gpt-4o"`, `"llama3.1:8b"`, `"claude-3-5-sonnet-20241022"`) |
| `family` | `str` | `"mock-llm"` | Provider family — see [Supported Providers](../installation.md#set-api-keys) |
| `parameters` | `dict \| None` | `None` | Extra kwargs forwarded to the model constructor (e.g. `{"temperature": 0, "base_url": "..."}`) |

### Task & Persona

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_file` | `dict \| Path \| None` | VVIQ-16 default | Task JSON as an inline dict or path to a `.json` file |
| `persona_files` | `list[Path] \| None` | `None` | Path(s) to persona JSON files; auto-loads default personas when `None` |
| `cogtype` | `"assistant"` / `"custom"` / `"no"` | `"custom"` | Persona mode: `"custom"` = use persona files, `"assistant"` = single system prompt, `"no"` = skip personas (use `nsim`) |
| `nsim` | `int \| None` | `None` | Number of simulated agents — only used when `cogtype="no"` |
| `task_context` | `True \| False \| None` | `None` | Format each item as `context: … situation: …` (survey tasks only) |

### Memory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory` | `"SingleTurn"` / `"Convo"` | `"SingleTurn"` | `"SingleTurn"` = each trial is independent; `"Convo"` = trials chained in one conversation |
| `memory_k` | `int` | `-1` | Max messages kept in context: `-1` = unlimited, `N` = most recent N messages |
| `summary_k` | `int` | `0` | Summarization batch size (Convo only): `0` = disabled, `N` = summarize when context overflows |
| `chain_type` | `"item"` / `"trial"` / `"task"` / `None` | `None` | Overrides the task JSON `chain_type`; controls how stimuli are grouped within a trial |

### Parsing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parser` | `str \| type[BaseModel] \| Callable \| None` | `None` | Structured-output parser — see [Parser modes](#parser-modes) |
| `parser_raw` | `bool` | `False` | When `True`, the raw `AIMessage` is included in output as `"_raw"` |
| `parser_config` | `dict \| None` | `None` | Forwarded to `model.with_structured_output(**parser_config)`; default `{"method": "json_schema"}` |

#### Parser modes

| Value | Behaviour |
|---|---|
| `None` or `"0"` | No structured output — raw `AIMessage` in `pred_resp` |
| `"1"` | Read parser name from the task JSON `"parser"` field, resolve via registry |
| Any other `str` | Direct registry lookup: `get_parser("MyParser")` |
| `BaseModel` subclass | Used directly |
| `callable` (not a class) | Called per trial as `fn(trcode) → type[BaseModel] \| None` |

### Feedback

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feedback` | `bool` | `False` | Enable trial-level feedback injection |
| `feedback_fn` | `type[FeedbackBase] \| None` | `None` | **Class** (not instance) of a `FeedbackBase` subclass — required when `feedback=True` |

### Output & Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `proj_dir` | `Path` | `~/psychscanner` | Root directory; data saved to `proj_dir/projectname/taskname/family_model_memory/` |
| `projectname` | `str` | `"DEFAULTPROJ"` | Sub-directory name identifying this project |
| `tags` | `list[str]` | `[]` | Arbitrary metadata tags |
| `tunnel_status` | `"0"` / `"1"` | `"0"` | `"1"` enables session checkpointing for pause/resume |
| `tunnel_k` | `int` | `-1` | Checkpoint interval: `-1` = after all trials, `N` = every N trials |
| `enabletqdm` | `bool` | `False` | Show a tqdm progress bar during the run |
| `login_env` | `type[Settings] \| None` | `None` | Path to a `.env` file for API keys |

---

## ExpCard

`ExpCard` validates an `ExpCardInit`, resolves all paths and parsers, creates the output
directory, and instantiates the `SessionTunnel`. Pass it to `ScannerModel`.

```python
from psychscanner import ExpCard, ExpCardInit, ScannerModel

card_in = ExpCardInit(model="gpt-4o-mini", family="openai", nsim=5)
exp     = ExpCard(card_in)
scanner = ScannerModel(expcard=exp)
```

### Constructor

```python
ExpCard(cls: ExpCardInit | None = None, **kwargs)
```

`**kwargs` override any field in the supplied `ExpCardInit` (or build one from scratch if `cls` is omitted).

**Raises `ValueError`** if:
- `feedback=True` but `feedback_fn` is not set.
- The resolved parser name is not in the registry (when `parser="1"` or a string name).

### Key attributes after construction

| Attribute | Description |
|---|---|
| `card_in` | The validated `ExpCardInit` instance |
| `parser` | Resolved parser class or callable (or `None`) |
| `data_root_dir` | Full output path: `proj_dir/projectname/taskname/family_model_memory/` |

---

## Saving and loading experiment cards

### save_expcard

```python
from psychscanner import save_expcard

portable = save_expcard(exp.card_in, path="my_card.json")
```

Serializes the card to a **machine-independent** JSON dict:

- Task JSON and persona JSON are **embedded inline** (no filesystem paths).
- Registered parsers become their string name; custom classes become `module.qualname`.
- Machine-specific fields (`proj_dir`, `login_env`) and runtime state are omitted.

### load_expcard

```python
from psychscanner import load_expcard

card_in = load_expcard(
    "my_card.json",
    proj_dir="./results",           # override output root
    parser=MyCustomParser,          # required if original was a lambda or non-importable class
    feedback_fn=MyFeedback,         # required if original feedback_fn is not importable
)
exp = ExpCard(card_in)
```

Reconstructs an `ExpCardInit` from a portable file or dict. `proj_dir` defaults to `~/psychscanner` if omitted.

---

## Complete example

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

card = ExpCardInit(
    model          = "gemma3:4b",
    family         = "ollama",
    parameters     = {"temperature": 0},
    task_file      = Path("tasks/vviq16.json"),
    cogtype        = "no",
    nsim           = 10,
    memory         = "SingleTurn",
    parser         = DefaultLiteralVivid15,
    proj_dir       = Path("./results"),
    projectname    = "vviq_pilot",
    tunnel_status  = "1",
    enabletqdm     = True,
)

exp     = ExpCard(card)
scanner = ScannerModel(expcard=exp)
results = scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## See also

- [Configuration reference](../configuration.md) — all parameters with detailed descriptions
- [ScannerModel](scanner_model.md) — executing simulations
- [Parsers](parsers.md) — structured output
- [SessionTunnel](session_tunnel.md) — checkpointing
