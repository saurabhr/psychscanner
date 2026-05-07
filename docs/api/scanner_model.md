# ScannerModel

`ScannerModel` is the main simulation engine. It takes a validated `ExpCard`, builds
the agent, and runs every simulated participant through every trial.

---

## Constructor

```python
ScannerModel(expcard: ExpCard)
```

Initialises the scanner from a validated experiment card:

- Extracts project metadata (`projectname`, `proj_dir`, `data_root_dir`)
- Builds system prompts and task trial lists from `expcard`
- Creates an `AgentConfig` with the LLM, memory settings, and parser
- Sets up the feedback flag and session tunnel reference

```python
from psychscanner import ExpCard, ExpCardInit, ScannerModel

card    = ExpCardInit(model="gpt-4o-mini", family="openai", nsim=5)
exp     = ExpCard(card)
scanner = ScannerModel(expcard=exp)
```

---

## run()

```python
scanner.run(
    progress_bar: bool = False,
    feedback: Any | None = None,
    feedback_fn: Callable | None = None,
    save_str: str | None = None,
    tunnel: Any | None = None,
) -> list[list[dict]]
```

Executes the full simulation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progress_bar` | `bool` | `False` | Show tqdm bar per system message (overrides `expcard.enabletqdm`) |
| `feedback` | `Any` | `None` | Deprecated — set `feedback=True` on the `ExpCard` instead |
| `feedback_fn` | `Callable` | `None` | Override the feedback handler at runtime |
| `save_str` | `str` | `None` | Custom suffix appended to `.psyscan` output filenames |
| `tunnel` | `Any` | `None` | Override the session tunnel at runtime |

### Return value

A `list[list[dict]]` where:

- **Outer list** — one entry per simulated participant (system message / persona)
- **Inner list** — one `dict` per trial

Each trial dict contains:

| Key | Description |
|-----|-------------|
| `trial_idx` | Integer index of the trial |
| `trcode` | Trial code string from the task JSON |
| `stimulus` | The stimulus shown to the model |
| `pred_resp` | Model response — parsed dict when a parser is set, raw `AIMessage` otherwise |
| `fb_response` | Feedback string from the previous trial (or `None`) |
| `tunnel_id` | Checkpoint identifier for this simulation run |
| `system_message_idx` | Index of the system message (participant index) |
| `model`, `family`, `memory`, … | Metadata copied from the `ExpCard` |

### Session resume behavior

When `tunnel_status="1"` on the `ExpCard`, `run()` automatically:

1. Checks the tunnel log for existing checkpoints
2. Skips participants whose scans are already complete
3. Resumes from the last completed checkpoint

Re-run the same script after interruption — no code changes needed.

---

## model_dump()

```python
scanner.model_dump(
    data: object | None = None,
    sim_idx: int | str = "curr_scan",
    data_root_dir: Path | None = None,
    save_str: str = "iterations",
    data_type: str = "session",
) -> None
```

Saves scan data to a `.psyscan` file (JSON).

| Parameter | Description |
|-----------|-------------|
| `data` | Data to save; defaults to `self.current_scanner_data` |
| `sim_idx` | Participant index used in the filename |
| `data_root_dir` | Output directory; defaults to `expcard.data_root_dir` |
| `save_str` | Filename suffix |
| `data_type` | `"session"` uses `SimulationModel`; `"task"` uses `TaskSimulationModel` |

Output path: `data_root_dir/{sim_idx}-{save_str}.psyscan`

---

## Exporting results

After `run()`, export results to CSV with `to_csv` or concatenate multiple runs:

```python
from psychscanner import to_csv, concat_csv

# Single run → CSV
to_csv(scanner, path=card.proj_dir)

# Multiple runs → one CSV
df = concat_csv([scanner1, scanner2], path=card.proj_dir)
```

### to_csv()

```python
to_csv(
    source,
    path: str | Path | None = None,
    *,
    expcard=None,
    sep: str = ",",
    combined: bool = False,
) -> pl.DataFrame
```

`source` accepts a `ScannerModel`, the list returned by `.run()`, an `ExpCard`, or a directory path.
Returns a Polars DataFrame and writes a `.csv` file.

**CSV columns:** `sim_idx`, `model`, `family`, `memory`, `projectname`, `taskname`,
`chain_type`, `trial_idx`, `trcode`, `stimulus`, `pred_resp_raw`, `resp_*` (one per parser field), `fb_response`, `tunnel_id`, and more.

### concat_csv()

```python
concat_csv(
    sources: list,
    path: str | Path | None = None,
    *,
    sep: str = ",",
) -> pl.DataFrame
```

Concatenates multiple `to_csv`-compatible sources into a single aligned DataFrame.
Missing columns are filled with `null`; numeric `resp_*` columns are coerced to `Float64`.

---

## Full example

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv

card = ExpCardInit(
    model        = "llama3.1:8b",
    family       = "ollama",
    parameters   = {"temperature": 0},
    task_file    = Path("tasks/vviq16.json"),
    cogtype      = "no",
    nsim         = 30,
    memory       = "SingleTurn",
    parser       = "1",           # resolve from task JSON
    proj_dir     = Path("./results"),
    projectname  = "vviq_study",
    tunnel_status = "1",
    enabletqdm   = True,
)

exp     = ExpCard(card)
scanner = ScannerModel(expcard=exp)
results = scanner.run(progress_bar=True)

# Export
to_csv(scanner, path=card.proj_dir)
print(f"Saved {len(results)} participant runs")
```

---

## See also

- [ExpCard & ExpCardInit](expcard.md) — configuration
- [SessionTunnel](session_tunnel.md) — checkpointing & resume
- [Parsers](parsers.md) — structured output
- [FeedbackBase](../guides/custom_parsers.md) — trial feedback
