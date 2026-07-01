# SessionTunnel

`SessionTunnel` is the checkpointing engine. It writes a structured log file (`.json`)
alongside each simulation run so that long experiments can be paused and resumed without
losing completed work.

You rarely interact with `SessionTunnel` directly — enable it by setting
`tunnel_status="1"` on your `ExpCard` and PsychScanner manages it automatically.

---

## How it works

When `tunnel_status="1"`, `ScannerModel.run()`:

1. Calls `tunnel.create_tunnel()` to open or re-open the log file.
2. Checks the log for a prior `"END"` marker — raises if the session already completed.
3. Reads the last `scan_checkpoint` to find the resume index.
4. Skips participants already logged; runs the rest.
5. Writes `scan_checkpoint` after each participant.
6. Calls `tunnel.end_checkpoint()` when all participants are done.

The tunnel log lives at:
```
proj_dir/projectname/taskname/family_model_memory/{projectname}-tunnel.json
```

---

## Constructor

```python
SessionTunnel(
    tunnel_status: str,      # "0" or "1"
    project_name: str,
    tunnel_dir: Path,
)
```

| Parameter | Description |
|-----------|-------------|
| `tunnel_status` | `"0"` disables all logging; `"1"` enables |
| `project_name` | Used as the log filename stem |
| `tunnel_dir` | Directory where the log file is written |

---

## Methods

### create_tunnel()

```python
tunnel.create_tunnel(tunnel_file: Path | None = None) -> None
```

Opens (or re-opens) the log file and writes a `"BEGIN"` CRITICAL entry.

**Raises `ValueError`** if the log already contains an `"END"` entry — the session
completed previously. Delete the tunnel file (or the entire project directory) to re-run.

### end_checkpoint()

```python
tunnel.end_checkpoint() -> None
```

Writes an `"END"` CRITICAL entry. Called automatically by `ScannerModel.run()` on completion.

### scan_checkpoint()

```python
tunnel.scan_checkpoint(
    session_id: str,
    run_type: str = ">>>>SCAN<<<<",
    state: object = "--",
) -> None
```

Writes an INFO-level entry recording that one participant (system message) was completed.
`ScannerModel.run()` calls this after each participant with the participant index as `session_id`.

### subscan_checkpoint()

```python
tunnel.subscan_checkpoint(
    subscan_id: str,
    run_type: str = ">>SUB_SCAN<<",
    state: str = "--",
) -> None
```

Writes a TRACE-level entry for trial-level checkpointing (finer granularity than `scan_checkpoint`).

### load_tunnel_logs()

```python
tunnel.load_tunnel_logs(
    tunnel_file: Path | None = None,
    *,
    return_all: bool = False,
    to_frame: bool = False,
) -> list[dict] | pd.DataFrame
```

Reads and deserializes the tunnel log. `return_all` and `to_frame` are keyword-only.

Both settings return the same set of log entries (all levels — CRITICAL/INFO/TRACE —
are included either way; nothing is filtered by level):

| Parameter | Description |
|-----------|-------------|
| `return_all` | `True` = return the raw loguru log dicts (full nested `record` structure); `False` (default) = return a flattened subset with just `timestamp`, `level`, `run_type`, `session_id`, `state`, `message` |
| `to_frame` | `True` = return a pandas DataFrame |

### Inspection helpers

| Method | Returns | Description |
|--------|---------|-------------|
| `all_past_scans()` | `pd.DataFrame` | All INFO entries (one per completed participant) |
| `all_past_sscans()` | `pd.DataFrame` | All TRACE entries (trial-level) |
| `last_scan()` | `pd.Series` | Most recent scan checkpoint |
| `last_sscan()` | `pd.Series` | Most recent subscan checkpoint |
| `get_last_state(tunnel_status, subscan)` | `pd.Series \| None` | Last state; `None` when `tunnel_status="0"` |

---

## Enabling checkpointing

```python
from psychscanner import ExpCardInit, ExpCard, ScannerModel

card = ExpCardInit(
    model         = "gpt-4o-mini",
    family        = "openai",
    nsim          = 100,
    tunnel_status = "1",   # enable
    tunnel_k      = -1,    # accepted but currently unused by the run loop
    proj_dir      = "./results",
    projectname   = "large_study",
    ...
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run()
```

If the process is killed at participant 47, re-running the same script resumes from participant 48.

---

## Inspecting the log manually

```python
from psychscanner import SessionTunnel
from pathlib import Path

tunnel = SessionTunnel(
    tunnel_status="1",
    project_name="large_study",
    tunnel_dir=Path("results/large_study/vviq/openai_gpt-4o-mini_SingleTurn"),
)

logs = tunnel.load_tunnel_logs(to_frame=True)
print(logs[["time", "level", "run_type", "state"]])

print("Completed participants:", len(tunnel.all_past_scans()))
print("Last checkpoint:",       tunnel.last_scan()["state"])
```

---

## Resetting a session

To re-run from scratch, delete the tunnel file:

```python
from pathlib import Path

tunnel_file = Path("results/large_study/vviq/openai_gpt-4o-mini_SingleTurn/large_study-tunnel.json")
tunnel_file.unlink(missing_ok=True)
```

Or delete the entire project data directory.

---

## See also

- [Session Recovery Guide](../guides/session_recovery.md) — practical patterns
- [ExpCard](expcard.md) — `tunnel_status` and `tunnel_k` parameters
- [ScannerModel](scanner_model.md) — how `run()` uses the tunnel
