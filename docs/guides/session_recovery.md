# Session Recovery

For large-scale experiments (hundreds of participants, slow models, HPC clusters),
PsychScanner can checkpoint progress so that interrupted runs resume exactly
where they left off.

---

## Enabling checkpointing

Set `tunnel_status="1"` on the `ExpCard`:

```python
card = ExpCardInit(
    ...
    tunnel_status = "1",
    tunnel_k      = -1,   # accepted but currently unused — see note below
)
```

> `tunnel_k` is stored on the card but is not currently read anywhere in the
> run loop — checkpointing always happens after every system prompt,
> regardless of its value.

A tunnel log file is written alongside the data:
```
proj_dir/projectname/taskname/family_model_memory/{projectname}-tunnel.json
```

---

## How it works

On every `ScannerModel.run()` call:

1. The tunnel log is opened.
2. If it contains a previous `END` marker the run is blocked — the session already completed.
3. If it contains partial checkpoints, `ScannerModel.tunnel_systemtrials()` reads the last `SCAN`
   entry's `len_completed_system_prompts` and computes the resume index.
4. System prompts (one per participant, in the common case) already logged are skipped.
5. Each completed system prompt triggers a `scan_checkpoint` log entry.
6. After all system prompts finish, an `END` entry is written.

Re-running the exact same script resumes from the next incomplete system prompt.
No code changes are needed.

---

## Typical HPC pattern

```python
# run.py  — run with:  sbatch run.py  (re-submit if wall-time exceeded)
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv

card = ExpCardInit(
    model         = "llama3.1:70b",
    family        = "groq",
    task_file     = Path("tasks/study.json"),
    cogtype       = "no",
    nsim          = 200,
    memory        = "SingleTurn",
    parser        = "1",
    proj_dir      = Path("/scratch/results"),
    projectname   = "large_study",
    tunnel_status = "1",     # ← enable checkpoint
    enabletqdm    = True,
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

Submit this script; if the job times out at participant 80, re-submitting it
will skip participants 0–79 and continue from 80.

---

## Inspecting the tunnel log

```python
from psychscanner import SessionTunnel
from pathlib import Path

tunnel = SessionTunnel(
    tunnel_status = "1",
    project_name  = "large_study",
    tunnel_dir    = Path("/scratch/results/large_study/study/groq_llama3.1:70b_SingleTurn"),
)

logs = tunnel.load_tunnel_logs(to_frame=True)
print(f"Completed participants: {len(tunnel.all_past_scans())}")
print(f"Last state: {tunnel.last_scan()['state']}")
```

---

## Resetting a session

To discard progress and start over, delete the tunnel file:

```python
import shutil
from pathlib import Path

# Delete only the tunnel log (keeps data files)
tunnel_file = Path("results/large_study/study/groq_llama3.1:70b_SingleTurn/large_study-tunnel.json")
tunnel_file.unlink(missing_ok=True)

# — or delete all data for this run —
shutil.rmtree("results/large_study")
```

---

## Saving and loading experiment cards

For reproducible experiments, serialize the full card configuration to a
portable JSON file before running:

```python
from psychscanner import save_expcard, load_expcard, ExpCard

# Save — embeds task JSON and persona JSON inline
save_expcard(exp.card_in, path="my_study_card.json")

# Reload on any machine
card_in = load_expcard("my_study_card.json", proj_dir="./results")
exp     = ExpCard(card_in)
```

The saved card contains no machine-specific paths; `proj_dir` is re-supplied at load time.

---

## See also

- [SessionTunnel API](../api/session_tunnel.md) — full method reference
- [ScannerModel.run()](../api/scanner_model.md) — resume behavior details
- [ExpCard](../api/expcard.md) — `tunnel_status`, `tunnel_k` parameters
