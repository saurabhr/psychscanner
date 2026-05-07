# CLI Reference

PsychScanner can be run from the command line for quick experiments without
writing a Python script.

---

## Installation check

After installing, verify the package is available:

```bash
python -m psychscanner --version
```

---

## Python API (recommended)

For most use cases, the Python API gives you full control and is the recommended
approach:

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    cogtype     = "no",
    nsim        = 20,
    memory      = "SingleTurn",
    parser      = DefaultLiteralVivid15,
    proj_dir    = Path("./results"),
    projectname = "my_study",
)

scanner = ScannerModel(expcard=ExpCard(card))
scanner.run(progress_bar=True)
to_csv(scanner, path=card.proj_dir)
```

---

## Running example notebooks

The `examples/` directory contains Jupyter notebooks covering all major features:

```bash
cd psychscanner
jupyter lab examples/
```

| Notebook | Topic |
|----------|-------|
| `00_quickstart.ipynb` | Minimal working example (Groq) |
| `01_ollama_local_models.ipynb` | Local models via Ollama |
| `02_parameters_reference.ipynb` | Full `ExpCard` parameter reference |
| `03_parsers.ipynb` | Response parsing overview |
| `04_parser_modules.ipynb` | Custom parser modules |
| `05_rm_task.ipynb` | Reality monitoring cognitive task |
| `06_feedback_api.ipynb` | Feedback / scoring API |
| `07_rm_feedback_task.ipynb` | Reality monitoring with feedback |
| `08_ps_parser_guide.ipynb` | Structured output parsing guide |
| `09_vviq16_study.ipynb` | VVIQ-16 imagery questionnaire study |

---

## Running a script directly

For HPC clusters or automated pipelines, write a plain Python script and run it
with `sbatch` or a scheduler:

```bash
# run.py
python run.py

# SLURM
sbatch --time=04:00:00 run.py
```

Enable checkpointing (`tunnel_status="1"`) so that if the job is killed by a
wall-time limit, re-submitting resumes from the last completed participant.

---

## Environment variables

```bash
# Set API keys before running
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...

# Or use a .env file (auto-loaded by psychscanner)
cat .env
# OPENAI_API_KEY=sk-...
```

---

## See also

- [Installation](installation.md) — setting up the environment
- [Configuration Reference](configuration.md) — all `ExpCardInit` parameters
- [Session Recovery](guides/session_recovery.md) — HPC checkpoint pattern
