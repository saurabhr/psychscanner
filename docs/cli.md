# CLI Reference

Installing the package registers a `psychscanner` console command
(`pyproject.toml`'s `[project.scripts]` entry point), which you can also invoke
as `python -m psychscanner`. Both forms run the same `click` command defined in
`src/psychscanner/cli.py`.

> **Current behavior:** the CLI currently just parses and echoes its options
> back to stdout — it does **not** build an `ExpCard`/`ScannerModel` or run an
> experiment yet. Use it to sanity-check option parsing, or use the Python API
> below to actually run experiments.

---

## Installation check

After installing, verify the package is available:

```bash
python -m psychscanner --version
# or
psychscanner --version
```

---

## Options

```bash
psychscanner --help
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | `str` | — | Model name passed to the provider |
| `-f, --family` | `str` | — | Provider family (`openai`, `anthropic`, `ollama`, …) |
| `-p, --parameters` | `dict` | `None` | Extra kwargs forwarded to the model constructor |
| `-mem, --memory` | `Choice(SingleTurn, Convo)` | `SingleTurn` | Memory mode |
| `-memk, --memory_k` | `int` | `-1` | Number of past interactions to keep |
| `-pers, --persona_files` | `list[Path]` | `None` | Persona JSON file(s) |
| `-t, --task_file` | `Path` | `None` | Task JSON file |
| `-tc, --task_context` | `Choice(True, False, None)` | `None` | Whether to include context text in prompts |
| `-tus, --tunnel_status` | `Choice("0", "1")` | `"0"` | Enable checkpointing (`"1"`) |
| `-tuk, --tunnel_k` | `int` | `-1` | Accepted but currently unused by the run loop |
| `-projname, --projectname` | `str` | `"DEFAULTPROJ"` | Project name / output sub-folder |
| `-tg, --tags` | `list[str]` | `[]` | Arbitrary metadata tags |
| `-pa, --parser` | `str` | `"0"` | Parser name, or `"0"` for none |
| `-praw, --parser_raw` | `bool` | `False` | Keep the raw `AIMessage` alongside the parsed response |
| `-pcon, --parser_config` | `dict` | `None` | Forwarded to `with_structured_output()` |
| `-pd, --proj_dir` | `Path` | `~/psychscanner` | Root output directory |
| `-le, --login_env` | `str` | `None` | Path to a `.env` file for API keys |
| `-tq, --enabletqdm` | flag | `False` | Show a tqdm progress bar |
| `-v, --version` | flag | — | Print the installed version and exit |
| `-h, --help` | flag | — | Show help and exit |

---

## Python API (recommended)

For actually running experiments today, use the Python API, which gives you
full control:

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
```

Or keep them in a `.env` file and load it yourself before constructing the
card — psychscanner does **not** load `.env` files automatically (the
`login_env` field exists but is currently unused):

```python
from dotenv import load_dotenv
load_dotenv()  # requires `pip install python-dotenv`
```

---

## See also

- [Installation](installation.md) — setting up the environment
- [Configuration Reference](configuration.md) — all `ExpCardInit` parameters
- [Session Recovery](guides/session_recovery.md) — HPC checkpoint pattern
