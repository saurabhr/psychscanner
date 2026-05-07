<!-- start docs-include-index -->

# psychscanner

PsychScanner: A framework for running psychological experiments with large language models.

Tested on Python 3.11

![image](docs/psychscanner.png)

> The package is under active development. Contributions are welcome.

## Features

- **Automated Cognitive Tasks** — run validated psychological experiments (surveys, rating scales, cognitive tasks) with any LLM
- **Flexible Memory** — stateless (no-memory) and conversational (session memory) conditions in the same experiment
- **Persona System** — configure AI agents with system prompts, personality traits, and response formats
- **Multi-provider** — works with OpenAI, Anthropic, Groq, Mistral, Google, Ollama (local/remote), HuggingFace, and more via LangChain
- **Session Recovery** — checkpoint and resume long-running simulations with `SessionTunnel`
- **Structured Output** — export trial-level results to CSV with parsed response fields

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate psyscan
```

### 2. Install psychscanner

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
pip install -e .
```

### 3. Set API keys

Create a `.env` file in your project directory (or export variables in your shell):

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=hf_...

# Ollama remote server (only needed when not using localhost)
OLLAMA_API_KEY=...
```

Keys are loaded automatically from the environment for each provider family. Ollama running locally needs no key.

## Quick Start

This is the same code pinned by `tests/test_readme_quickstart.py::test_readme_quickstart_live_ollama`.
It runs against a local Ollama smol model (no API key) — pull it once with
`ollama pull smollm2:360m-instruct-fp16`, then run:

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

# 1. Configure the experiment
card = ExpCardInit(
    model       = "smollm2:360m-instruct-fp16",   # local Ollama; alt: "openai/gpt-oss-120b" with family="groq"
    family      = "ollama",
    parameters  = {"temperature": 0},
    projectname = "readme_quickstart",
    proj_dir    = Path.cwd() / "results",         # output goes to ./results in your CWD
    cogtype     = "no",                           # no persona files — use nsim instead
    nsim        = 1,                              # 1 simulated participant (bump up for real studies)
    memory      = "SingleTurn",
    parser      = DefaultLiteralVivid15,
)

# 2. Run (uses built-in VVIQ-16 imagery questionnaire by default)
scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()

# 3. Export to CSV (auto-named under proj_dir)
df = to_csv(scanner, path=card.proj_dir)
```

## Supported Providers

| Family | Env var | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, o1, … |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 3.5, Claude 3 Haiku, … |
| `groq` | `GROQ_API_KEY` | Llama 3, Mixtral on Groq Cloud |
| `mistral` | `MISTRAL_API_KEY` | Mistral, Codestral |
| `google` / `gemini` | `GOOGLE_API_KEY` | Gemini 2.0, 1.5 |
| `together` | `TOGETHER_API_KEY` | Together.ai hosted models |
| `fireworks` | `FIREWORKS_API_KEY` | Fireworks.ai |
| `azure` | `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `huggingface` | `HUGGINGFACEHUB_API_TOKEN` | HuggingFace Inference API |
| `ollama` | — (local) / `OLLAMA_API_KEY` (remote) | Pass `base_url` in `parameters` for remote |

## Examples

| Notebook | Description |
|---|---|
| `00_quickstart.ipynb` | Minimal working example |
| `01_ollama_local_models.ipynb` | Local models via Ollama |
| `02_parameters_reference.ipynb` | Full `ExpCard` parameter reference |
| `03_parsers.ipynb` | Response parsing overview |
| `04_parser_modules.ipynb` | Custom parser modules |
| `05_rm_task.ipynb` | Reality monitoring task |
| `06_feedback_api.ipynb` | Feedback / scoring API |
| `07_rm_feedback_task.ipynb` | Reality monitoring with feedback |
| `08_ps_parser_guide.ipynb` | Structured output parsing guide |
| `09_vviq16_study.ipynb` | VVIQ-16 imagery questionnaire study |


