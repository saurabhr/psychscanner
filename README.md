<!-- start docs-include-index -->

# psychscanner

Psych Scanner: A framework for running psychological experiments with large language models.

[![Documentation Status](https://readthedocs.org/projects/psychscanner/badge/?version=latest)](https://psychscanner.readthedocs.io/en/latest/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/saurabhr/psychscanner/pulls)
[![Contributors](https://img.shields.io/github/contributors/saurabhr/psychscanner.svg)](https://github.com/saurabhr/psychscanner/graphs/contributors)

Tested on Python 3.11

![image](docs/psychscanner.png)

> The package is under active development. [Contributions](CONTRIBUTING.md) are welcome. Find the documentation here:https://psychscanner.readthedocs.io/en/latest/
>
> 
> Archived version of psychscanner 0.1.0 can be found here: https://github.com/saurabhr/psyschscanner_v_0_1_0

## Features

- **Automated Cognitive Tasks** — run validated psychological experiments (surveys, rating scales, cognitive tasks) with any LLM
- **Flexible Memory** — stateless (no-memory) and conversational (session memory) conditions in the same experiment
- **Persona System** — configure AI agents with system prompts, personality traits, and response formats
- **Multi-provider** — works with OpenAI, Anthropic, Groq, Mistral, Google, Ollama (local/remote), HuggingFace, and more via LangChain
- **Session Recovery** — checkpoint and resume long-running simulations with `SessionTunnel`
- **Structured Output** — export trial-level results to CSV with parsed response fields

## Installation

### 1. Create the uv environment

Install `uv` first if you don't have it ([astral.sh/uv](https://astral.sh/uv)):

```bash
conda install -c conda-forge uv                    # if you use conda
# curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
```

```bash
uv venv psyscan --python 3.11
source psyscan/bin/activate
```

### 2. Install psychscanner

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv pip install -e .
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

Keys are read automatically from `os.environ` for each provider family — psychscanner does not load `.env` files itself, so either `export` the variables in your shell or call `load_dotenv()` (from `python-dotenv`) before constructing the card. Ollama running locally needs no key.

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

Only `ollama` ships out of the box (`langchain-ollama` is a base dependency). Every other family needs its own LangChain integration package too, e.g. `uv pip install langchain-openai` for `openai`, `langchain-anthropic` for `anthropic` — see [LangChain's provider list](https://docs.langchain.com/oss/python/integrations/providers) for the rest.

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

## Citation

If you use PsychScanner in your research, please cite the framework paper:

```bibtex
@unpublished{ranjan2026psychscanner,
  author = {Ranjan, Saurabh and Sokratous, Konstantina and Makwana, Mukesh},
  title  = {Psych Scanner: A Framework for Systematic Cognitive Evaluation of Large Language Models},
  note   = {Manuscript submitted for publication},
  year   = {2026},
}
```

Task-specific citations (Reality Monitoring, VVIQ) and the full reference list live in [`CITATION.cff`](CITATION.cff) and the [docs](https://psychscanner.readthedocs.io/en/latest/#citation) — or use GitHub's "Cite this repository" button.

