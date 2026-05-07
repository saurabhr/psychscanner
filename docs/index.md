# PsychScanner

**A framework for running psychological experiments with large language models.**

![PsychScanner](psychscanner.png)

> The package is under active development. Contributions are welcome.

---

## Features

- **Automated Cognitive Tasks** — run validated psychological experiments (surveys, rating scales, cognitive tasks) with any LLM
- **Flexible Memory** — stateless (no-memory) and conversational (session memory) conditions in the same experiment
- **Persona System** — configure AI agents with system prompts, personality traits, and response formats
- **Multi-provider** — works with OpenAI, Anthropic, Groq, Mistral, Google, Ollama (local/remote), HuggingFace, and more via LangChain
- **Session Recovery** — checkpoint and resume long-running simulations with `SessionTunnel`
- **Structured Output** — export trial-level results to CSV with parsed response fields

---

## Quick Start

```python
from pathlib import Path
from psychscanner import ExpCard, ExpCardInit, ScannerModel

# 1. Configure the experiment
card = ExpCardInit()
card.model         = "gpt-4o-mini"
card.family        = "openai"
card.projectname   = "my_experiment"
card.proj_dir      = Path("./results")
card.system_prompt = "You are a human participant in a psychology study."
card.task_prompt   = "Rate how vividly you can imagine the following scene (1–5): {scene}"
card.response_format = {"Vividness": "int (1–5)"}
card.nsim          = 10   # number of simulated participants
card.memory_type   = "0"  # "0" = no memory, "1" = conversation memory

# 2. Run
scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()

# 3. Export
from psychscanner import to_csv
to_csv(scanner, path=card.proj_dir)
```

---

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

---

## Example Notebooks

| Notebook | Description |
|---|---|
| `00_quickstart.ipynb` | Minimal working example |
| `01_ollama_local_models.ipynb` | Local models via Ollama |
| `02_parameters_reference.ipynb` | Full `ExpCard` parameter reference |
| `03_parsers.ipynb` | Response parsing overview |
| `04_parser_modules.ipynb` | Custom parser modules |
| `05_rm_task.ipynb` | Reality monitoring task |
| `06_feedback_api.ipynb` | Feedback / scoring API |
| `08_ps_parser_guide.ipynb` | Structured output parsing guide |
| `09_vviq16_study.ipynb` | VVIQ-16 imagery questionnaire study |

---

## Citation

If you use PsychScanner in your research, please cite:

```
Ranjan, S. (2025). PsychScanner: A framework for simulating psychological experiments
with large language models. [Preprint]
```
