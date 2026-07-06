# Tutorials

These tutorials are the Jupyter notebooks from [`examples/`](https://github.com/saurabhr/psychscanner/tree/main/examples) in the repository, rendered here with their saved outputs so you can read them without running anything. Each page also has a download link to the raw `.ipynb` file if you want to run it yourself.

## Getting started

| Notebook | What it covers |
|---|---|
| [Quickstart](00_quickstart.ipynb) | Minimal working example — configure an `ExpCard`, run a scanner, export to CSV |
| [Local Models with Ollama](01_ollama_local_models.ipynb) | Running experiments against local models with no API key |
| [Parameters Reference](02_parameters_reference.ipynb) | A tour of every `ExpCard` / `ExpCardInit` parameter |

## Parsing responses

| Notebook | What it covers |
|---|---|
| [Parsers](03_parsers.ipynb) | Overview of the built-in response parsers |
| [Parser Modules](04_parser_modules.ipynb) | Writing and registering custom parser modules |
| [Parser System Guide](08_ps_parser_guide.ipynb) | Deep dive into the parser dispatch system |

## Cognitive tasks and feedback

| Notebook | What it covers |
|---|---|
| [Reality Monitoring Task](05_rm_task.ipynb) | Reality monitoring across memory and feedback configurations |
| [Feedback API](06_feedback_api.ipynb) | The trial-by-trial feedback/scoring mechanism |
| [Reality Monitoring with Feedback](07_rm_feedback_task.ipynb) | Combining the RM task with feedback |

## Full study walkthrough

| Notebook | What it covers |
|---|---|
| [VVIQ-16 Study](09_vviq16_study.ipynb) | End-to-end imagery questionnaire study, from run to figure |

## Extending psychscanner

| Notebook | What it covers |
|---|---|
| [Bring Your Own LLM/VLM](10_custom_agents.ipynb) | Wrapping any custom LLM or VLM callable with `psychscanner.agents.CustomAgent` and running it through `TaskRunner` / `ScannerModel` |

## Memory and context

| Notebook | What it covers |
|---|---|
| [Conversation Memory & Context Quantization](11_memory_context_management.ipynb) | `Convo` memory under `memory_k` (hard truncation window) and `summary_k` (folding overflow into a rolling summary) |

## Running these yourself

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
pip install -e ".[dev]"
jupyter lab examples/
```

For narrative, task-oriented walkthroughs (rather than notebook cell-by-cell tutorials), see the [Examples](../examples/index.md) section.
