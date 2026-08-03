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
| [Reality Monitoring with Feedback](07_rm_feedback_task.ipynb) | Combining the RM task with feedback. Reads its task JSON from `RM_TASKS_DIR` (defaults to `examples/tasks/`) — bring your own RM word-pair files with feedback fields, since these aren't bundled |

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

## Tool binding, multimodal stimuli, and custom agent architectures

| Notebook | What it covers |
|---|---|
| [Tool Binding and Multimodal Stimuli](12_tool_binding_and_multimodal.ipynb) | `card.tools`, per-trial tool subsetting, the four multimodal content-block builders, and the same stimulus run under every `chain_type`/`memory` combination |
| [A Real Tool-Calling Loop](13_react_tool_agent.ipynb) | `psychscanner.agents.make_react_agent` — LangChain's `create_agent` ReAct loop, adapted to the `ScanningAgent` contract |
| [A Multimodal Supervisor/Planner/Worker Agent](14_supervisor_multiagent.ipynb) | `psychscanner.agents.make_supervisor_agent` — a Jockey-style (TwelveLabs + LangGraph) routing agent generalized from video to arbitrary multimodal content blocks |
| [A Modular Planner/Executor/Validator Agent](15_planner_executor_agent.ipynb) | `psychscanner.agents.make_planner_executor_agent` — the modular agentic architecture from arxiv:2310.00194 |
| [Reflection Agents](16_reflection_agents.ipynb) | `psychscanner.agents.reflection_agents` — Basic Reflection, Reflexion, and LATS, after the LangChain reflection-agents blog post |

## Cognitive reinforcement learning

| Notebook | What it covers |
|---|---|
| [Cognitive RL: Bandits and the Prisoner's Dilemma](17_cognitive_rl_bandits_and_pd.ipynb) | Reusing the planner/executor/validator agent (arxiv:2310.00194) with a `FeedbackBase`-based Q-learning environment, on an n-armed bandit and an iterated Prisoner's Dilemma vs. Tit-for-Tat |

## Running these yourself

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
pip install -e ".[dev]"
jupyter lab examples/
```

For narrative, task-oriented walkthroughs (rather than notebook cell-by-cell tutorials), see the [Examples](../examples/index.md) section.

## Further reading

Each notebook above ends with a "Further reading" cell citing two papers showing advanced applications of that notebook's technique. BibTeX entries for every paper cited across the tutorials are collected in [`references.bib`](../references.bib).
