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

## Installation

```bash
conda install -c conda-forge uv                    # if you use conda
# curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
uv venv psyscan --python 3.11
source psyscan/bin/activate

git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv pip install -e .
```

See [Installation](installation.md) for API key setup and full details.

---

## Quick Start

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, to_csv
from psychscanner.parsers import DefaultLiteralVivid15

# 1. Configure the experiment
card = ExpCardInit(
    model       = "gpt-4o-mini",
    family      = "openai",
    projectname = "my_experiment",
    proj_dir    = Path("./results"),
    cogtype     = "no",         # no persona files — set participant count directly
    nsim        = 10,           # 10 simulated participants
    memory      = "SingleTurn", # "SingleTurn" (stateless) or "Convo" (chained)
    parser      = DefaultLiteralVivid15,
)

# 2. Run (uses built-in VVIQ-16 imagery questionnaire by default)
scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()

# 3. Export
to_csv(scanner, path=card.proj_dir)
```

---

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
| `cohere` | `COHERE_API_KEY` | Cohere Command models |
| `openrouter` | `OPENROUTER_API_KEY` | Routed through the OpenAI-compatible endpoint at openrouter.ai |

---

## Tutorials

Every notebook in [`examples/`](https://github.com/saurabhr/psychscanner/tree/main/examples) is also rendered in the [Tutorials](tutorials/index.md) section with its saved outputs:

| Notebook | Description |
|---|---|
| [`00_quickstart.ipynb`](tutorials/00_quickstart.ipynb) | Minimal working example |
| [`01_ollama_local_models.ipynb`](tutorials/01_ollama_local_models.ipynb) | Local models via Ollama |
| [`02_parameters_reference.ipynb`](tutorials/02_parameters_reference.ipynb) | Full `ExpCard` parameter reference |
| [`03_parsers.ipynb`](tutorials/03_parsers.ipynb) | Response parsing overview |
| [`04_parser_modules.ipynb`](tutorials/04_parser_modules.ipynb) | Custom parser modules |
| [`05_rm_task.ipynb`](tutorials/05_rm_task.ipynb) | Reality monitoring task |
| [`06_feedback_api.ipynb`](tutorials/06_feedback_api.ipynb) | Feedback / scoring API |
| [`07_rm_feedback_task.ipynb`](tutorials/07_rm_feedback_task.ipynb) | Reality monitoring with feedback |
| [`08_ps_parser_guide.ipynb`](tutorials/08_ps_parser_guide.ipynb) | Structured output parsing guide |
| [`09_vviq16_study.ipynb`](tutorials/09_vviq16_study.ipynb) | VVIQ-16 imagery questionnaire study |

---

## References

Papers that motivate PsychScanner's design and the instruments it ships:

- Lin, Z. **Large Language Models as Psychological Simulators: A Methodological Guide.** [arXiv:2506.16702](https://arxiv.org/abs/2506.16702), 2025 — methodology for using LLMs as simulated participants, the core premise behind `ScannerModel`.
- Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P. & Hashimoto, T. **Whose Opinions Do Language Models Reflect?** [arXiv:2303.17548](https://arxiv.org/abs/2303.17548), 2023 — how persona/steering choices shift which population a model's simulated responses resemble; relevant to `cogtype`/`persona_files`.
- Ranjan, S., Sokratous, K. & Odegaard, B. **Reality Monitoring in Large Language Models: Self-Knowledge That Transforms with Conversation Memory.** [arXiv:2607.23927](https://arxiv.org/abs/2607.23927), 2026 — LLM source-monitoring accuracy, the paradigm behind the bundled Reality Monitoring (RM) task.
- McCarty, M. & Morales, J. **Artificial Phantasia: Emergent Mental Imagery in Large Language Models.** [arXiv:2509.23108](https://arxiv.org/abs/2509.23108), 2025 — uses the VVIQ (the package's default task) to probe emergent mental imagery in LLMs.
- Ranjan, S. & Odegaard, B. **Psychological Imagination Networks Show Cross-Population Centrality and Clustering Alignment in Humans That Large Language Models Fail to Replicate.** [arXiv:2510.04391](https://arxiv.org/abs/2510.04391), 2025 — network-science analysis of VVIQ ratings across humans and LLMs, from the psychscanner team.
- Sorokovikova, A., Fedorova, N., Rezagholi, S. & Yamshchikov, I. P. **LLMs Simulate Big Five Personality Traits: Further Evidence.** [arXiv:2402.01765](https://arxiv.org/abs/2402.01765), 2024 — Big Five / BFI trait simulation, relevant to the bundled `bfi44` task.

Every paper cited here and across the [Tutorials](tutorials/index.md) is collected as BibTeX in [`references.bib`](references.bib).

---

## Citation

If you use PsychScanner in your research, please cite the framework paper:

```
Ranjan, S., Sokratous, K. & Makwana, M. Psych Scanner: A Framework for Systematic Cognitive Evaluation of
Large Language Models. [Under Submission]
```

```bibtex
@unpublished{ranjan2026psychscanner,
  author = {Ranjan, Saurabh and Sokratous, Konstantina and Makwana, Mukesh},
  title  = {Psych Scanner: A Framework for Systematic Cognitive Evaluation of Large Language Models},
  note   = {Manuscript submitted for publication},
  year   = {2026},
}
```

If you use the bundled Reality Monitoring (RM) task, please also cite:

```
Ranjan, S., Sokratous, K., & Odegaard, B. (2026). Reality Monitoring in Large Language Models: Self-Knowledge
That Transforms with Conversation Memory. arXiv preprint arXiv:2607.23927.
```

arXiv: [2607.23927](https://arxiv.org/abs/2607.23927)

```bibtex
@misc{ranjan2026reality,
      title={Reality Monitoring in Large Language Models: Self-Knowledge That Transforms with Conversation Memory},
      author={Saurabh Ranjan and Konstantina Sokratous and Brian Odegaard},
      year={2026},
      eprint={2607.23927},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.23927},
}
```

If you use the bundled VVIQ / imagination-network analysis, please also cite:

```
Ranjan, S., & Odegaard, B. (2025). Psychological Imagination Networks Show Cross-Population Centrality and
Clustering Alignment in Humans That Large Language Models Fail to Replicate. arXiv preprint arXiv:2510.04391.
```

arXiv: [2510.04391](https://arxiv.org/abs/2510.04391)

```bibtex
@misc{ranjan2025psychological,
      title={Psychological Imagination Networks Show Cross-Population Centrality and Clustering Alignment in Humans That Large Language Models Fail to Replicate},
      author={Saurabh Ranjan and Brian Odegaard},
      year={2025},
      eprint={2510.04391},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.04391},
}
```

---

## Related Projects

- **psychscanner-primal** — the slim, Hub-optimized alternative distribution of this package, for RL/eval tooling like the Prime Intellect Environments Hub.

Docs: [saurabhr.github.io/psychscanner-primal](https://saurabhr.github.io/psychscanner-primal/)

- **metasignal** — a companion package from the Psych Scanner group on metacognitive science.

```
Ranjan, S., Makwana, M., Sokratous, K. & Odegaard, B. (2026). metasignal: A Python Package for Comprehensive
Metacognitive Analysis and Decision-Making. arXiv preprint arXiv:2607.29093.
```

Docs: [metasignal.readthedocs.io](https://metasignal.readthedocs.io/en/latest/)
arXiv: [2607.29093](https://arxiv.org/abs/2607.29093)
