# Installation

Tested on Python 3.11.

## 1. Create the uv environment

Install `uv` first if you don't have it ([astral.sh/uv](https://astral.sh/uv)):

```bash
conda install -c conda-forge uv                    # if you use conda
# curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
```

```bash
uv venv psyscan --python 3.11
source psyscan/bin/activate
```

## 2. Install psychscanner

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv pip install -e .
```

> **The task/experiment library is dev-only, by design.** `task_library()`
> and `experiment_library()` search plain directories on disk
> (`examples/tasks/`, `examples/experiments/`, or your own) — those
> directories, and the bundled default task cards under
> `src/psychscanner/datasets/prompts/defaults/`, are **not** included in a
> `pip install psychscanner` from PyPI (verified: 0 `.json` files ship in
> the built wheel). They're only present when you clone the repo, as
> above. If you just need one task card without a full clone, download it
> directly from
> [`examples/tasks/`](https://github.com/saurabhr/psychscanner/tree/main/examples/tasks)
> on GitHub and pass it to `ExpCardInit(task_file=...)` or
> `task_library(..., dirs=<wherever you saved it>)`.

## 3. Set API keys

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
