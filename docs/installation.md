# Installation

Tested on Python 3.11.

## 1. Create the uv environment

Install `uv` first if you don't have it ([astral.sh/uv](https://astral.sh/uv)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# already on conda? `conda install -c conda-forge uv` instead
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

Keys are loaded automatically from the environment for each provider family. Ollama running locally needs no key.
