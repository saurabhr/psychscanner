# Troubleshooting

Common problems and their solutions.

---

## API key errors

### `AuthenticationError` / `401 Unauthorized`

The API key for the selected provider is missing or invalid.

**Check your `.env` file:**
```bash
# .env — must be in the directory where you run the script
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

**Verify the key is being loaded:**
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))   # should print your key, not None
```

**Pass the key directly (as a fallback):**
```python
card = ExpCardInit(
    model      = "gpt-4o-mini",
    family     = "openai",
    parameters = {"api_key": "sk-..."},
)
```

---

## Session already ended

### `ValueError: Session already has ended. Delete old files to run.`

This means a previous run completed successfully and left an `END` marker in the
tunnel log. Re-running is blocked to prevent accidental data overwriting.

**To start a fresh run, delete the project data:**
```python
import shutil
from pathlib import Path

# Delete only the tunnel log (keeps raw data)
tunnel = Path("results/my_project/my_task/openai_gpt-4o-mini_SingleTurn/my_project-tunnel.json")
tunnel.unlink(missing_ok=True)

# — or delete the entire run directory —
shutil.rmtree("results/my_project")
```

If you don't need checkpointing, set `tunnel_status="0"` to disable it.

---

## Parser / structured output errors

### `ValidationError` or garbled responses

The model's output did not match the expected schema.

**Try `parser_raw=True` to see the raw response:**
```python
card.parser_raw = True
# After running, trial["pred_resp"]["_raw"] contains the full AIMessage
```

**Try `json_mode` or `function_calling` if `json_schema` fails:**
```python
card.parser_config = {"method": "function_calling"}
# or
card.parser_config = {"method": "json_mode"}
```

**Use temperature=0 for more reliable structured output:**
```python
card.parameters = {"temperature": 0}
```

---

## Task JSON errors

### `KeyError` on `contexts_id`

The `contexts_id` values in your task JSON must match the keys in `items` exactly.

```json
// WRONG — mismatch
"contexts_id": ["E", "T"],
"items": { "encode": [...], "test": [...] }

// CORRECT
"contexts_id": ["encode", "test"],
"items": { "encode": [...], "test": [...] }
```

### `ValueError: trcode must be unique`

Each trial needs a unique `trcode` across the entire task (not just within a context).

---

## Memory / context errors

### Responses seem unaware of previous trials

Check that `memory` is set to `"Convo"` for multi-phase tasks:
```python
card.memory = "Convo"   # not "0" or "1"
```

### Context window overflow in long Convo runs

Use `memory_k` to limit conversation length:
```python
card.memory_k = 20   # keep only the last 20 messages
```

Or use `summary_k` to compress old messages:
```python
card.summary_k = 30   # summarize the oldest 30 messages when overflow occurs
```

---

## Ollama errors

### `ConnectError` / `Connection refused`

Ollama is not running. Start it with:
```bash
ollama serve
```

For a remote Ollama server, pass the base URL:
```python
card.parameters = {"base_url": "http://my-server:11434"}
```

### Model not found

Pull the model first:
```bash
ollama pull llama3.1:8b
```

Then confirm it's available:
```bash
ollama list
```

---

## Slow runs / timeouts

### HPC wall-time exceeded

Enable checkpointing so the run resumes where it left off:
```python
card.tunnel_status = "1"
card.tunnel_k      = -1   # checkpoint after every participant
```

Re-submit the same script — completed participants are automatically skipped.

### Mock model for fast testing

Use the built-in mock provider to test your pipeline without API calls:
```python
card = ExpCardInit(
    model  = "mock-chat-model",
    family = "mock-llm",
    ...
)
```

---

## Import errors

### `ModuleNotFoundError: No module named 'psychscanner'`

Activate the conda environment before running:
```bash
conda activate psyscan
```

Or reinstall:
```bash
cd /path/to/psychscanner
pip install -e .
```

---

## Getting help

- Open an issue at [github.com/saurabhr/psychscanner/issues](https://github.com/saurabhr/psychscanner/issues)
- Check the [Configuration Reference](configuration.md) for valid parameter values
- Run notebooks in `examples/` to see working code
