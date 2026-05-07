# Changelog

All notable changes to PsychScanner are documented here.

---

## 0.2.0 (2025)

### New features

- **`FeedbackBase` API** — subclass `FeedbackBase` and implement `on_response(trial, response)` to inject trial-level feedback into the conversation. Replaces the legacy `generate_feedback` interface. `on_response` receives the raw trial dict and a pre-parsed response dict; return a JSON string to inject or `None` to skip.

- **Session checkpointing** — `SessionTunnel` writes `BEGIN`, `SCAN`, and `END` log entries to a JSON file alongside experiment data. Set `tunnel_status="1"` on `ExpCardInit` to enable. Re-running the same script after an interruption resumes from the last completed participant.

- **`save_expcard` / `load_expcard`** — serialize and reload a complete experiment configuration, including inline task JSON and persona JSON, to a portable file. Designed for reproducibility and sharing across machines.

- **`get_task_template()`** — returns an empty task JSON skeleton for quick scaffolding.

- **Mock LLM provider** — use `model="mock-chat-model"` and `family="mock-llm"` for zero-API-key testing of pipelines.

- **`parser_raw` flag** — set `parser_raw=True` to store the raw `AIMessage` alongside the parsed response dict in `trial["pred_resp"]["_raw"]`.

- **`summary_k` context compression** — when context overflows in `Convo` mode, automatically summarize the oldest `summary_k` messages into a single summary message.

- **Per-trial parser dispatch** — pass a callable to `parser` that takes a `trcode` string and returns the appropriate parser class. Enables multi-phase tasks with different output schemas per phase.

### Provider support additions

- Google / Gemini (`google`, `gemini`, `google-genai` family aliases)
- Together.ai (`together`)
- Fireworks.ai (`fireworks`)
- Cohere (`cohere`)
- Azure OpenAI (`azure`)

### Documentation

- New MkDocs Material documentation site at `https://saurabhr.github.io/psychscanner/`
- Full configuration reference, API reference, and user guides
- Example notebooks updated to use current API

### Breaking changes

- `FeedbackBase.generate_feedback(trdata, pred_dict, input_dict, parser_status, trial_item_collector)` is removed. Replace with `on_response(trial, response)`.
- `memory_type` is not a valid field; use `memory` with values `"SingleTurn"` or `"Convo"`.
- `system_prompt`, `task_prompt`, and `response_format` are not fields on `ExpCardInit`. Task content is set via `task_file`; personas via `persona_files`.

---

## 0.1.0 (2024)

### Initial release

- `ExpCardInit` / `ExpCard` — experiment configuration via Pydantic model
- `ScannerModel` — runs experiment trials against LangChain chat models
- `SingleTurn` and `Convo` memory modes
- Bundled parsers: `DefaultLiteralAgree`, `DefaultLiteralVivid15`, `DefaultLiteralVivid010`, `DefaultResponseRating`, `Response_part_1_rm`, `Response_part_2_rm`
- Built-in VVIQ-16 imagery questionnaire as the default task
- `to_csv` / `concat_csv` export utilities
- Support for OpenAI, Anthropic, Groq, Mistral, HuggingFace, and Ollama providers
