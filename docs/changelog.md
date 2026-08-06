# Changelog

All notable changes to PsychScanner are documented here.

---

## Unreleased

### New features

- **Task library** — `psychscanner.task_library()` fetches a task card JSON file by name, searching `dirs=`, `PSYCHSCANNER_TASK_LIBRARY_DIRS`, `./demonstrations`, then `./tasks` (first match wins); `list_task_library()` lists every name discoverable across those directories. Lets contributors share a task card by dropping `<name>.json` into `examples/demonstrations/`, no registration step or code change required. See the [Task Library guide](guides/task_library.md).

---

## 0.3.0 (2026-07-16)

### New features

- **Multimodal stimuli** — `psychscanner.datasets.prompts.multimodal` provides `image_block`, `audio_block`, `file_block`, and `website_block` helpers that turn a local path, URL, or scraped webpage into a standard LangChain content block. `resolve_path_block` lets a JSON task card reference local media by plain `{"path": ...}` instead of a Python call. Trial stimuli accept a list of these blocks for mixed image/audio/text content. See [Cognitive Tasks § Visual Search and Attention](guides/cognitive_tasks.md#visual-search-and-attention).
- **Media externalization** — inline base64 media is written once to `<data_root>/media/<sha256>.<ext>` and replaced with a path reference before a `.psyscan` checkpoint is persisted, so repeated personas/`nsim` conditions reusing the same stimulus don't duplicate the blob (`psychscanner.scanner_models.media_store`).
- **Tool binding** — `card.tools` binds a list of LangChain tools to the model for the whole run; a trial's `"tools"` key selects a named subset (or `[]` to opt out) for that trial only. See [Cognitive Tasks § Tool Binding](guides/cognitive_tasks.md#tool-binding).
- **`CustomAgent` / `ScanningAgent` protocol** — `psychscanner.agents.CustomAgent` adapts any callable (a raw provider SDK call, a local model, a REST API) to the contract `TaskRunner`/`ScannerModel` expect, so a custom LLM or VLM can be dropped in via `ScannerModel.run(custom_agent=...)` without building a LangGraph graph. `ScanningAgent` is `@runtime_checkable`.
- **Five LangGraph agent architectures**, each adapted to the `ScanningAgent` contract via `CustomAgent` (`psychscanner.agents`):
  - `make_react_agent` — a real tool-calling loop via `langgraph.prebuilt`/`langchain.agents.create_agent`.
  - `make_planner_executor_agent` — modular Planner/Executor/Validator loop (arxiv:2310.00194).
  - `make_basic_reflection_agent`, `make_reflexion_agent`, `make_lats_agent` — Basic Reflection, Reflexion, and LATS (Monte Carlo tree search), after the LangChain reflection-agents blog post.
  - `make_supervisor_agent` — Jockey-style Supervisor/Planner/Worker routing generalized from video to arbitrary multimodal content blocks.

### Bug fixes

- Fixed a bug in `single_turn_convo.py` where the model's response was clobbering the `RemoveMessage` list produced by history trimming, making `memory_k` a no-op and causing unbounded conversation growth in `Convo` mode. Trim updates and the new response now ride in the same state update.

### Documentation

- Tutorial notebooks 10–17 cover `CustomAgent`, tool binding + multimodal stimuli, and each of the five new agent architectures.

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
