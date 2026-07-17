# Graph Report - src/psychscanner  (2026-07-16)

## Corpus Check
- Corpus is ~29,005 words - fits in a single context window. You may not need a graph.

## Summary
- 457 nodes · 917 edges · 21 communities (20 shown, 1 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 153 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Agent Framework
- Base Agent & Mock LLM
- Parser Registry & Settings
- General Response Parsers
- Session Tunnel Logging
- Chat & Multimodal Prompts
- Task Response Parsers
- Feedback & Scanner Model
- Psyscan I/O (Polars)
- Legacy Task Prompts
- Task Prompts V2
- Media Externalization
- Staging Utils
- Dataset Loading
- Task Runner
- CLI Entry Point
- Legacy Task Runner

## God Nodes (most connected - your core abstractions)
1. `TwoResponses` - 33 edges
2. `SessionTunnel` - 23 edges
3. `CustomAgent` - 22 edges
4. `Source` - 20 edges
5. `DefaultRmChoiceConf16` - 20 edges
6. `ResponseRmScSt` - 20 edges
7. `Task_1_ResponseRate` - 20 edges
8. `Task_2_ResponseRate` - 20 edges
9. `Task_3_ResponseRate` - 20 edges
10. `DefaultRMEncodingPhase` - 19 edges

## Surprising Connections (you probably didn't know these)
- `_source_to_df()` --indirect_call--> `ExpCard`  [INFERRED]
  scanner_models/psyscan_io.py → staging/scanner_cards.py
- `ScannerModel` --uses--> `TaskRunner`  [INFERRED]
  scanner_models/scanner_model.py → task_runner/task_runner.py
- `ExpCard` --uses--> `SessionTunnel`  [INFERRED]
  staging/scanner_cards.py → session_tunnel/session_tunnel.py
- `ExpCardInit` --uses--> `SessionTunnel`  [INFERRED]
  staging/scanner_cards.py → session_tunnel/session_tunnel.py
- `Settings` --uses--> `SessionTunnel`  [INFERRED]
  staging/scanner_cards.py → session_tunnel/session_tunnel.py

## Import Cycles
- 1-file cycle: `__init__.py -> __init__.py`
- 1-file cycle: `datasets/__init__.py -> datasets/__init__.py`
- 1-file cycle: `datasets/prompts/__init__.py -> datasets/prompts/__init__.py`
- 1-file cycle: `staging/__init__.py -> staging/__init__.py`

## Communities (21 total, 1 thin omitted)

### Community 0 - "Agent Framework"
Cohesion: 0.06
Nodes (47): CustomAgent, Any, Plug a researcher-supplied LLM or VLM into the simulation loop.  ``TaskRunner``, The contract ``TaskRunner`` expects from any scanning agent., Adapt a plain callable to the :class:`ScanningAgent` contract.      ``call_fn``, ScanningAgent, make_planner_executor_agent(), _PlannerExecutorState (+39 more)

### Community 1 - "Base Agent & Mock LLM"
Cohesion: 0.05
Nodes (38): BaseChatModel, BaseException, CallbackManagerForLLMRun, ChatGenerationChunk, ChatResult, AgentInitializer, chat_template_parser_chain(), Any (+30 more)

### Community 2 - "Parser Registry & Settings"
Cohesion: 0.06
Nodes (36): BaseSettings, _collect(), get_parser(), list_parsers(), BaseModel, Top-level namespace for all bundled parser classes.  Provides a single import pa, Return a sorted list of all bundled parser class names., Look up a parser class by name.      Raises     ------     KeyError         If ` (+28 more)

### Community 3 - "General Response Parsers"
Cohesion: 0.19
Nodes (41): Backward-compatibility shim for ``parser_extra.py``.  The parser classes that us, DefaultLiteralAgree, DefaultParser, DefaultRatingParser, DefaultResponseRating, DefaultResponseRatingConvo, DefaultWordCaseNonWord, DefaultWordCaseNonWordConf16 (+33 more)

### Community 4 - "Session Tunnel Logging"
Cohesion: 0.08
Nodes (20): Series, A tool to bridge natural psychology with the artificial., DataFrame, Path, Patch a log record with serialized data.          Parameters:         ----------, A class to manage session tunnels for logging and tracking.      This class hand, Create a log file for the session tunnel.          Parameters:         ---------, Log a checkpoint for a scan session.          Parameters:         ---------- (+12 more)

### Community 5 - "Chat & Multimodal Prompts"
Cohesion: 0.09
Nodes (28): This module defines chat prompt templates for use in the psychscanner project., A tool to bridge natural psychology with the artificial., audio_block(), _block(), file_block(), image_block(), _is_url(), Path (+20 more)

### Community 6 - "Task Response Parsers"
Cohesion: 0.16
Nodes (33): TwoResponses, Backward-compatibility shim for ``parser.py``.  The parser classes that used to, AllResponseRMEI, AllResponseRMEIN, AllResponseRMIE, AllResponseRMIEN, Confidence16, DefaultLiteralVivid010 (+25 more)

### Community 7 - "Feedback & Scanner Model"
Cohesion: 0.13
Nodes (21): ABC, FeedbackBase, Abstract base class for trial-level feedback handlers.      Subclass this, overr, Generate feedback for a completed trial.          Called by ``TaskRunner`` after, Merge previous-trial feedback into the next trial's ``input_dict``.          Ove, A tool to bridge natural psychology witth  the artificial., Any, Path (+13 more)

### Community 8 - "Psyscan I/O (Polars)"
Cohesion: 0.15
Nodes (26): A tool to bridge natural psychology with the artificial., _auto_path(), concat_csv(), _expcard_meta(), _parse_pred_resp(), _parse_tunnel_id(), Any, DataFrame (+18 more)

### Community 9 - "Legacy Task Prompts"
Cohesion: 0.15
Nodes (20): all_system_msg_prompts(), create_symsg_data_prompt(), gen_sc_trial_prompt(), gen_symsg_promptdata(), gen_trial_prompt(), gen_trial_promptdata(), get_human_feedback_prompt(), get_sc_task_with_hmsg() (+12 more)

### Community 10 - "Task Prompts V2"
Cohesion: 0.18
Nodes (16): all_system_msg_prompts(), create_symsg_data_prompt(), gen_sc_trial_prompt(), gen_symsg_promptdata(), gen_trial_prompt(), gen_trial_promptdata(), get_sc_task_with_hmsg(), get_surveytrials_with_human_msg() (+8 more)

### Community 11 - "Media Externalization"
Cohesion: 0.36
Nodes (8): externalize_media(), _guess_ext(), Any, Path, Externalize inline base64 media from trial records before they're persisted.  Ev, Return a deep copy of *trials* with inline base64 media externalized.      Walks, _rewrite(), _write_block()

### Community 12 - "Staging Utils"
Cohesion: 0.25
Nodes (6): get_model_card(), Any, Recursively converts an object with attributes (or a nested structure) into a di, modelname (str): ollama modelnames, structured_to_dict(), ticTok

### Community 13 - "Dataset Loading"
Cohesion: 0.36
Nodes (6): A tool to bridge natural psychology with the artificial., get_persona_data(), get_task_data(), open_json(), Retrieve task data based on the provided experiment card.      Parameters:     -, Load and return JSON data from the specified file path.      Parameters:     ---

### Community 14 - "Task Runner"
Cohesion: 0.32
Nodes (4): _parse_response(), Convert an AIMessage into a plain dict for feedback handlers.      For structure, Execute all task trials and return a list of per-trial result dicts.          Pa, TaskRunner

### Community 15 - "CLI Entry Point"
Cohesion: 0.33
Nodes (5): cli(), Path, Main CLI for psychscanner., Repeat the input.      A tool to bridge natural psychology with the artificial., psychscanner as a module entry point.  This allows psychscanner to be executable

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SessionTunnel` connect `Session Tunnel Logging` to `Parser Registry & Settings`, `Feedback & Scanner Model`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `CustomAgent` connect `Agent Framework` to `Feedback & Scanner Model`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `TwoResponses` (e.g. with `DefaultRmChoiceConf16` and `DefaultRMEncodingPhase`) actually correct?**
  _`TwoResponses` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `SessionTunnel` (e.g. with `ExpCard` and `.__init__()`) actually correct?**
  _`SessionTunnel` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `CustomAgent` (e.g. with `_PlannerExecutorState` and `_LATSNode`) actually correct?**
  _`CustomAgent` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Source` (e.g. with `DefaultLiteralAgree` and `DefaultParser`) actually correct?**
  _`Source` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `A tool to bridge natural psychology witth  the artificial.`, `psychscanner as a module entry point.  This allows psychscanner to be executable`, `Plug a researcher-supplied LLM or VLM into the simulation loop.  ``TaskRunner``` to the rest of the system?**
  _147 weakly-connected nodes found - possible documentation gaps or missing edges._