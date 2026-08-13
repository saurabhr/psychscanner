# User Guides

Comprehensive guides for using PsychScanner.

## Getting Started

- [Cognitive Tasks](cognitive_tasks.md) - The task JSON schema (task card), plus the Reality Monitoring paradigm and visual search examples
- [PsychScanner Workflow](psychscanner_workflow.md) - A slide-ready summary of the execution path (experiment card)
- [Survey Tasks](survey_tasks.md) - Creating and running surveys
- [Running a Survey with Persona Levels](../examples/multi_persona.md) - Simulating participants with distinct personas
- [Generating Task Cards with SweetPea](sweetpea_task_cards.md) - Factorial counterbalancing for a task's `items` sequence
- [Task Library](task_library.md) - Fetching task cards by name, and sharing your own
- [Memory Types](memory_types.md) - Understanding memory management
- [Custom Parsers](custom_parsers.md) - Building response parsers
- [Session Recovery](session_recovery.md) - Checkpoint and resume experiments

## Advanced Topics

- **Multi-agent experiments** — bring your own LLM/VLM via `CustomAgent`, or
  drop in one of the built-in LangGraph agent architectures (ReAct tool-calling
  loop, Planner/Executor/Validator, Basic Reflection/Reflexion/LATS, and a
  Jockey-style multimodal Supervisor/Planner/Worker) — see the
  [Extending psychscanner](../tutorials/index.md#extending-psychscanner) and
  [Tool binding, multimodal stimuli, and custom agent architectures](../tutorials/index.md#tool-binding-multimodal-stimuli-and-custom-agent-architectures)
  tutorial sections.

Coming soon:
- Real-time monitoring
- Custom task types
- Integration with analysis pipelines
