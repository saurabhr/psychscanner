# User Guides

Comprehensive guides for using PsychScanner.

## Getting Started

- [Survey Tasks](survey_tasks.md) - Creating and running surveys
- [Cognitive Tasks](cognitive_tasks.md) - Implementing cognitive experiments
- [Custom Parsers](custom_parsers.md) - Building response parsers
- [Memory Types](memory_types.md) - Understanding memory management
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
