# API Reference

Complete API documentation for PsychScanner.

## Core Classes

- [ExpCard](expcard.md) - Experiment configuration
- [ScannerModel](scanner_model.md) - Main scanning engine
- [SessionTunnel](session_tunnel.md) - Session management
- [Parsers](parsers.md) - Response parsers

## Functions

- [run_card](run_card.md) - Task name to results in one call

Also exported from the top-level `psychscanner` package (no dedicated page —
documented inline where they're used):

- `save_expcard` / `load_expcard` — see [ExpCard § Saving and loading](expcard.md#saving-and-loading-experiment-cards)
- `FeedbackBase` / `NextTrialBase` — see [Configuration § Feedback](../configuration.md#feedback) and the [Conditional Next Trial example](../examples/conditional_next_trial.md)
- `task_library` / `list_task_library` / `download_lib` — see the [Task Library guide](../guides/task_library.md)

## Quick Links

- [Installation](../installation.md)
- [Quick Start](../usage.md)
- [Examples](../examples/index.md)
