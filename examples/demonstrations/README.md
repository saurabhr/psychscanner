# Demonstrations

Shared/contributed task cards, discoverable by name via `psychscanner.task_library()`.

This is separate from [`../tasks/`](../tasks/), which holds the task cards the
tutorials themselves depend on. Files here are for sharing — drop in a task
card and anyone with this repo checked out can fetch it by filename, no code
change required:

```python
from psychscanner import task_library, list_task_library

list_task_library()                      # every task name found here and in ../tasks/
task = task_library("your_task_name")     # -> dict (parsed JSON)
path = task_library("your_task_name", format="path")  # -> Path, for card.task_file

# task card living somewhere else entirely (not this repo) -- checked first
task_library("your_task_name", dirs="/path/to/your/own/tasks")
```

## Contributing a task card

1. Drop `<your_task_name>.json` in this directory (see [`../tasks/README.md`](../tasks/README.md) for the task JSON structure).
2. That's it — `task_library("your_task_name")` and `list_task_library()` pick it up immediately, no registration step.

## How lookup works

`task_library()` searches, in order: the `dirs=` argument if you passed one
(a single directory or a list), then any directories listed in the
`PSYCHSCANNER_TASK_LIBRARY_DIRS` environment variable (`os.pathsep`-separated),
then `./demonstrations`, then `./tasks` — the last two relative to the current
working directory. The first `<taskname>.json` found wins.
