# Task Library

`psychscanner.task_library()` fetches a task card JSON file by name, without
you having to hardcode a path. `psychscanner.list_task_library()` lists every
name currently available.

It is **not** a bundled registry — nothing ships inside the installed
package. Both functions search plain directories on disk at call time, so any
task card file placed in one of those directories becomes fetchable by its
filename immediately, with no registration step and no code change.

```python
from psychscanner import task_library, list_task_library

list_task_library()
# ['bfi44', 'example_survey', 'pal50', 'rm_singleturn_demo', 'vviq16', ...]

task = task_library("rm_singleturn_demo")           # -> dict (parsed JSON)
path = task_library("rm_singleturn_demo", format="path")  # -> Path

card = ExpCardInit(task_file=path, ...)
```

## Where it looks

In order, first match wins:

1. `dirs=` passed to the call — a single directory or a list of them. For a
   task card living anywhere on disk, including outside this repo entirely.
2. Each directory in the `PSYCHSCANNER_TASK_LIBRARY_DIRS` environment
   variable, if set (`os.pathsep`-separated — `:` on macOS/Linux, `;` on
   Windows).
3. `./demonstrations` (relative to the current working directory) — a
   generic default for a project-local "shared task cards" folder. This repo
   doesn't populate one (see below — contributed cards go straight into
   `examples/tasks/`), but the check is harmless if the directory doesn't
   exist; it's there for your own project's use.
4. `./tasks` (relative to the current working directory) — the bundled
   tutorial task cards in [`examples/tasks/`](https://github.com/saurabhr/psychscanner/tree/main/examples/tasks).

Because 3 and 4 are resolved relative to the current working directory, this
works the same way the tutorials themselves already expect task files to be
found — run from inside `examples/` (as `jupyter lab examples/` does) and
`./tasks` resolves to `examples/tasks/` with no configuration.

## Fetching from your own directory

Point `dirs=` at wherever your task cards actually live — no need to copy
them into this repo first:

```python
task = task_library("my_study_task", dirs="/path/to/my/tasks")

# or check several places, in priority order
task = task_library("my_study_task", dirs=["./my_tasks", "/shared/lab_tasks"])
```

Prefer not to pass `dirs=` on every call? Set
`PSYCHSCANNER_TASK_LIBRARY_DIRS` once instead (e.g. in your shell profile or
`.env`) and every `task_library()` / `list_task_library()` call in that
session will search it automatically.

## Contributing a task card

[`examples/tasks/`](https://github.com/saurabhr/psychscanner/tree/main/examples/tasks)
holds both the bundled tutorial task cards and anything contributed by
others — there's no separate "shared cards" folder to keep in sync. Getting
one in is the standard GitHub fork-and-PR flow — nothing psychscanner-specific:

1. **Fork** [saurabhr/psychscanner](https://github.com/saurabhr/psychscanner)
   on GitHub, then clone your fork and branch:

   ```bash
   git clone https://github.com/<your-username>/psychscanner.git
   cd psychscanner
   git checkout -b add-<your_task_name>-task
   ```

2. **Write your task card** as JSON (see the [Cognitive Tasks](cognitive_tasks.md)
   and [Survey Tasks](survey_tasks.md) guides for the structure, or copy one
   of the files in [`examples/tasks/`](https://github.com/saurabhr/psychscanner/tree/main/examples/tasks)
   as a starting point), and **save it at the exact path**
   `examples/tasks/<your_task_name>.json` — that path is what makes
   it discoverable; there's no separate registration step.

3. **Commit, push, and open the PR:**

   ```bash
   git add examples/tasks/<your_task_name>.json
   git commit -m "Add <your_task_name> task card"
   git push -u origin add-<your_task_name>-task
   gh pr create --title "Add <your_task_name> task card" --fill
   # or open the PR from the compare-changes page GitHub links to after the push
   ```

   See [CONTRIBUTING.md](https://github.com/saurabhr/psychscanner/blob/main/CONTRIBUTING.md)
   for the general setup/test steps expected before a PR.

Once merged, anyone with the repo checked out can fetch it immediately —
`task_library("your_task_name")` — with no index file to update and no name
to register anywhere: `list_task_library()` discovers it by scanning the
directory, so the moment the file exists on `main`, it's part of the library.

## Errors

A missing name raises `FileNotFoundError` listing every directory that was
searched, so you can see exactly why the lookup failed:

```pycon
>>> task_library("does_not_exist")
FileNotFoundError: No task named 'does_not_exist' found. Searched:
.../examples/demonstrations, .../examples/tasks. Pass dirs=<your directory>
to search somewhere else, set PSYCHSCANNER_TASK_LIBRARY_DIRS, or place
does_not_exist.json in one of the directories above.
```

## See also

- [Cognitive Tasks](cognitive_tasks.md) / [Survey Tasks](survey_tasks.md) — writing a task card
- [`examples/tasks/README.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/tasks/README.md)
