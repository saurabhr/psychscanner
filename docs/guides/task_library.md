# Task Library

`psychscanner.task_library()` fetches a task card JSON file by name, without
you having to hardcode a path. `psychscanner.list_task_library()` lists every
name currently available.

It is **not** a bundled registry — nothing ships inside the installed
package. Both functions search plain directories on disk at call time, so any
task card file placed in one of those directories becomes fetchable by its
filename immediately, with no registration step and no code change.

**The pattern every example in this repo actually uses** — pass `dirs=`
explicitly, since it's the only search tier that doesn't depend on which
directory your shell happened to be in when you called it:

```python
from psychscanner import task_library, list_task_library

list_task_library(dirs="examples/tasks")
# ['bfi44', 'example_survey', 'pal50', 'rm_singleturn_demo', 'vviq16', ...]

task = task_library("rm_singleturn_demo", dirs="examples/tasks")           # -> dict (parsed JSON)
path = task_library("rm_singleturn_demo", format="path", dirs="examples/tasks")  # -> Path

card = ExpCardInit(task_file=path, ...)
```

## Where it looks

In order, first match wins:

1. `dirs=` passed to the call — a single directory or a list of them. For a
   task card living anywhere on disk, including outside this repo entirely.
   **This is the reliable option** — the two below depend on your current
   working directory at call time, which varies by how you launched Python
   (JupyterLab from the repo root vs. a notebook's own directory vs. an IDE's
   Jupyter extension all set it differently).
2. Each directory in the `PSYCHSCANNER_TASK_LIBRARY_DIRS` environment
   variable, if set (`os.pathsep`-separated — `:` on macOS/Linux, `;` on
   Windows). Set this once (e.g. in `.env`) if you don't want to pass `dirs=`
   on every call.
3. `./demonstrations` (relative to the current working directory) — a
   generic default for a project-local "shared task cards" folder. This repo
   doesn't populate one (see below — contributed cards go straight into
   `examples/tasks/`), but the check is harmless if the directory doesn't
   exist; it's there for your own project's use.
4. `./tasks` (relative to the current working directory) — a project-local
   "bundled task cards" convention. In this repo, that's
   [`examples/tasks/`](https://github.com/saurabhr/psychscanner/tree/main/examples/tasks) —
   which only resolves via this tier if your cwd is `examples/` (e.g.
   `jupyter lab examples/`). From anywhere else, pass `dirs="examples/tasks"`
   as shown above.

If the same task name turns up in more than one search directory, the first
(by the order above) wins and a `UserWarning` names which directory was
shadowed — so a silent collision doesn't go unnoticed.

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

A name found in more than one search directory doesn't error — it warns:

```pycon
>>> task_library("my_task")  # exists in both dirs= and PSYCHSCANNER_TASK_LIBRARY_DIRS
UserWarning: Task 'my_task' found in more than one search directory. Using
/path/from/dirs/my_task.json — shadowed: /path/from/env/my_task.json.
{'taskname': 'my_task', ...}
```

## Fetching the vetted card index (`download_lib`)

The recipes above assume you already have a directory of cards on disk (a
checkout of this repo, or your own project's `tasks/`). For the separate,
versioned index of vetted cards —
[psyscan-library](https://github.com/saurabhr/psyscan-library) — `download_lib()`
clones/updates a checkout for you and returns paths ready to pass to
`dirs=`:

```python
from psychscanner import download_lib, task_library

paths = download_lib()  # library="psychscanner", kind="both"
card = task_library("example_survey", dirs=paths["tasks"])
```

`download_lib()` checks that the `library=` you asked for matches the
package actually installed (`psychscanner` vs. `psychscanner-primal` — cards
aren't portable between them) and raises rather than handing back cards that
won't run. See psyscan-library's
[`download_lib()` docs](https://psyscan-library.readthedocs.io/en/latest/download_lib/)
for the full parameter reference.

## See also

- [Cognitive Tasks](cognitive_tasks.md) / [Survey Tasks](survey_tasks.md) — writing a task card
- [PsychScanner Workflow](psychscanner_workflow.md) — the experiment card
  (`ExpCard`) that a fetched task card gets wrapped into
- [Cognitive Tasks with SweetPea](sweetpea_task_cards.md#3-convert-the-sampled-sequence-into-items) —
  generate a task card's `items` sequence, then save it here for lookup
  by name
- [Running a Survey with Persona Levels](../examples/multi_persona.md) —
  a worked example that fetches a task card this way before crossing it
  with personas
- [`examples/tasks/README.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/tasks/README.md)
