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

New task/experiment cards no longer go through a PR against
`examples/tasks/` in this repo — that path is now reserved for the bundled
tutorial cards only. Contribute to
[`psyscan-library`](https://github.com/saurabhr/psyscan-library) instead,
the public, versioned index of vetted cards for this package:

1. Add your card as a single file at `tasks/psychscanner/<name>.json` (or
   `.tcard.psyscan`) in that repo — see the [Cognitive Tasks](cognitive_tasks.md)
   and [Survey Tasks](survey_tasks.md) guides here for the card structure.
2. Validate it locally with `psychscanner` installed:

   ```bash
   python scripts/validate_contribution.py tasks/psychscanner/<name>.json
   ```

   This checks required fields, rejects duplicates, and — the hard
   requirement — runs the card end-to-end against the mock LLM.
3. Open a PR against `psyscan-library`. See that repo's
   [`CONTRIBUTING.md`](https://github.com/saurabhr/psyscan-library/blob/main/CONTRIBUTING.md)
   for the full process.

Once merged, fetch it with [`download_lib()`](#fetching-the-vetted-card-index-download_lib)
to get a local `tasks/` path, then pass that as `dirs=` to `task_library()`
as shown below.

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

## Running a fetched card in one call (`run_card`)

Everything above gets you a task card. Running it still means building an
`ExpCardInit`, wrapping it in `ExpCard`, and calling `ScannerModel(...).run()`
by hand — three more names to import for the common case. `run_card()`
chains all four steps (`task_library` included) into one call:

```python
from psychscanner import run_card

results = run_card("rm_singleturn_demo", dirs="examples/tasks")
```

Equivalent to:

```python
from psychscanner import task_library, ExpCardInit, ExpCard, ScannerModel

task_file = task_library("rm_singleturn_demo", format="path", dirs="examples/tasks")
card = ExpCardInit(task_file=task_file, model="mock-chat-model", family="mock-llm",
                    cogtype="no", nsim=1)
results = ScannerModel(expcard=ExpCard(card)).run()
```

`run_card`'s keyword defaults (`model`, `family`, `memory`, `cogtype`, `nsim`)
match `ExpCardInit`'s own defaults — override any of them, or pass through
anything else `ExpCardInit` accepts (`parser`, `parameters`, `feedback_fn`,
...) as extra keyword arguments:

```python
results = run_card(
    "vviq16",
    dirs=paths["tasks"],       # from download_lib()
    model="gpt-4o-mini",
    family="openai",
    nsim=10,
    parser="DefaultLiteralVivid15",
)
```

Reach for the longer, four-step form instead when you need the `ExpCard` or
`ScannerModel` object itself — e.g. to call `to_csv(scanner, ...)` after
`.run()`, or to inspect `card_in`/`data_root_dir` before running. See the
[API reference](../api/run_card.md) for the full signature.

## See also

- [Cognitive Tasks](cognitive_tasks.md) / [Survey Tasks](survey_tasks.md) — writing a task card
- [PsychScanner Workflow](psychscanner_workflow.md) — the experiment card
  (`ExpCard`) that a fetched task card gets wrapped into
- [Cognitive Tasks with SweetPea](sweetpea_task_cards.md#3-convert-the-sampled-sequence-into-items) —
  generate a task card's `items` sequence, then save it here for lookup
  by name
- [Running a Survey with Persona Levels](../examples/demonstration_suite/03_personality_survey.md) —
  a worked example that fetches a task card this way before crossing it
  with personas
- [`examples/tasks/README.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/tasks/README.md)
