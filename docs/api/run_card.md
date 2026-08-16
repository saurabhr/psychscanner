# run_card

One call from a task name to results — `task_library()` + `ExpCardInit` +
`ExpCard` + `ScannerModel(...).run()`, chained.

```python
from psychscanner import run_card

results = run_card("example_survey", dirs="examples/tasks")
```

## Signature

```python
run_card(
    taskname: str,
    dirs: str | os.PathLike | list | None = None,
    *,
    model: str = "mock-chat-model",
    family: str = "mock-llm",
    memory: str = "SingleTurn",
    cogtype: str = "no",
    nsim: int | None = 1,
    projectname: str | None = None,
    proj_dir: str | Path | None = None,
    **exp_kwargs,
) -> list
```

| Parameter | Description |
|---|---|
| `taskname` | Task card name, as passed to [`task_library()`](../guides/task_library.md). |
| `dirs` | Forwarded to `task_library()` — extra directories to search first. |
| `model`, `family` | Forwarded to `ExpCardInit`. Default to the built-in mock LLM — no API key, no network. |
| `memory` | `"SingleTurn"` or `"Convo"`. Forwarded to `ExpCardInit`. |
| `cogtype`, `nsim` | Default to `cogtype="no"`, `nsim=1` — one simulated agent, no persona load. Pass `cogtype="custom"` and drop `nsim` to use persona files instead. |
| `projectname` | Defaults to `taskname` if not given. |
| `proj_dir` | Defaults to `ExpCardInit`'s own default (`~/psychscanner`) when omitted. Passing `proj_dir=None` explicitly is the same as omitting it. |
| `**exp_kwargs` | Any other [`ExpCardInit`](expcard.md) field — `parser`, `parameters`, `feedback`/`feedback_fn`, `next_trial`/`next_trial_fn`, `login_env`, ... — forwarded unchanged. |

## Returns

Whatever `ScannerModel.run()` returns: one trial-result list per simulated
participant. `run_card` does not expose the `ScannerModel` or `ExpCard`
object it builds internally — if you need those (e.g. to call
`to_csv(scanner, ...)` afterward), use the four-step form directly instead:

```python
from psychscanner import task_library, ExpCardInit, ExpCard, ScannerModel, to_csv

task_file = task_library("example_survey", format="path", dirs="examples/tasks")
card = ExpCardInit(task_file=task_file, model="mock-chat-model", family="mock-llm",
                    cogtype="no", nsim=1)
scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()
to_csv(scanner, path=card.card_in.proj_dir)
```

## See also

- [Task Library guide](../guides/task_library.md#running-a-fetched-card-in-one-call-run_card) — narrative walkthrough
- [ExpCard & ExpCardInit](expcard.md) — full field reference for `**exp_kwargs`
- [ScannerModel](scanner_model.md) — what `.run()` actually does
