"""One-call convenience wrapper: task name in, scanner results out.

Chains the steps every quickstart already does by hand::

    from psychscanner import task_library, ExpCardInit, ExpCard, ScannerModel
    task_file = task_library("example_survey", format="path", dirs="tasks/psychscanner")
    card = ExpCardInit(task_file=task_file, model="mock-chat-model", family="mock-llm",
                        cogtype="no", nsim=1)
    results = ScannerModel(expcard=ExpCard(card)).run()

into one call::

    from psychscanner import run_card
    results = run_card("example_survey", dirs="tasks/psychscanner")

Reach for the longer form instead when you need the `ExpCard`/`ScannerModel`
object itself (e.g. to pass to `to_csv()` afterward), not just the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .scanner_models.scanner_model import ScannerModel
from .staging.scanner_cards import ExpCard, ExpCardInit
from .task_library import DirsArg, task_library

__all__ = ["run_card"]


def run_card(
    taskname: str,
    dirs: DirsArg = None,
    *,
    model: str = "mock-chat-model",
    family: str = "mock-llm",
    memory: str = "SingleTurn",
    cogtype: str = "no",
    nsim: int | None = 1,
    projectname: str | None = None,
    proj_dir: str | Path | None = None,
    **exp_kwargs: Any,
) -> list:
    """Look up a task card by name and run it end to end.

    Equivalent to `task_library(taskname, format="path", dirs=dirs)` fed into
    `ExpCardInit` -> `ExpCard` -> `ScannerModel(...).run()`.

    Parameters
    ----------
    taskname : str
        Task card name, as passed to `task_library()`.
    dirs : str | os.PathLike | list[str | os.PathLike] | None
        Forwarded to `task_library()` -- extra directories to search first.
    model, family, memory, cogtype, nsim, projectname, proj_dir
        Forwarded to `ExpCardInit`; defaults match its own (mock LLM, no API
        key, no network). `projectname` defaults to `taskname` if not given.
    **exp_kwargs
        Any other `ExpCardInit` field (`parser`, `parameters`, `feedback_fn`,
        `next_trial_fn`, `login_env`, ...), forwarded unchanged.

    Returns
    -------
    list
        `ScannerModel.run()`'s result: one trial-result list per simulated
        participant.
    """
    task_file = task_library(taskname, format="path", dirs=dirs)
    kwargs: dict[str, Any] = dict(
        task_file=task_file,
        model=model,
        family=family,
        memory=memory,
        cogtype=cogtype,
        nsim=nsim,
        projectname=projectname or taskname,
        **exp_kwargs,
    )
    if proj_dir is not None:
        # ExpCardInit's own default (~/psychscanner) only applies when the
        # field is omitted entirely -- passing proj_dir=None explicitly would
        # override it with None and break ExpCard's proj_dir / ... path join.
        kwargs["proj_dir"] = proj_dir
    card = ExpCardInit(**kwargs)
    return ScannerModel(expcard=ExpCard(card)).run()
