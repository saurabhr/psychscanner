"""Name-based lookup for task card JSON files, shared across a project.

Not a bundled registry — this searches plain directories on disk at call time,
so any task card dropped into one of the search directories becomes available
by its filename, with no code change.

Quick recipe (what every example in this repo actually uses)::

    task_library("rm_singleturn_demo", dirs="examples/tasks")

Search order (first match wins):
  1. `dirs`, if passed to the call — one directory or a list of them, checked
     in the given order. For a task card living anywhere else on disk. This
     is the reliable way to call it, since 3/4 below depend on your shell's
     current directory at call time.
  2. Each directory in the `PSYCHSCANNER_TASK_LIBRARY_DIRS` environment
     variable, if set (`os.pathsep`-separated, e.g. `:` on macOS/Linux).
  3. `./demonstrations` (relative to the current working directory) — a
     project-local "shared task cards" convention, if your own project uses
     one.
  4. `./tasks` (relative to the current working directory) — a project-local
     "bundled task cards" convention.

If the same task name is found in more than one search directory, the first
one wins and a `UserWarning` names the directory that was shadowed.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Union

__all__ = ["list_task_library", "task_library"]

DirsArg = Union[str, "os.PathLike[str]", list, None]


def _search_dirs(dirs: DirsArg = None) -> list[Path]:
    result = []
    if dirs is not None:
        if isinstance(dirs, (str, os.PathLike)):
            dirs = [dirs]
        result.extend(Path(d) for d in dirs)

    env = os.getenv("PSYCHSCANNER_TASK_LIBRARY_DIRS")
    if env:
        result.extend(Path(p) for p in env.split(os.pathsep) if p)

    cwd = Path.cwd()
    result.extend([cwd / "demonstrations", cwd / "tasks"])
    return result


def _warn_if_shadowed(taskname: str, matches: list[Path]) -> None:
    if len(matches) > 1:
        warnings.warn(
            f"Task {taskname!r} found in more than one search directory. "
            f"Using {matches[0]} — shadowed: "
            f"{', '.join(str(m) for m in matches[1:])}.",
            stacklevel=3,
        )


def task_library(taskname: str, format: str = "json", dirs: DirsArg = None) -> dict | Path:
    """Look up a task card by name across the task-library search directories.

    Parameters
    ----------
    taskname : str
        Base filename of the task card, without extension (e.g.
        `"rm_singleturn_demo"` for a file named `rm_singleturn_demo.json`).
    format : str
        `"json"` (default) — parse the file and return a `dict`.
        `"path"` — return the resolved `Path` without reading it (what
        `ExpCardInit.task_file` expects).
    dirs : str | os.PathLike | list[str | os.PathLike] | None
        One directory, or a list of directories, to search first — for a
        task card living anywhere on disk, not just the default locations.

    Returns
    -------
    dict | Path

    Raises
    ------
    FileNotFoundError
        If no `<taskname>.json` is found in any search directory. The error
        lists every directory that was searched.
    ValueError
        If `format` is not `"json"` or `"path"`, or the matched file is not
        valid JSON.

    Warns
    -----
    UserWarning
        If `<taskname>.json` exists in more than one search directory — the
        first (by search order) is used silently otherwise.
    """
    if format not in ("json", "path"):
        raise ValueError(f"format must be 'json' or 'path', got {format!r}")

    search_dirs = _search_dirs(dirs)
    matches = [d / f"{taskname}.json" for d in search_dirs if (d / f"{taskname}.json").is_file()]

    if matches:
        _warn_if_shadowed(taskname, matches)
        candidate = matches[0]
        if format == "path":
            return candidate
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            msg = f"Task card {candidate} is not valid JSON: {e}"
            raise ValueError(msg) from e

    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"No task named {taskname!r} found. Searched: {searched}. "
        "Pass dirs=<your directory> to search somewhere else, set "
        f"PSYCHSCANNER_TASK_LIBRARY_DIRS, or place {taskname}.json in one "
        "of the directories above."
    )


def list_task_library(dirs: DirsArg = None) -> list[str]:
    """Return the sorted, de-duplicated names of every task card discoverable
    across the task-library search directories (see `task_library`).

    Parameters
    ----------
    dirs : str | os.PathLike | list[str | os.PathLike] | None
        Extra directory (or directories) to include in the search, same as
        `task_library`'s `dirs` argument.

    Warns
    -----
    UserWarning
        For every task name found in more than one search directory.
    """
    sources: dict[str, list[Path]] = {}
    for d in _search_dirs(dirs):
        if d.is_dir():
            for p in d.glob("*.json"):
                sources.setdefault(p.stem, []).append(p)

    for taskname, matches in sources.items():
        if len(matches) > 1:
            _warn_if_shadowed(taskname, matches)

    return sorted(sources)
