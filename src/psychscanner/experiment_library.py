"""Name-based lookup for experiment card (.xcard.psyscan) files, shared
across a project.

Mirrors `psychscanner.task_library` exactly, but for experiment cards
(`ExpCard`/`ExpCardInit` configs saved via `save_experiment_card`) instead
of task cards. Not a bundled registry — this searches plain directories on
disk at call time, so any `.xcard.psyscan` file dropped into a search
directory becomes available by its filename, with no code change.

Quick recipe::

    card = experiment_library("my_experiment", dirs="experiments")

Search order (first match wins):
  1. `dirs`, if passed to the call — one directory or a list of them. The
     reliable option, since 3/4 below depend on your shell's current
     directory at call time.
  2. Each directory in the `PSYCHSCANNER_EXPERIMENT_LIBRARY_DIRS`
     environment variable, if set (`os.pathsep`-separated).
  3. `./experiments` (relative to the current working directory) — the
     project-local "shared experiment cards" convention.
  4. `./demonstrations` (relative to the current working directory) —
     same shared-cards convention `task_library` also checks, for a
     project that keeps task and experiment cards in one place.

If the same experiment name is found in more than one search directory,
the first one wins and a `UserWarning` names the directory that was
shadowed — same behavior as `task_library`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from .staging.scanner_cards import EXPERIMENT_CARD_EXT, load_experiment_card

if TYPE_CHECKING:
    from .staging.scanner_cards import ExpCardInit

__all__ = ["list_experiment_library", "experiment_library"]

DirsArg = Union[str, "os.PathLike[str]", list, None]


def _search_dirs(dirs: DirsArg = None) -> list[Path]:
    result = []
    if dirs is not None:
        if isinstance(dirs, (str, os.PathLike)):
            dirs = [dirs]
        result.extend(Path(d) for d in dirs)

    env = os.getenv("PSYCHSCANNER_EXPERIMENT_LIBRARY_DIRS")
    if env:
        result.extend(Path(p) for p in env.split(os.pathsep) if p)

    cwd = Path.cwd()
    result.extend([cwd / "experiments", cwd / "demonstrations"])
    return result


def _warn_if_shadowed(name: str, matches: list[Path]) -> None:
    if len(matches) > 1:
        warnings.warn(
            f"Experiment {name!r} found in more than one search directory. "
            f"Using {matches[0]} — shadowed: "
            f"{', '.join(str(m) for m in matches[1:])}.",
            stacklevel=3,
        )


def experiment_library(
    name: str,
    format: str = "expcard",
    dirs: DirsArg = None,
    **load_kwargs: Any,
) -> "ExpCardInit | dict | Path":
    """Look up an experiment card by name across the experiment-library
    search directories.

    Parameters
    ----------
    name : str
        Base filename of the experiment card, without extension (e.g.
        `"my_experiment"` for a file named `my_experiment.xcard.psyscan`).
    format : str
        `"expcard"` (default) — reconstruct and return an `ExpCardInit`,
        via `load_experiment_card`.
        `"json"` — return the raw saved dict, without reconstruction.
        `"path"` — return the resolved `Path` without reading it.
    dirs : str | os.PathLike | list[str | os.PathLike] | None
        One directory, or a list of directories, to search first.
    **load_kwargs
        Forwarded to `load_experiment_card` when `format="expcard"` — e.g.
        `proj_dir=`, or `parser=`/`feedback_fn=`/`next_trial_fn=`/`tools=`
        overrides for anything that couldn't be re-imported.

    Returns
    -------
    ExpCardInit | dict | Path

    Raises
    ------
    FileNotFoundError
        If no `<name>{EXPERIMENT_CARD_EXT}` is found in any search
        directory. The error lists every directory that was searched.
    ValueError
        If `format` is not `"expcard"`, `"json"`, or `"path"`.

    Warns
    -----
    UserWarning
        If `<name>{EXPERIMENT_CARD_EXT}` exists in more than one search
        directory — the first (by search order) is used silently
        otherwise.
    """
    if format not in ("expcard", "json", "path"):
        raise ValueError(f"format must be 'expcard', 'json', or 'path', got {format!r}")

    search_dirs = _search_dirs(dirs)
    filename = f"{name}{EXPERIMENT_CARD_EXT}"
    matches = [d / filename for d in search_dirs if (d / filename).is_file()]

    if matches:
        _warn_if_shadowed(name, matches)
        candidate = matches[0]
        if format == "path":
            return candidate
        if format == "json":
            import json

            return json.loads(candidate.read_text(encoding="utf-8"))
        return load_experiment_card(candidate, **load_kwargs)

    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"No experiment named {name!r} found. Searched: {searched}. "
        "Pass dirs=<your directory> to search somewhere else, set "
        f"PSYCHSCANNER_EXPERIMENT_LIBRARY_DIRS, or place {filename} in one "
        "of the directories above."
    )


def list_experiment_library(dirs: DirsArg = None) -> list[str]:
    """Return the sorted, de-duplicated names of every experiment card
    discoverable across the experiment-library search directories (see
    `experiment_library`).

    Parameters
    ----------
    dirs : str | os.PathLike | list[str | os.PathLike] | None
        Extra directory (or directories) to include in the search, same as
        `experiment_library`'s `dirs` argument.

    Warns
    -----
    UserWarning
        For every experiment name found in more than one search directory.
    """
    sources: dict[str, list[Path]] = {}
    for d in _search_dirs(dirs):
        if d.is_dir():
            for p in d.glob(f"*{EXPERIMENT_CARD_EXT}"):
                name = p.name.removesuffix(EXPERIMENT_CARD_EXT)
                sources.setdefault(name, []).append(p)

    for name, matches in sources.items():
        if len(matches) > 1:
            _warn_if_shadowed(name, matches)

    return sorted(sources)
