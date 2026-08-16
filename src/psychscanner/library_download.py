"""Fetch psyscan-library's task/experiment cards onto local disk.

`download_lib()` clones (or updates) a checkout of the `psyscan-library`
index repo and returns paths you can pass straight to `task_library()`/
`experiment_library()` as `dirs=`.

Quick recipe::

    from psychscanner import download_lib, task_library

    paths = download_lib()  # library="psychscanner" (this package), kind="both"
    card = task_library("example_survey", dirs=paths["tasks"])

Cards aren't portable between the `psychscanner` and `primal` distributions
(different card conventions, different compatibility floor) -- see
`psyscan-library`'s README. By default `download_lib()` refuses to fetch
cards for a distro other than the one actually installed; pass
`library="all"` to opt into fetching everything anyway (browsing, or CI that
covers both distros).
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
from pathlib import Path
from typing import Literal

__all__ = ["download_lib"]

LIBRARY_REPO_URL = "https://github.com/saurabhr/psyscan-library.git"
_CACHE_DIR = Path.home() / ".cache" / "psychscanner" / "psyscan-library"

Library = Literal["psychscanner", "primal", "all"]
Kind = Literal["tasks", "experiments", "both"]

_KIND_DIRS: dict[str, tuple[str, ...]] = {
    "tasks": ("tasks",),
    "experiments": ("experiments",),
    "both": ("tasks", "experiments"),
}
_DISTROS: tuple[str, ...] = ("psychscanner", "primal")


# ponytail: _installed_distro/_sync_repo are byte-identical to their copy in
# psychscanner-primal/src/psychscanner/library_download.py. Not factored into
# a shared package on purpose -- the two distros can never be co-installed
# (same `psychscanner` import name), so a shared dependency would only add a
# third package to version without removing that constraint. Keep both
# copies in sync by hand; if a fix here also applies there, apply it there
# too.
def _installed_distro() -> str | None:
    """Which of psychscanner/psychscanner-primal is installed here, if either
    (they share the `psychscanner` import name, so at most one really is)."""
    for dist_name, distro in (("psychscanner-primal", "primal"), ("psychscanner", "psychscanner")):
        try:
            metadata.version(dist_name)
            return distro
        except metadata.PackageNotFoundError:
            continue
    return None


# ponytail: shells out to the system `git` (already a dev-environment
# dependency) instead of adding a git library. No lockfile around the cache
# dir -- concurrent callers racing the same dest is unhandled, add a lock if
# this ever runs from parallel processes against one cache dir.
def _sync_repo(dest: Path, ref: str) -> None:
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref], check=True)
        subprocess.run(["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, LIBRARY_REPO_URL, str(dest)],
            check=True,
        )


def download_lib(
    library: Library = "psychscanner",
    kind: Kind = "both",
    dest: str | Path | None = None,
    ref: str = "main",
) -> dict[str, Path] | dict[str, dict[str, Path]]:
    """Clone/update `psyscan-library` and return the requested card subfolders.

    Parameters
    ----------
    library : "psychscanner" | "primal" | "all"
        Which distro's cards to hand back. `"psychscanner"` (default) and
        `"primal"` each check that they match the package actually
        installed in this environment -- cards from the other distro
        aren't guaranteed to run here -- and raise if they don't. `"all"`
        skips that check and returns both.
    kind : "tasks" | "experiments" | "both"
        Which card type(s) to return paths for. Primal has no experiment
        cards, so `kind` other than `"tasks"` with `library="primal"`
        raises.
    dest : str | os.PathLike | None
        Where to clone/update the checkout. Defaults to a shared cache dir
        (`~/.cache/psychscanner/psyscan-library`) so repeat calls just
        `git fetch` instead of re-cloning.
    ref : str
        Branch/tag to check out. Defaults to `"main"`.

    Returns
    -------
    dict[str, Path]
        `{"tasks": Path, ...}` for a single library. For `library="all"`,
        `{"psychscanner": {"tasks": Path, ...}, "primal": {...}}`.

    Raises
    ------
    ValueError
        `kind`/`library` invalid, or `library="primal"` combined with a
        `kind` that includes experiment cards.
    RuntimeError
        `library` doesn't match the installed package and isn't `"all"`.
    """
    if kind not in _KIND_DIRS:
        raise ValueError(f"kind must be one of {tuple(_KIND_DIRS)}, got {kind!r}")
    if library not in (*_DISTROS, "all"):
        raise ValueError(f"library must be one of {(*_DISTROS, 'all')}, got {library!r}")
    if library == "primal" and kind != "tasks":
        raise ValueError("primal has no experiment cards -- pass kind='tasks'")

    if library != "all":
        installed = _installed_distro()
        if installed != library:
            raise RuntimeError(
                f"library={library!r} but the installed package is {installed!r} -- "
                f"'{library}' cards aren't guaranteed to run here. Install the matching "
                "package, or pass library='all' to fetch both anyway."
            )

    dest = Path(dest) if dest is not None else _CACHE_DIR
    _sync_repo(dest, ref)

    distros = _DISTROS if library == "all" else (library,)
    results: dict[str, dict[str, Path]] = {}
    for distro in distros:
        distro_kinds = [k for k in _KIND_DIRS[kind] if not (distro == "primal" and k == "experiments")]
        results[distro] = {k: dest / k / distro for k in distro_kinds}

    return results if library == "all" else results[library]
