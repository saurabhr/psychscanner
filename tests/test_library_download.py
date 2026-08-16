"""Unit tests for psychscanner.download_lib -- parameter validation and path
construction. _sync_repo is monkeypatched everywhere so no test touches the
network or git.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from psychscanner import library_download as ld
from psychscanner.library_download import download_lib


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    calls = []
    monkeypatch.setattr(ld, "_sync_repo", lambda dest, ref: calls.append((dest, ref)))
    return calls


@pytest.fixture
def installed_as(monkeypatch):
    def _set(distro):
        monkeypatch.setattr(ld, "_installed_distro", lambda: distro)

    return _set


def test_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        download_lib(kind="bogus")


def test_rejects_bad_library():
    with pytest.raises(ValueError, match="library"):
        download_lib(library="bogus")


def test_primal_with_experiments_rejected_before_any_network_call(no_network):
    with pytest.raises(ValueError, match="experiment"):
        download_lib(library="primal", kind="experiments")
    assert no_network == []


def test_primal_with_both_rejected():
    with pytest.raises(ValueError, match="experiment"):
        download_lib(library="primal", kind="both")


def test_distro_mismatch_raises_before_network_call(installed_as, no_network):
    installed_as("psychscanner")
    with pytest.raises(RuntimeError, match="primal"):
        download_lib(library="primal", kind="tasks")
    assert no_network == []


def test_matching_distro_returns_task_and_experiment_paths(tmp_path, installed_as):
    installed_as("psychscanner")
    paths = download_lib(library="psychscanner", kind="both", dest=tmp_path)
    assert paths == {"tasks": tmp_path / "tasks" / "psychscanner", "experiments": tmp_path / "experiments" / "psychscanner"}


def test_kind_tasks_only_omits_experiments_key(tmp_path, installed_as):
    installed_as("psychscanner")
    paths = download_lib(library="psychscanner", kind="tasks", dest=tmp_path)
    assert paths == {"tasks": tmp_path / "tasks" / "psychscanner"}


def test_library_all_skips_distro_check_and_covers_both(tmp_path, installed_as):
    installed_as("primal")  # deliberately mismatched -- "all" bypasses the check
    paths = download_lib(library="all", kind="both", dest=tmp_path)
    assert set(paths) == {"psychscanner", "primal"}
    assert paths["psychscanner"] == {
        "tasks": tmp_path / "tasks" / "psychscanner",
        "experiments": tmp_path / "experiments" / "psychscanner",
    }
    # primal has no experiment cards, even under library="all"
    assert paths["primal"] == {"tasks": tmp_path / "tasks" / "primal"}


def test_sync_repo_called_with_dest_and_ref(tmp_path, installed_as, no_network):
    installed_as("psychscanner")
    download_lib(library="psychscanner", dest=tmp_path, ref="v1.2")
    assert no_network == [(tmp_path, "v1.2")]


def test_default_dest_is_shared_cache_dir(installed_as):
    installed_as("psychscanner")
    paths = download_lib(library="psychscanner", kind="tasks")
    assert paths["tasks"] == Path.home() / ".cache" / "psychscanner" / "psyscan-library" / "tasks" / "psychscanner"


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-q"]))
