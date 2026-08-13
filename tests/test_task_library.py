"""Unit tests for name-based task card lookup (psychscanner.task_library)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychscanner.task_library import list_task_library, task_library


@pytest.fixture
def cwd_with_task_dirs(tmp_path, monkeypatch):
    """A tmp cwd with demonstrations/ and tasks/ subdirs, each holding one task card."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PSYCHSCANNER_TASK_LIBRARY_DIRS", raising=False)

    demos = tmp_path / "demonstrations"
    tasks = tmp_path / "tasks"
    demos.mkdir()
    tasks.mkdir()

    (demos / "shared_task.json").write_text(json.dumps({"taskname": "shared_task"}))
    (tasks / "builtin_task.json").write_text(json.dumps({"taskname": "builtin_task"}))
    return tmp_path


def test_task_library_finds_task_in_demonstrations(cwd_with_task_dirs):
    assert task_library("shared_task") == {"taskname": "shared_task"}


def test_task_library_finds_task_in_tasks(cwd_with_task_dirs):
    assert task_library("builtin_task") == {"taskname": "builtin_task"}


def test_task_library_format_path_returns_path_without_reading(cwd_with_task_dirs):
    result = task_library("builtin_task", format="path")
    assert isinstance(result, Path)
    assert result == cwd_with_task_dirs / "tasks" / "builtin_task.json"


def test_task_library_missing_raises_with_searched_dirs_listed(cwd_with_task_dirs):
    with pytest.raises(FileNotFoundError, match="demonstrations"):
        task_library("does_not_exist")


def test_task_library_invalid_format_raises():
    with pytest.raises(ValueError, match="format"):
        task_library("anything", format="yaml")


def test_task_library_malformed_json_raises_with_path(cwd_with_task_dirs):
    (cwd_with_task_dirs / "tasks" / "broken_task.json").write_text("{not valid json")

    with pytest.raises(ValueError, match="broken_task.json"):
        task_library("broken_task")


def test_task_library_env_var_dir_takes_priority(cwd_with_task_dirs, tmp_path, monkeypatch):
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    (override_dir / "builtin_task.json").write_text(json.dumps({"taskname": "overridden"}))
    monkeypatch.setenv("PSYCHSCANNER_TASK_LIBRARY_DIRS", str(override_dir))

    assert task_library("builtin_task") == {"taskname": "overridden"}


def test_list_task_library_aggregates_and_dedupes(cwd_with_task_dirs):
    names = list_task_library()
    assert names == ["builtin_task", "shared_task"]


def test_task_library_custom_dir_single_path(cwd_with_task_dirs, tmp_path):
    custom = tmp_path.parent / "elsewhere_single"
    custom.mkdir()
    (custom / "custom_task.json").write_text(json.dumps({"taskname": "from_custom_dir"}))

    assert task_library("custom_task", dirs=custom) == {"taskname": "from_custom_dir"}


def test_task_library_custom_dir_list_and_priority(cwd_with_task_dirs, tmp_path):
    """A custom dir is checked before demonstrations/tasks, even for a name that exists there too."""
    custom = tmp_path.parent / "elsewhere_list"
    custom.mkdir()
    (custom / "builtin_task.json").write_text(json.dumps({"taskname": "overridden_by_dirs_arg"}))

    result = task_library("builtin_task", dirs=[custom])
    assert result == {"taskname": "overridden_by_dirs_arg"}


def test_task_library_custom_dir_not_found_still_falls_back(cwd_with_task_dirs, tmp_path):
    """If the name isn't in the custom dir, the default search dirs are still checked."""
    custom = tmp_path.parent / "elsewhere_empty"
    custom.mkdir()

    assert task_library("builtin_task", dirs=custom) == {"taskname": "builtin_task"}


def test_list_task_library_includes_custom_dir(cwd_with_task_dirs, tmp_path):
    custom = tmp_path.parent / "elsewhere_for_listing"
    custom.mkdir()
    (custom / "custom_task.json").write_text(json.dumps({"taskname": "x"}))

    assert list_task_library(dirs=custom) == ["builtin_task", "custom_task", "shared_task"]


def test_task_library_warns_on_shadowed_name(cwd_with_task_dirs):
    (cwd_with_task_dirs / "tasks" / "shared_task.json").write_text(
        json.dumps({"taskname": "shadowed_copy"})
    )

    with pytest.warns(UserWarning, match="shared_task.*more than one"):
        result = task_library("shared_task")

    # demonstrations/ still wins (search order), the warning is informational only
    assert result == {"taskname": "shared_task"}


def test_task_library_no_warning_when_name_is_unique(cwd_with_task_dirs, recwarn):
    task_library("shared_task")
    assert len(recwarn) == 0


def test_list_task_library_warns_on_shadowed_name(cwd_with_task_dirs):
    (cwd_with_task_dirs / "tasks" / "shared_task.json").write_text(
        json.dumps({"taskname": "shadowed_copy"})
    )

    with pytest.warns(UserWarning, match="shared_task.*more than one"):
        names = list_task_library()

    assert names == ["builtin_task", "shared_task"]
