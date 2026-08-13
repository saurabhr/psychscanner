"""Unit tests for name-based experiment card lookup (psychscanner.experiment_library).

Mirrors tests/test_task_library.py's structure and fixture pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychscanner import ExpCard, ExpCardInit, EXPERIMENT_CARD_EXT, save_experiment_card
from psychscanner.experiment_library import experiment_library, list_experiment_library


def _card(**overrides) -> ExpCardInit:
    kwargs = dict(
        model="mock-llm",
        family="mock-llm",
        task_file={"taskname": "t", "items": {}},
        cogtype="no",
        nsim=1,
        memory="SingleTurn",
    )
    kwargs.update(overrides)
    return ExpCardInit(**kwargs)


@pytest.fixture
def cwd_with_experiment_dirs(tmp_path, monkeypatch):
    """A tmp cwd with experiments/ and demonstrations/ subdirs, each holding one experiment card."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PSYCHSCANNER_EXPERIMENT_LIBRARY_DIRS", raising=False)

    experiments = tmp_path / "experiments"
    demos = tmp_path / "demonstrations"
    experiments.mkdir()
    demos.mkdir()

    save_experiment_card(_card(), experiments / f"builtin_exp{EXPERIMENT_CARD_EXT}")
    save_experiment_card(_card(), demos / f"shared_exp{EXPERIMENT_CARD_EXT}")
    return tmp_path


def test_experiment_library_finds_in_experiments_dir(cwd_with_experiment_dirs):
    card = experiment_library("builtin_exp", proj_dir=cwd_with_experiment_dirs / "results")
    assert card.model == "mock-llm"


def test_experiment_library_finds_in_demonstrations_dir(cwd_with_experiment_dirs):
    card = experiment_library("shared_exp", proj_dir=cwd_with_experiment_dirs / "results")
    assert card.model == "mock-llm"


def test_experiment_library_format_path(cwd_with_experiment_dirs):
    result = experiment_library("builtin_exp", format="path")
    assert isinstance(result, Path)
    assert result == cwd_with_experiment_dirs / "experiments" / f"builtin_exp{EXPERIMENT_CARD_EXT}"


def test_experiment_library_format_json(cwd_with_experiment_dirs):
    result = experiment_library("builtin_exp", format="json")
    assert isinstance(result, dict)
    assert result["model"] == "mock-llm"


def test_experiment_library_missing_raises_with_searched_dirs_listed(cwd_with_experiment_dirs):
    with pytest.raises(FileNotFoundError, match="experiments"):
        experiment_library("does_not_exist")


def test_experiment_library_invalid_format_raises():
    with pytest.raises(ValueError, match="format"):
        experiment_library("anything", format="yaml")


def test_experiment_library_env_var_dir_takes_priority(cwd_with_experiment_dirs, tmp_path, monkeypatch):
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    save_experiment_card(_card(model="from-override"), override_dir / f"builtin_exp{EXPERIMENT_CARD_EXT}")
    monkeypatch.setenv("PSYCHSCANNER_EXPERIMENT_LIBRARY_DIRS", str(override_dir))

    card = experiment_library("builtin_exp", proj_dir=cwd_with_experiment_dirs / "results")
    assert card.model == "from-override"


def test_list_experiment_library_aggregates_and_dedupes(cwd_with_experiment_dirs):
    names = list_experiment_library()
    assert names == ["builtin_exp", "shared_exp"]


def test_experiment_library_custom_dir(cwd_with_experiment_dirs, tmp_path):
    custom = tmp_path.parent / "elsewhere"
    custom.mkdir()
    save_experiment_card(_card(model="from-custom"), custom / f"custom_exp{EXPERIMENT_CARD_EXT}")

    card = experiment_library("custom_exp", dirs=custom, proj_dir=cwd_with_experiment_dirs / "results")
    assert card.model == "from-custom"


def test_experiment_library_warns_on_shadowed_name(cwd_with_experiment_dirs):
    save_experiment_card(
        _card(), cwd_with_experiment_dirs / "demonstrations" / f"builtin_exp{EXPERIMENT_CARD_EXT}"
    )

    with pytest.warns(UserWarning, match="builtin_exp.*more than one"):
        experiment_library("builtin_exp", proj_dir=cwd_with_experiment_dirs / "results")


def test_experiment_library_no_warning_when_name_is_unique(cwd_with_experiment_dirs, recwarn):
    experiment_library("builtin_exp", proj_dir=cwd_with_experiment_dirs / "results")
    assert len(recwarn) == 0
