from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner

from psychscanner.cli import (
    _parse_bool_or_none,
    _parse_csv_option,
    _parse_json_option,
    cli,
)

from .utils import run_command_in_shell


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_main_module() -> None:
    """Exercise (most of) the code in the `__main__` module."""
    import_module("psychscanner.__main__")


def test_run_as_module() -> None:
    """Is the script runnable as a Python module?"""
    result = run_command_in_shell("python -m psychscanner --help")
    assert result.exit_code == 0


def test_run_as_executable() -> None:
    """Is the script installed (as a `console_script`) and runnable as an executable?"""
    result = run_command_in_shell("psychscanner --help")
    assert result.exit_code == 0


def test_version_runner(runner: CliRunner) -> None:
    """Does `--version` display the correct version?"""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert (
        result.output == f"cli, version {version('psychscanner')}\n"
    )


def test_cli_runs_experiment_with_mock_llm(
    tmp_path: Path, runner: CliRunner
) -> None:
    """Does the CLI actually run an experiment, not just echo the flags?"""
    result = runner.invoke(
        cli,
        [
            "-m", "mock-chat-model",
            "-f", "mock-llm",
            "-projname", "cli_smoke",
            "-pd", str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Ran 1 result batch" in result.output

    csv_files = list(tmp_path.rglob("*.csv"))
    assert csv_files, f"expected a saved CSV under {tmp_path}, got: {result.output}"


# ── Regression tests for click option callbacks ──────────────────────────────
# type=dict/type=list[str]/type=click.Choice([True, False, None]) on a
# click.Option don't parse CLI strings the way their type hints suggest --
# these callbacks replace them. Covers a real bug: -p/-pcon crashed on any
# real JSON input, -tg/-pers silently exploded a comma-separated string into
# single characters, and -tc's non-string Choice values were undocumented,
# Click-version-dependent behavior.


def test_parse_json_option_parses_real_json() -> None:
    assert _parse_json_option(None, None, '{"temperature": 0.5}') == {"temperature": 0.5}


def test_parse_json_option_none_on_empty() -> None:
    assert _parse_json_option(None, None, None) is None
    assert _parse_json_option(None, None, "") is None


def test_parse_json_option_rejects_invalid_json() -> None:
    import click

    with pytest.raises(click.BadParameter):
        _parse_json_option(None, None, "{not json}")


def test_parse_csv_option_splits_on_commas() -> None:
    assert _parse_csv_option(None, None, "a,b,c") == ["a", "b", "c"]
    assert _parse_csv_option(None, None, "a, b , c") == ["a", "b", "c"]


def test_parse_csv_option_none_on_empty() -> None:
    assert _parse_csv_option(None, None, None) is None
    assert _parse_csv_option(None, None, "") is None


def test_parse_bool_or_none_maps_choices() -> None:
    assert _parse_bool_or_none(None, None, "true") is True
    assert _parse_bool_or_none(None, None, "false") is False
    assert _parse_bool_or_none(None, None, "none") is None
    assert _parse_bool_or_none(None, None, None) is None


def test_cli_accepts_json_and_csv_and_choice_flags(
    tmp_path: Path, runner: CliRunner
) -> None:
    """End-to-end: -p/-tg/-tc used to crash or silently corrupt input."""
    result = runner.invoke(
        cli,
        [
            "-m", "mock-chat-model",
            "-f", "mock-llm",
            "-projname", "cli_flags_smoke",
            "-pd", str(tmp_path),
            "-p", '{"temperature": 0.5}',
            "-tg", "a,b,c",
            "-tc", "false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Ran 1 result batch" in result.output
