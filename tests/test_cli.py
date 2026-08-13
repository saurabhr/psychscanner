from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner

from psychscanner.cli import cli

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
