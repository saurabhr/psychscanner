"""Shared analysis infrastructure for the demonstration_suite experiments.

Each experiment's own analyze.py keeps its own statistical logic — the four
existing scripts use different libraries (polars/pandas) and answer
genuinely different questions, so there's no single "the" analysis to run.
This module only deduplicates the boilerplate every one of them already
reimplements: locating data/output dirs, the Agg backend setup, writing a
text summary alongside printing it, and saving a figure with consistent
settings.
"""
from __future__ import annotations

from pathlib import Path


def demo_dirs(analysis_file: str | Path, *, data_subdir: str = "data") -> tuple[Path, Path]:
    """Return ``(DATA_DIR, OUT_DIR)`` for an ``analyze.py`` living in ``<demo>/analysis/``.

    ``OUT_DIR`` is the ``analysis/`` directory itself (where scripts write
    output alongside their own source); ``DATA_DIR`` is ``<demo>/<data_subdir>``.
    """
    out_dir = Path(analysis_file).resolve().parent
    data_dir = out_dir.parent / data_subdir
    return data_dir, out_dir


def require_data(found, data_dir: Path, hint: str) -> None:
    """Raise ``SystemExit`` with a helpful message if no input files were found."""
    if not found:
        raise SystemExit(f"No result files found in {data_dir} -- {hint}")


def write_summary(out_dir: Path, lines: list[str], filename: str = "summary.txt") -> str:
    """Join ``lines``, print them, and write them to ``out_dir/filename``. Returns the text."""
    text = "\n".join(lines)
    print(text)
    (out_dir / filename).write_text(text + "\n")
    return text


def use_agg_backend() -> None:
    """Set matplotlib's non-interactive Agg backend.

    Call this before ``import matplotlib.pyplot as plt`` in the caller —
    ``matplotlib.use()`` only takes effect if run before pyplot is imported.
    """
    import matplotlib
    matplotlib.use("Agg")


def save_figure(fig, out_dir: Path, filename: str, *, dpi: int = 150, **savefig_kwargs) -> Path:
    """Save a matplotlib figure to ``out_dir/filename`` at a consistent dpi, print the path."""
    out = out_dir / filename
    fig.savefig(out, dpi=dpi, **savefig_kwargs)
    print(f"-> {out}")
    return out


def _demo() -> None:
    """assert-based self-check — no data/matplotlib required."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        demo_root = Path(tmp).resolve() / "01_demo"
        (demo_root / "analysis").mkdir(parents=True)
        (demo_root / "data").mkdir()
        fake_analyze_py = demo_root / "analysis" / "analyze.py"
        fake_analyze_py.write_text("# fake")

        data_dir, out_dir = demo_dirs(fake_analyze_py)
        assert data_dir == demo_root / "data"
        assert out_dir == demo_root / "analysis"

        try:
            require_data([], data_dir, "run the simulation first.")
        except SystemExit as e:
            assert "run the simulation first." in str(e)
        else:
            raise AssertionError("require_data([], ...) should have raised SystemExit")

        require_data(["not-empty"], data_dir, "unused")  # must not raise

        text = write_summary(out_dir, ["line one", "line two"], filename="check.txt")
        assert text == "line one\nline two"
        assert (out_dir / "check.txt").read_text() == "line one\nline two\n"

    print("_analysis_common self-check OK")


if __name__ == "__main__":
    _demo()
