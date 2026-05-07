"""Run the README Quick Start unit test via pytest, scoped to its file only.

Why a custom runner?
--------------------
A bare `pytest` would also collect `tests/test_cli.py`, which has known pre-existing
failures unrelated to the README example. This runner targets only
`tests/test_readme_quickstart.py` so the README example can be validated in
isolation as a real pytest run.

Usage
-----
    python scripts/run_readme_test.py            # run all README tests
    python scripts/run_readme_test.py -k smoke   # forward extra pytest args

Exit code is the pytest exit code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = REPO_ROOT / "tests" / "test_readme_quickstart.py"


def main(extra_args: list[str] | None = None) -> int:
    if not TEST_FILE.exists():
        print(f"ERROR: test file not found: {TEST_FILE}", file=sys.stderr)
        return 2

    args = [
        str(TEST_FILE),
        "-v",
        "--no-header",
        "-p", "no:cacheprovider",
        "--rootdir", str(REPO_ROOT),
        "--confcutdir", str(REPO_ROOT / "tests"),
    ]
    if extra_args:
        args.extend(extra_args)

    print(f"$ pytest {' '.join(args)}")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
