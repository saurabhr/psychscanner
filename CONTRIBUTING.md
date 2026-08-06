# Contributing

Contributions are welcome — bug fixes, new tasks/parsers, docs, examples.

## Setup

```bash
uv venv psyscan --python 3.11
source psyscan/bin/activate
uv pip install -e ".[tests,mkdocs]"
```

## Before opening a PR

```bash
nox -s pre-commit   # lint/format
nox -s tests        # pytest
```

## Pull requests

- Keep PRs focused on one change.
- Add/update tests for behavior changes.
- Link any related issue in the PR description.
- If citation/reference info changes, update `CITATION.cff`, `README.md`, and `docs/index.md` together — they don't sync automatically.
