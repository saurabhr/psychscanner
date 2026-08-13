# Contributing

Contributions are welcome — bug fixes, parsers, docs, examples.

**Contributing a task or experiment card?** That goes to
[`psyscan-library`](https://github.com/saurabhr/psyscan-library), the public,
versioned index of vetted cards for this package, not a PR against
`examples/tasks/` in this repo. It's validated for structure, checked for
duplicates, and actually run end-to-end against the mock LLM before merging —
see that repo's `CONTRIBUTING.md`. The rest of this doc covers code
contributions to `psychscanner` itself.

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
