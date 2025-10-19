<!-- start docs-include-index -->

# psychscanner-cli

[![PyPI](https://img.shields.io/pypi/v/psychscanner)](https://img.shields.io/pypi/v/psychscanner)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/psychscanner)](https://pypi.org/project/psychscanner/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/saurabhr/psychscanner/main.svg)](https://results.pre-commit.ci/latest/github/saurabhr/psychscanner/main)
[![Test](https://github.com/saurabhr/psychscanner/actions/workflows/test.yml/badge.svg)](https://github.com/saurabhr/psychscanner/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/psychscanner/badge/?version=latest)](https://psychscanner.readthedocs.io/en/latest/?badge=latest)
[![PyPI - License](https://img.shields.io/pypi/l/psychscanner)](https://img.shields.io/pypi/l/psychscanner)

A tool to bridge natural psychology witth  the artificial.

<!-- end docs-include-index -->

## Installation

<!-- start docs-include-installation -->

psychscanner-cli is available on [PyPI](https://pypi.org/project/psychscanner/). Install with [uv](https://docs.astral.sh/uv/) or your package manager of choice:

```sh
uv tool install psychscanner
```

<!-- end docs-include-installation -->

## Documentation

Check out the [psychscanner-cli documentation](https://psychscanner.readthedocs.io/en/stable/) for the [User's Guide](https://psychscanner.readthedocs.io/en/stable/usage.html) and [CLI Reference](https://psychscanner.readthedocs.io/en/stable/cli.html).

## Usage

<!-- start docs-include-usage -->

Running `psychscanner --help` or `python -m psychscanner --help` shows a list of all of the available options and arguments:

<!-- [[[cog
import cog
from psychscanner import cli
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(cli.cli, ["--help"], terminal_width=88)
help = result.output.replace("Usage: cli", "Usage: psychscanner")
cog.outl(f"\n```sh\npsychscanner --help\n{help.rstrip()}\n```\n")
]]] -->
<!-- [[[end]]] -->

<!-- end docs-include-usage -->
