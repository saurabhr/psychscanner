# CLI Reference

This page lists the `--help` for `psychscanner`.

## psychscanner

Running `psychscanner --help` or `python -m psychscanner --help` shows a list of all of the available options and arguments:

<!-- [[[cog
import cog
from psychscanner import cli
from click.testing import CliRunner
result = CliRunner().invoke(cli.cli, ["--help"], terminal_width=88)
help = result.output.replace("Usage: cli", "Usage: psychscanner")
cog.outl(f"\n```sh\npsychscanner --help\n{help.rstrip()}\n```\n")
]]] -->
<!-- [[[end]]] -->
