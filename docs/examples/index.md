# Examples

Practical examples demonstrating PsychScanner capabilities. These pages are
narrative, code-snippet walkthroughs — for runnable code, see the
[Tutorials](../tutorials/index.md) section, which renders the actual notebooks
from [`examples/`](https://github.com/saurabhr/psychscanner/tree/main/examples).

## Basic Examples

- [Simple Survey](basic_survey.md) — running a basic questionnaire

## Advanced Examples

- [Multi-Persona Study](multi_persona.md) — testing multiple personas
- [Reality Monitoring Task](reality_monitoring.md) — complex cognitive task
- [Feedback Loop System](feedback_loop.md) — trial-by-trial feedback

## Running the tutorial notebooks

```bash
# Clone the repository
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv pip install -e ".[dev]"

# Launch the notebooks
jupyter lab examples/
```

The `examples/advanced/` directory also contains larger, multi-experiment
study scripts (organized by `set1_surveys/`, `set2_episodic/`, etc.) — see
[`examples/advanced/README.md`](https://github.com/saurabhr/psychscanner/tree/main/examples/advanced)
for their current status.

## Contributing Examples

Have an interesting use case? Open an issue on [GitHub](https://github.com/saurabhr/psychscanner/issues)!
