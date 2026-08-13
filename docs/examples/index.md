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

[`examples/demonstration_suite/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite)
also contains larger, multi-experiment study scripts, each nested under its
topically matching demo (`03_personality_survey/advanced/`,
`02_association_memory/advanced/`, `04_vlm_task/advanced/`) — see each
subfolder's own `README.md` for status and factorial design details.

## Contributing Examples

Have an interesting use case? Open an issue on [GitHub](https://github.com/saurabhr/psychscanner/issues)!
