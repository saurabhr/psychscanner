# Examples

Practical examples demonstrating PsychScanner capabilities. These pages are
narrative, code-snippet walkthroughs — for runnable code, see the
[Tutorials](../tutorials/index.md) section, which renders the actual notebooks
from [`examples/`](https://github.com/saurabhr/psychscanner/tree/main/examples).

## Advanced Examples

- [Conditional Next-Trial](conditional_next_trial.md) — dynamic trial insertion

## Demonstration Suite

Real, multi-experiment studies with actual run data and reports — see the
[Demonstration Suite overview](demonstration_suite/index.md). Covers surveys
with persona conditioning (formerly "Simple Survey" / "Multi-Persona Study"
above) and trial-by-trial feedback (formerly "Feedback Loop System") with
real data behind them, plus reward-learning, VLM, and mechanistic
interpretability demos this page didn't previously have narrative examples
for.

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
