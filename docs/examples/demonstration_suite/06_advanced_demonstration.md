# Advanced Demonstration

**Status:** Othello-GPT and ROME complete and reported (see
[ROME's report.md](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/06_advanced_demonstration/reports/report.md) —
3/3 rank-one edits succeeded on GPT-2 small, ~44% mean collateral-change
rate); CCS not yet run.

Source: [`examples/demonstration_suite/06_advanced_demonstration/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/06_advanced_demonstration)

## Features demonstrated

Capstone: reproduces three classic interpretability studies on top of
[Extracting Internals](05_extracting_internals.md)'s nnsight/nnterp backend.

1. **Othello-GPT** — linear probing of residual-stream activations for an
   internal world model (Li et al., 2023)
2. **ROME** — causal tracing and rank-one editing to locate and edit
   factual associations (Meng et al., 2022)
3. **CCS** — unsupervised discovery of a "truth direction" via
   contrast-consistent search (Burns et al., 2022)

## Run it

```bash
source .venv/bin/activate
python examples/demonstration_suite/06_advanced_demonstration/simulation/othello/run_othello_probe.py
python examples/demonstration_suite/06_advanced_demonstration/simulation/rome/run_rome_lite.py
```

Results: [`data/othello/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite/06_advanced_demonstration/data/othello), [`reports/report.md`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/06_advanced_demonstration/reports/report.md).
