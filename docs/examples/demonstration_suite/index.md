# Demonstration Suite

[`examples/demonstration_suite/`](https://github.com/saurabhr/psychscanner/tree/main/examples/demonstration_suite)
holds larger, multi-experiment study scripts — real model calls, real data,
and (where complete) an analyzed report, as opposed to the narrative
walkthroughs elsewhere in Examples. Start with
[`00_install_from_github.ipynb`](https://github.com/saurabhr/psychscanner/blob/main/examples/demonstration_suite/00_install_from_github.ipynb)
to install psychscanner before running any of them.

Suggested reading order, roughly increasing in complexity (not a technical
dependency between subprojects). "Complete" means real, non-mock run data and
an analyzed `reports/report.md` exist; "skeleton" means readme + folder only.

| # | Demo | Status |
|---|------|--------|
| 1 | [Reward Task](01_reward_task.md) — swapping agents into the same task harness | Complete |
| 2 | [Association Memory](02_association_memory.md) — trial structures and feedback types | Complete |
| 3 | [Personality Survey](03_personality_survey.md) — persona conditioning, stateful memory | VVIQ-16 arm complete |
| 4 | [VLM Task](04_vlm_task.md) — the same machinery extended to vision-language models | 1 of 4 conditions collected |
| 5 | [Extracting Internals](05_extracting_internals.md) — activation extraction and steering | Skeleton |
| 6 | [Advanced Demonstration](06_advanced_demonstration.md) — Othello-GPT, ROME, CCS | Othello-GPT and ROME complete |
| 7 | [Introspective Self-Report](07_introspective_selfreport.md) — instilled preferences + self-report | Complete |
| 8 | [Prospect Theory Planning](08_prospect_theory_planning.md) — risky-choice gambles, 3 agent architectures | Complete |

Each subfolder's own `README.md` in the repo has the full status and (where
applicable) factorial design details, including the `advanced/` arms nested
under `03_personality_survey`, `02_association_memory`, and `04_vlm_task`.
