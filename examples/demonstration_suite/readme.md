# Psychscanner Demonstrations

[00_install_from_github.ipynb](00_install_from_github.ipynb) — install psychscanner from GitHub before running any of the below.

Suggested reading order, roughly increasing in complexity (not a technical dependency between subprojects — each is currently just a standalone readme + folder skeleton).

1. [reward_task](01_reward_task/readme.md) — swapping different agents/models into the same task harness.
2. [association_memory](02_association_memory/readme.md) — core trial structures (single-turn, trial-chain, episodic-chain) and feedback types (correct/incorrect, reward).
3. [personality_survey](03_personality_survey/readme.md) — persona conditioning plus stateful (conversation/summary, windowed) memory.
4. [vlm_task](04_vlm_task/readme.md) — the same trial/summary/feedback machinery extended to vision-language models.
5. [extracting_internals](05_extracting_internals/readme.md) — mechanistic interpretability: extracting and steering internal activations (Persona Vectors).
6. [advanced_demonstration](06_advanced_demonstration/readme.md) — capstone: reproduces three classic interpretability studies (Othello-GPT, ROME, CCS) on top of everything above.
7. [introspective_selfreport](07_introspective_selfreport/readme.md) — adapts Plunkett et al. (2025)'s LLM self-interpretability paradigm (instilled attribute-weight preferences + introspective self-report), plus an SDT/metacognition extension via `metasignal` and a `psychscanner-primal` task-card integration.
