# Psychscanner Demonstrations

Suggested reading order, roughly increasing in complexity (not a technical dependency between subprojects — each is currently just a standalone readme + folder skeleton).

1. [reward_task](01_reward_task/readme.md) — swapping different agents/models into the same task harness.
2. [association_memory](02_association_memory/readme.md) — core trial structures (single-turn, trial-chain, episodic-chain) and feedback types (correct/incorrect, reward).
3. [personality_survey](03_personality_survey/readme.md) — persona conditioning plus stateful (conversation/summary, windowed) memory.
4. [vlm_task](04_vlm_task/readme.md) — the same trial/summary/feedback machinery extended to vision-language models.
5. [extracting_internals](05_extracting_internals/readme.md) — mechanistic interpretability: extracting and steering internal activations (Persona Vectors).
6. [advanced_demonstration](06_advanced_demonstration/readme.md) — capstone: reproduces three classic interpretability studies (Othello-GPT, ROME, CCS) on top of everything above.
