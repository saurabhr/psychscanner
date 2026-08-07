# Psychscanner Features Demonstrated

- 1: Two levels of persona
- 2: Independent (stateless) trial sampling
- 3: Conversation and Summary memory (stateful)
- 4: Windowing Conversation and Summary Memory

## What's here

A BFI-44 (Big Five Inventory, 44 items / OCEAN traits) survey run across
2 persona conditions x 4 memory conditions, using real model calls
(Groq, `llama-3.3-70b-versatile`).

| Feature bullet | Condition(s) |
|---|---|
| 1. Two levels of persona | `weird`, `non_weird_jung` (axis files from `examples/advanced/personas/`) |
| 2. Independent stateless sampling | `stateless` (`memory=SingleTurn`) |
| 3. Conversation / Summary memory | `conversation` (`memory=Convo`, unbounded); `summary_windowed` (`memory=Convo` + `summary_k`) |
| 4. Windowing Convo / Summary memory | `conversation_windowed` (`memory_k=8`, no summary); `summary_windowed` (`memory_k=8`, `summary_k=4`) |

- `simulation/run_personality_survey.py` — runs all 8 cells (2 persona x 4 memory), idempotent
- `data/raw/*.csv` — per-cell trial output; `data/processed/combined.csv` — all cells combined
- `analysis/analyze_personality_survey.py` — reverse-scores items, computes per-trait means, persona/memory effect sizes, and a comparison figure
- `reports/report.md` — results write-up

Run order:
```bash
source .venv/bin/activate
python demonstrations/03_personality_survey/simulation/run_personality_survey.py
python demonstrations/03_personality_survey/analysis/analyze_personality_survey.py
```
