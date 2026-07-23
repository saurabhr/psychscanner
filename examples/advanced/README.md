# Advanced PsychScanner Examples

Nine sub-experiments organised into three sets, all using the same base model:
**`ollama / smollm2:360m-instruct-fp16`**, temperature 0, n=5 simulated participants
per condition.

---

## Experiment overview

| Exp | Set | Task | Status |
|-----|-----|------|--------|
| 1.1 | Surveys | VVIQ-16 imagery survey | ✅ Data collected |
| 1.2a | Surveys | BFI-44, 2-persona (WEIRD / non-WEIRD+Jung) | 🔄 Running |
| 1.2b | Surveys | BFI-44, 3-persona (adds Zhang) | ⏳ Ready to run |
| 2.1 | Episodic | PAL-50, with feedback | 🔄 Running |
| 2.2 | Episodic | PAL-50, no feedback | ⏳ Ready to run |
| 3.1 | Multimodal | OCR-VLM × With Feedback | ⏳ Ready to run |
| 3.2 | Multimodal | OCR-VLM × No Feedback | ⏳ Ready to run |
| 3.3 | Multimodal | non-OCR-VLM × With Feedback | ⏳ Ready to run |
| 3.4 | Multimodal | non-OCR-VLM × No Feedback | ⏳ Ready to run |

---

## Set 1 — Surveys

Tests how **persona conditioning** and **cross-item memory** affect model self-report.

### Exp 1.1 — VVIQ-16
```
set1_surveys/exp1_1_vviq16/run.py
```
**Factors:** Persona(WEIRD/non-WEIRD) × Memory(item_ST / task_Cv / trial_Cv)
**Task:** 16-item VVIQ vividness rating (1–5 Likert)
**Output:** 480 rows, `figures/trial_trajectory.png`, `figures/total_vviq_violin.png`

### Exp 1.2a — BFI-44, 2-Persona
```
set1_surveys/exp1_2a_bigfive_2persona/run.py
```
**Factors:** Persona(WEIRD/non-WEIRD+Jung) × SummaryK(1,5) × MemoryK(4,16) + Control
**Task:** 44-item BFI-44 OCEAN personality inventory (1–5 agreement Likert)
**Participants:** axis(1) × traits_n5(5) = 5 per condition
**Output:** 2 200 rows, OCEAN radar, per-trait figures

### Exp 1.2b — BFI-44, 3-Persona
```
set1_surveys/exp1_2b_bigfive_3persona/run.py
```
**Adds Zhang** persona: 5 individuals from Zhang et al. (`persona_200.json`), used
without cultural-axis framing.
**Output:** 3 300 rows, 3-persona OCEAN radar

---

## Set 2 — Episodic (PAL-50)

Tests whether **memory architecture** affects paired-associate learning across 5 semantic
similarity levels.

**Task:** PAL-50 — 50 word pairs (10 per level: 0.0 / 0.25 / 0.50 / 0.75 / 1.0), study
phase then test phase. Parser: `PairedAssociateRecall` (recalled_word + confidence 1–6).

**Memory conditions:**

| Label | chain_type | memory | Description |
|-------|-----------|--------|-------------|
| item_ST | item | SingleTurn | Independent trials — pure prior knowledge baseline |
| task_Cv | task | Convo | All trials in one thread — full episodic memory |
| trial_Cv | trial | Convo | 50 dual-stimulus items — within-pair encoding+test |

### Exp 2.1 — With Feedback
```
set2_episodic/exp2_1_with_feedback/run.py
```
Feedback after each test trial: Correct/Incorrect + correct answer revealed.

### Exp 2.2 — No Feedback
```
set2_episodic/exp2_2_no_feedback/run.py
```
Also generates `figures/exp2_1_vs_2_2_feedback_contrast.png` (cross-sub-exp comparison).

---

## Set 3 — Multimodal (Behavioral Profiling)

2×2: VLM-type (OCR / non-OCR) × Feedback (Yes / No).

**Task:** Behavioral Profiling Task (BPT-10) — 10 sequences × 10 study items + 1 probe.

| Cell | Model | Feedback |
|------|-------|---------|
| Exp 3.1 | `nvidia/nemotron-nano-12b-v2-vl:free` | Yes |
| Exp 3.2 | `nvidia/nemotron-nano-12b-v2-vl:free` | No |
| Exp 3.3 | `mistralai/mistral-nemo-12b-instruct:free` | Yes |
| Exp 3.4 | `mistralai/mistral-nemo-12b-instruct:free` | No |

**Measures:**
- **Lost-in-the-Middle** — accuracy by serial position (1–10) of probe target
- **Attentional Residue** — accuracy by probe target modality (image-slot vs. text-slot)

**Shared code:** `set3_multimodal/_shared.py`  
**API:** OpenRouter (`OPENROUTER_API_KEY` in `.env`)

---

## Shared resources

| Path | Description |
|------|-------------|
| `personas/weird/weird_axis.json` | WEIRD cultural axis (1 stmt) |
| `personas/non_weird/non_weird_axis.json` | non-WEIRD axis (1 stmt) |
| `personas/jung/jung_axis.json` | Jungian I+N typology axis (1 stmt) |
| `personas/zhang/zhang_axis.json` | Zhang et al. individual descriptions (5 stmts) |
| `personas/traits_n5.json` | 5 diverse trait profiles from `persona_200.json` |
| `examples/tasks/vviq16.json` | VVIQ-16 task |
| `examples/tasks/bfi44.json` | BFI-44 task (44 items, `DefaultLiteralAgree` parser) |
| `examples/tasks/pal50.json` | PAL-50 task (100 items, `PairedAssociateRecall` parser) |

---

## Running order

```bash
source psyscan/bin/activate

# Set 1
python examples/advanced/set1_surveys/exp1_1_vviq16/run.py
python examples/advanced/set1_surveys/exp1_2a_bigfive_2persona/run.py
python examples/advanced/set1_surveys/exp1_2b_bigfive_3persona/run.py

# Set 2
python examples/advanced/set2_episodic/exp2_1_with_feedback/run.py
python examples/advanced/set2_episodic/exp2_2_no_feedback/run.py

# Set 3 (OpenRouter — requires OPENROUTER_API_KEY in .env)
python examples/advanced/set3_multimodal/exp3_1_ocr_feedback/run.py
python examples/advanced/set3_multimodal/exp3_2_ocr_nofeedback/run.py
python examples/advanced/set3_multimodal/exp3_3_nonocr_feedback/run.py
python examples/advanced/set3_multimodal/exp3_4_nonocr_nofeedback/run.py
```

All `run.py` scripts are **idempotent**: re-running skips conditions where output CSVs
already exist and clears stale session directories automatically.
