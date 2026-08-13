# Advanced Multimodal Experiments

Two sub-experiment sets on vision-language tasks, both using OpenRouter models.
These sit alongside the main [`04_vlm_task`](../readme.md) demo — same topic
(VLM + feedback), more elaborate factorial designs.

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

```bash
source psyscan/bin/activate

python examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal/exp3_1_ocr_feedback/run.py
python examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal/exp3_2_ocr_nofeedback/run.py
python examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal/exp3_3_nonocr_feedback/run.py
python examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal/exp3_4_nonocr_nofeedback/run.py
```

## Set 4 — VISER Visual Reasoning

Replication of VISER (Visual Input Structure for Enhanced Reasoning) from a NeurIPS
2025 poster. Counting task: count specific objects in scenes with 10-20 objects.

| Cell | Condition |
|------|-----------|
| Exp 4.1 | OCR-VLM + VISER (grid overlay on image) |
| Exp 4.2 | OCR-VLM − VISER |
| Exp 4.3 | Non-OCR + VISER (grid description in text) |
| Exp 4.4 | Non-OCR − VISER |

**Model:** `nvidia/nemotron-nano-12b-v2-vl:free`

```bash
source psyscan/bin/activate

for exp in exp4_1_ocr_viser exp4_2_ocr_no_viser exp4_3_nonocr_viser exp4_4_nonocr_no_viser; do
    python examples/demonstration_suite/04_vlm_task/advanced/set4_viser/$exp/run.py
done

# Then, generate the combined figure:
python examples/demonstration_suite/04_vlm_task/advanced/set4_viser/figures/make_figures.py
```

All `run.py` scripts are **idempotent**: re-running skips conditions where output CSVs
already exist and clears stale session directories automatically.
