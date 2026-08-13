# Exp 3.1 — Multimodal: OCR-VLM × With Feedback

**Status:** READY — all parameters resolved.

---

## Design

| Factor | Value |
|--------|-------|
| **Model** | `nvidia/nemotron-nano-12b-v2-vl:free` (OpenRouter) |
| **VLM type** | OCR-VLM (receives PIL-rendered PNG images) |
| **Feedback** | Yes — correct/incorrect + correct answer after each probe |
| **Sequences** | 10 (one probe each) |
| **Simulations** | 5 (`nsim=5`, `cogtype="no"`) |
| **Memory** | `Convo`, `chain_type=task`, no trimming |

---

## Task: Behavioral Profiling Task (BPT-10)

Shared stimulus design across all 4 Set 3 sub-experiments (`_shared.py`).

Each of 10 sequences:
1. **Study phase** — 10 word-number pairs presented one at a time:
   - Odd positions (1,3,5,7,9): word-number pair rendered as a **PIL PNG image**
   - Even positions (2,4,6,8,10): plain text `"Memorize: WORD → NUMBER"`
2. **Probe phase** — text question: `"Recall test: What number was paired with WORD?"`
   - Parser: `SerialProbeResponse` → `recalled_number` + `confidence (1–6)`

**Probe serial positions** (one per sequence):
```
seq_01 → position 1   seq_06 → position 6
seq_02 → position 2   seq_07 → position 7
seq_03 → position 3   seq_08 → position 8
seq_04 → position 4   seq_09 → position 9
seq_05 → position 5   seq_10 → position 10
```

---

## Behavioral Measures

### Lost-in-the-Middle Effect
> Does accuracy decline for items at middle serial positions (4–7)?
> OCR-VLM hypothesis: re-attending to the image mitigates this compared to non-OCR.

**IV:** `serial_position` (1–10)  
**DV:** accuracy (`recalled_number == number`)

### Attentional Residue
> Is accuracy lower when the probe target was an image-slot item vs. a text-slot item?
> Image-slot recall requires a modality switch (image attention → text output).

**IV:** `probe_target_modality` (image | text)  
**DV:** accuracy

---

## Outputs

- `raw/exp3_1_ocr_feedback.csv` — all 550 trial rows (5 sims × 110 stimuli)
- `analysis/probe_scored.csv` — probe rows with `correct` column
- `figures/serial_position_curve.png`
- `figures/modality_switch_effect.png`
