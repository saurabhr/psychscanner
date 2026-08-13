# Exp 3.3 — Multimodal: non-OCR-VLM × With Feedback

**Status:** READY — all parameters resolved.

Text-only model baseline. Image-slot items delivered as `"[IMAGE TEXT] WORD → NUMBER"`.
See [Exp 3.1 README](../exp3_1_ocr_feedback/README.md) for full design.

| Factor | Value |
|--------|-------|
| **Model** | `mistralai/mistral-nemo-12b-instruct:free` (OpenRouter) |
| **VLM type** | non-OCR (text-only) |
| **Feedback** | Yes |

## Headline Test
> Do non-OCR models show a steeper Lost-in-the-Middle gradient vs. OCR?
> Does `probe_target_modality` (described vs. direct text) affect accuracy for text-only models?
