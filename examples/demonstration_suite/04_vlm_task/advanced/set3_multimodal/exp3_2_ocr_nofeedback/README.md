# Exp 3.2 — Multimodal: OCR-VLM × No Feedback

**Status:** READY — all parameters resolved.

Identical to Exp 3.1 except `feedback=False`. See [Exp 3.1 README](../exp3_1_ocr_feedback/README.md).

| Factor | Value |
|--------|-------|
| **Model** | `nvidia/nemotron-nano-12b-v2-vl:free` (OpenRouter) |
| **VLM type** | OCR-VLM |
| **Feedback** | No |

## Headline Test
> Does feedback amplify or attenuate the OCR advantage on Lost-in-the-Middle?
> Key comparison: `(3.1 − 3.2)` accuracy delta per serial position.
