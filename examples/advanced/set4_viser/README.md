# Set 4 — VISER Visual Reasoning Experiments

Replication of VISER (Visual Input Structure for Enhanced Reasoning) from NeurIPS 2025 poster.

Focus on counting task: count specific objects in scenes with 10-20 objects.

Conditions:
- Exp 4.1: OCR-VLM + VISER (grid overlay on image)
- Exp 4.2: OCR-VLM - VISER
- Exp 4.3: Non-OCR + VISER (grid description in text)
- Exp 4.4: Non-OCR - VISER

Model: nvidia/nemotron-nano-12b-v2-vl:free

To run all experiments:
```bash
conda activate psyscan
for exp in exp4_1_ocr_viser exp4_2_ocr_no_viser exp4_3_nonocr_viser exp4_4_nonocr_no_viser; do
    python examples/advanced/set4_viser/$exp/run.py
done
```

Then, generate combined figure:
```bash
python examples/advanced/set4_viser/figures/make_figures.py
```</content>
<parameter name="filePath">/Users/saurabhext/Documents/PSYCHSCANNER/psychscanner/examples/advanced/set4_viser/README.md