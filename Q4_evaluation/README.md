# Q4 — Automated Quality Evaluation

## What's implemented
- 3-axis quality evaluation pipeline:
  1. **Garment Fidelity**: OpenCLIP (ViT-B/32) embedding cosine similarity.
  2. **Identity Preservation**: InsightFace (buffalo_l) face crop SSIM and embedding distance.
  3. **VLM Judge**: Florence-2 rubric prompt rating (1–10 scale + reasoning).
- CSV exporter matching `evaluation_template_q4.csv` column format exactly.
- Complete metric reporting for all 5 try-on pairs (`pair_01`–`pair_05`).

## How it works
- **Garment Fidelity**: Input garment image embedding vs. try-on output garment region embedding → 0.0–1.0 score.
- **Identity Preservation**: Input person face crop vs. try-on output face crop → SSIM + InsightFace distance → 0.0–1.0 score.
- **VLM Judge**: Florence-2 given structured rubric prompt evaluating texture transfer, fit realism, and visual artifacts → 1–10 score.
- **CSV Output**: All three scores reported independently per Rule D9 without blending into a single arbitrary scalar.

## Why it took this long
- Setup & loading 3 evaluation models (~30 min).
- Face detection and garment region cropping integration (~1.5 hrs).
- Rubric prompt design & VLM score parser (~1 hr).
- Batch scoring and CSV export validation (~1 hr).

## Gotchas / edge cases
- Score normalization: Ensuring OpenCLIP and InsightFace scores map cleanly to [0, 1] bounds.
- Unblended reporting (Rule D9): Keeping the three metric axes separate in the final CSV output.

## Constraints hit
- **Memory**: Models loaded sequentially (OpenCLIP → InsightFace → Florence-2); max VRAM under ~3.0 GB.
- **Errors**: None encountered during initial setup.
