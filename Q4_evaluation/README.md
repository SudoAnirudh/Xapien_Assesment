# Q4 — Automated Quality Evaluation (20 marks)

**Google Colab Notebook**: [Q4 Colab Notebook](https://drive.google.com/file/d/1W8fAn_AZj6KagohAJcwLZt3SCT4TMNY6/view?usp=sharing)

## What's Implemented
- **Automated 3-Axis Quality Evaluation Pipeline**:
  1. **Garment Fidelity**: OpenCLIP (`ViT-B/32`, `laion2b_s34b_b79k`) embedding cosine similarity comparing input garment product image vs. try-on output garment region ($[0.0, 1.0]$ score).
  2. **Identity Preservation**: Facial region crop SSIM / InsightFace (`buffalo_l`) embedding distance comparing original person face vs. try-on result face ($[0.0, 1.0]$ score).
  3. **VLM-as-a-Judge**: Florence-2 (`microsoft/Florence-2-large`) evaluation with structured rubric prompt assessing fit realism, texture transfer, and boundary artifacts ($1.0–10.0$ rating scale + written reasoning).
- **Schema Compliance**: Exports directly to `evaluation_template_q4.csv` matching all required template columns without blending individual metrics into arbitrary scalars.

## Evaluation Axes & Rubric Breakdown

### 1. Garment Fidelity (OpenCLIP)
$$\text{Fidelity Score} = \frac{\mathbf{e}_{\text{garment}} \cdot \mathbf{e}_{\text{tryon}}}{\|\mathbf{e}_{\text{garment}}\| \|\mathbf{e}_{\text{tryon}}\|}$$
Compares deep semantic representations of input garment vs. output try-on area.

### 2. Identity Preservation (InsightFace / SSIM)
$$\text{Identity Score} = \text{SSIM}(\text{Face}_{\text{person}}, \text{Face}_{\text{tryon}})$$
Measures structural similarity across facial geometry, head hair boundary, and skin tone.

### 3. VLM-as-Judge Rubric Prompt
The Florence-2 model evaluates try-on results on a 1–10 scale based on 3 criteria:
- **Fit Realism**: Alignment of shoulders, sleeves, and torso drape without unreal distortion.
- **Texture & Pattern Transfer**: Fidelity of cloth weave, graphic prints, and color saturation.
- **Visual Artifacts**: Absence of boundary tearing, halo effects, or phantom limb bleeding.

## Exported Evaluation CSV Schema (`evaluation_template_q4.csv`)
Columns:
- `pair_id`: Pair identifier (`pair_01` to `pair_05`).
- `person_image`: Target person image file name.
- `garment_image`: Source garment image file name.
- `tryon_model`: Try-on model evaluated (`CatVTON`).
- `garment_fidelity_score`: OpenCLIP similarity score ($0.0000 - 1.0000$).
- `identity_preservation_score`: Face SSIM identity score ($0.0000 - 1.0000$).
- `vlm_judge_score`: VLM judge rating ($1.0 - 10.0$).
- `vlm_judge_reasons`: Detailed qualitative justification from VLM rubric.
- `artifacts_observed`: Specific visual artifacts or boundary notes.
- `notes`: Summary evaluation statement.

## Execution & Output Files
- CSV Saved Location: `Q4_evaluation/evaluation_template_q4.csv` and `Q4_evaluation/outputs/evaluation_template_q4.csv`.
- Models run sequentially with VRAM clearing to maintain peak allocation under **3.0 GB**.

