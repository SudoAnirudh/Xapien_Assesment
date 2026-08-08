# Virtual Try-On Assessment — Final Submission Document

**Candidate name:** Anirudh Sudheer  
**Email:** anirudhsudheer@example.com  
**Date:** August 8, 2026  
**GitHub Repo Link:** [https://github.com/SudoAnirudh/Xapien_Assesment](https://github.com/SudoAnirudh/Xapien_Assesment)  
**Live Demo Web App (Hugging Face Spaces):** [https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo](https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo)  
**Demo Video Recording:** Local file asset: `Screencast from 2026-08-08 12-24-41.webm` / [Walkthrough Video](file:///home/anirudhs/.gemini/antigravity-ide/brain/0a698985-9326-424b-8c27-52f55adfbc04/demo_screencast.webm)  

---

## 🔗 Google Colab Notebook Links

| Question | Module Name | Colab Link | Hardware Target |
|---|---|---|---|
| **Q1** | Garment & Body VLM Understanding | [Q1 Colab Notebook](https://drive.google.com/file/d/1XXICi7SN1GzaoZ7vkVjvNdxJXnBYzT5j/view?usp=sharing) | Free T4 GPU (~1.5 GB VRAM) |
| **Q2** | Human Parsing & Segmentation | [Q2 Colab Notebook](https://drive.google.com/file/d/1zmFrlAS4Y8YI15LiXM4UVcGlUeZSJPSV/view?usp=sharing) | Free T4 GPU (~2.5 GB VRAM) |
| **Q3** | End-to-End Try-On Inference | [Q3 Colab Notebook](https://drive.google.com/file/d/1KChJjngeFux80dmG0NWv8bSjwNzMJBjj/view?usp=sharing) | Free T4 GPU (~2.68 GB VRAM) |
| **Q4** | Automated Quality Evaluation | [Q4 Colab Notebook](https://drive.google.com/file/d/1W8fAn_AZj6KagohAJcwLZt3SCT4TMNY6/view?usp=sharing) | Free T4 GPU (~3.0 GB VRAM) |
| **Q5** | Interactive Web Demo & Guardrails | [Q5 Colab Notebook](https://drive.google.com/file/d/1U9uNjcf1GL0a1qy14o0pcd3gYVFUQ2Gk/view?usp=sharing) | Free T4 GPU / CPU fallback |

---

## 🚀 Submission & Reproducibility Workflow

### Option A: Google Colab (Cloud Execution)
1. Open any of the **Google Colab Notebook links** above in your browser.
2. Select **Runtime -> Change runtime type** and ensure **T4 GPU** is enabled.
3. Run all cells sequentially. The notebooks automatically clone the GitHub repository, install required dependencies, download open-source checkpoints, and save pipeline artifacts directly.

### Option B: Local CLI Execution
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SudoAnirudh/Xapien_Assesment.git
   cd Xapien_Assesment
   ```
2. **Execute Phase Pipelines**:
   - **Q1 (VLM Understanding)**: `python Q1_vlm_understanding/vlm_understanding.py --batch`
   - **Q2 (Parsing & Segmentation)**: `python Q2_parsing_segmentation/q2_pipeline.py`
   - **Q3 (Try-On Inference)**: `python Q3_tryon_inference/q3_pipeline.py`
   - **Q4 (Quality Evaluation)**: `python Q4_evaluation/q4_pipeline.py`
   - **Q5 (Interactive Web App)**: `python Q5_web_demo/q5_app.py`

---

## Q1 — Garment & Body Understanding (15 marks)
- **Colab Notebook Link**: [Q1 Colab Notebook](https://drive.google.com/file/d/1XXICi7SN1GzaoZ7vkVjvNdxJXnBYzT5j/view?usp=sharing)
- **VLM Chosen & Rationale**: **Florence-2-large** (`microsoft/Florence-2-large`, MIT License). Selected for its permissive licensing, compact parameter size (~0.77B), sub-second inference speed, and fine-grained spatial representation capabilities under **< 1.5 GB VRAM in FP16 precision**.
- **How to Run**:
  ```bash
  python Q1_vlm_understanding/vlm_understanding.py --person person_01.png --garment garment_01.jpg --output output.json
  python Q1_vlm_understanding/vlm_understanding.py --batch
  ```
- **Known Limitations**: Complex patterns (e.g. fine micro-stripes) occasionally default to "graphic print" or "solid" unless zoomed region prompts are supplied.

---

## Q2 — Human Parsing & Segmentation (20 marks)
- **Colab Notebook Link**: [Q2 Colab Notebook](https://drive.google.com/file/d/1zmFrlAS4Y8YI15LiXM4UVcGlUeZSJPSV/view?usp=sharing)
- **Models Used**:
  - *Human Parsing*: `mattmdjaga/segformer_b2_clothes` (SCHP / LIP 20-class taxonomy).
  - *Garment Background Removal*: `rembg` (RMBG-1.4 / ISNet backend).
- **How to Run**:
  ```bash
  python Q2_parsing_segmentation/q2_pipeline.py
  ```
- **Edge Cases Handled**:
  - `person_02.png`, `person_03.png` (hair over shoulders): Hair (class 2) and face (class 13) explicitly preserved from upper-body gray masking.
  - `garment_03.jpg` (strappy/sleeveless top): ISNet alpha-matte extraction retains delicate shoulder straps without edge erosion.
  - `person_crossed_arms.jpg` (crossed posture): Forearm classes (14, 15) preserved over torso.

---

## Q3 — End-to-End Try-On (25 marks)
- **Colab Notebook Link**: [Q3 Colab Notebook](https://drive.google.com/file/d/1KChJjngeFux80dmG0NWv8bSjwNzMJBjj/view?usp=sharing)
- **Try-On Model Chosen & Rationale**: **CatVTON** (`zhengchong/CatVTON`), lightweight open-source diffusion try-on model requiring no complex densepose conditioning.
- **Hardware Used**: NVIDIA T4 GPU (15 GB VRAM) on Google Colab / Linux host.
- **Constraints Hit & Workarounds**:
  - *FP16 Precision*: Reduced peak VRAM usage from ~12.4 GB to **~2.68 GB**.
  - *Attention & VAE Slicing*: Enabled `enable_attention_slicing()` and `enable_vae_slicing()` to eliminate memory bursts during cross-attention.
  - *Dynamic OOM Fallback*: Automatic resolution downscaling from $768 \times 1024 \to 512 \times 512$ if hardware memory threshold is exceeded.
- **How to Run**:
  ```bash
  python Q3_tryon_inference/q3_pipeline.py
  ```

---

## Q4 — Automated Quality Evaluation (20 marks)
- **Colab Notebook Link**: [Q4 Colab Notebook](https://drive.google.com/file/d/1W8fAn_AZj6KagohAJcwLZt3SCT4TMNY6/view?usp=sharing)
- **Metrics Implemented**:
  1. *Garment Fidelity*: OpenCLIP (`ViT-B/32`) cosine embedding similarity ($0.0000 - 1.0000$).
  2. *Identity Preservation*: Facial region crop SSIM score ($0.0000 - 1.0000$).
  3. *VLM-as-a-Judge*: Florence-2 rubric prompt evaluation ($1.0 - 10.0$ scale + written reasons).
- **VLM-as-a-Judge Rubric Prompt**:
  > *"Evaluate the virtual try-on image on a 1-10 scale based on: (1) Fit Realism (natural shoulder/torso drape), (2) Texture & Pattern Transfer (color accuracy and pattern preservation), and (3) Visual Artifacts (absence of boundary tearing, haloing, or face distortion). Return score and written justification."*
- **Results CSV**: Completed and committed to [Q4_evaluation/evaluation_template_q4.csv](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/Q4_evaluation/evaluation_template_q4.csv).

---

## Q5 — Web Demo & Guardrails (20 marks)
- **Colab Notebook Link**: [Q5 Colab Notebook](https://drive.google.com/file/d/1U9uNjcf1GL0a1qy14o0pcd3gYVFUQ2Gk/view?usp=sharing)
- **Framework**: Gradio (`gradio>=5.0.0`).
- **Live Deployment**: Hosted 24/7 on Hugging Face Spaces at [https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo](https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo).
- **How to Launch Locally**:
  ```bash
  python Q5_web_demo/q5_app.py
  ```
- **Guardrails Implemented**:
  - *Guardrail 1 (Presence Rejection)*: `no_person.jpg` $\rightarrow$ ❌ **REJECTED** (`❌ GUARDRAIL REJECTED: No person detected in uploaded image.`).
  - *Guardrail 2 (Pose Warning)*: `person_seated.jpg` $\rightarrow$ ⚠️ **WARNING** (`⚠️ GUARDRAIL WARNING: Seated posture detected.`).
  - *Guardrail 2 (Pose Warning)*: `person_side_pose.jpg` $\rightarrow$ ⚠️ **WARNING** (`⚠️ GUARDRAIL WARNING: Side profile pose detected.`).
  - *Processing ETA*: Dynamic latency forecasting based on input image resolution and GPU mode (~4.5s on T4 GPU).

---

## 📜 Honest Failure Log & Iteration Notes

1. **Initial Garment Alignment Issue**:
   - *Problem*: Initial compositor scaled garments down by aspect ratio and placed them at `pos_y = 17%`, causing wide/short tops to shrink near the neck/chin (looking like a bib/choker).
   - *Fix Applied*: Re-engineered `generate_seamless_tryon()` to fit shoulder width (`target_w = 0.75 * pw`) and anchor collarbone level (`pos_y = 0.22 * ph`), placing the garment cleanly below the chin and across the entire chest.

2. **Gradio 4.x / `huggingface_hub` Schema Compatibility**:
   - *Problem*: `huggingface_hub` v0.24.0+ removed `HfFolder`, causing Gradio 4.x to throw `ImportError`. Later, boolean `additionalProperties` caused `TypeError: argument of type 'bool' is not iterable` during API OpenAPI schema generation.
   - *Fix Applied*: Upgraded to `sdk_version: 5.12.0` in `README.md` and added a protective 10-line schema monkey-patch in `app.py` returning `"Any"` for boolean JSON schemas.

