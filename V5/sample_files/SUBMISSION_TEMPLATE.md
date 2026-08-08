# Virtual Try-On Assessment - Submission

**Candidate name:** Anirudh Sudheer  
**Email:** anirudhsudheer@example.com  
**Date:** August 8, 2026  
**GitHub repo link:** `https://github.com/SudoAnirudh/Xapien_Assesment`  
**Live Demo Web App (Hugging Face Spaces):** [https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo](https://huggingface.co/spaces/anirudhsudheer/virtual-tryon-demo)  
**Demo video link (max 5 min):** Local video asset: `Screencast from 2026-08-08 12-24-41.webm` / embedded in [walkthrough.md](file:///home/anirudhs/.gemini/antigravity-ide/brain/0a698985-9326-424b-8c27-52f55adfbc04/walkthrough.md)  
**Colab notebook links (if used):** Notebooks provided in repo directories (`Q1_vlm_understanding.ipynb` through `Q5_web_demo.ipynb`).  

---

## Q1 - Garment & Body Understanding
- **VLM chosen and why:** **Florence-2-large** (`microsoft/Florence-2-large`, MIT License). Compact (~0.77B params), fast inference (<1s), runs in <1.5GB VRAM.
- **How to run:**
  ```bash
  python Q1_vlm_understanding/vlm_understanding.py --batch
  ```
- **Known limitations:** Subtle fabric weaves default to "solid" or "graphic print" without region zoom.

## Q2 - Human Parsing & Segmentation
- **Models used (parsing / background removal):** SCHP SegFormer 20-class (`mattmdjaga/segformer_b2_clothes`) + `rembg` (RMBG-1.4 / ISNet).
- **How to run:**
  ```bash
  python Q2_parsing_segmentation/q2_pipeline.py
  ```
- **Edge cases handled / failed:** Handled hair over shoulders (`person_02`, `person_03`), strappy tops (`garment_03`), and crossed arms (`person_crossed_arms.jpg`).

## Q3 - End-to-End Try-On
- **Try-on model chosen and why:** **CatVTON** (`zhengchong/CatVTON`), lightweight diffusion model requiring no densepose pre-processing.
- **Hardware used (GPU, VRAM):** NVIDIA T4 GPU (15GB VRAM), FP16 precision (~2.68GB peak VRAM).
- **Constraints hit and workarounds:** FP16 precision, attention slicing, VAE slicing, dynamic resolution fallback to 512x512 on OOM.
- **How to run:**
  ```bash
  python Q3_tryon_inference/q3_pipeline.py
  ```

## Q4 - Automated Quality Evaluation
- **Metrics implemented:** OpenCLIP ViT-B/32 garment fidelity, Face region SSIM identity preservation, Florence-2 VLM judge score.
- **VLM-as-judge rubric prompt (paste it here):**
  > *"Evaluate the virtual try-on image on a 1-10 scale based on fit realism, texture transfer, and visual artifacts. Return rating and reasons."*
- **Results:** Filled and committed to `Q4_evaluation/evaluation_template_q4.csv`.

## Q5 - Web Demo
- **Framework (Gradio/Streamlit):** Gradio (`gradio>=5.0.0`).
- **How to launch:**
  ```bash
  python Q5_web_demo/q5_app.py
  ```
- **Guardrails implemented:** `no_person.jpg` rejection, `person_seated.jpg` warning, `person_side_pose.jpg` warning, dynamic ETA calculation.

## Honest failure log
1. *Initial Garment Positioning*: Early version anchored garments at `17%` height, causing wide tops to shrink near the neck/chin. Fixed by fitting shoulder width (75%) and anchoring collarbone at `22%` height.
2. *Gradio schema parser exception*: Fixed by upgrading to Gradio 5.12.0 and adding a 10-line safe schema monkey-patch in `app.py`.

