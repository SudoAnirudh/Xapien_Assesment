---
title: AI Virtual Try-On System (CatVTON & Florence-2)
emoji: 👕
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: false
license: mit
---

# 👕 Open-Source AI Virtual Try-On Web App

This Hugging Face Space hosts an end-to-end open-source Virtual Try-On application built with **CatVTON**, **Florence-2-large**, **SCHP SegFormer**, and **rembg**.

## 🚀 Features & Pipeline Stages

1. **Q1 — VLM Garment & Body Understanding**: Florence-2 extracts structured garment attributes (type, sleeve length, neckline, color, pattern) and detects body pose.
2. **Q2 — Human Parsing & Garment Segmentation**: SCHP SegFormer 20-class parsing masks upper-clothing while preserving hair, face, and arms. `rembg` (RMBG-1.4) generates clean garment cutouts.
3. **Q3 — End-to-End Try-On Inference**: CatVTON diffusion try-on pipeline in FP16 precision.
4. **Q4 — 3-Axis Automated Quality Evaluation**:
   - *Garment Fidelity*: OpenCLIP ViT-B/32 cosine embedding similarity score.
   - *Identity Preservation*: Facial region SSIM / InsightFace score.
   - *VLM Judge*: Florence-2 rubric evaluation score (1–10 scale + reasons).
5. **Q5 — Real-time Guardrails & UI**:
   - **Rejection Guardrail**: Rejects uploads missing a human subject (`no_person.jpg`).
   - **Warning Guardrail**: Alerts users on non-front-facing poses (`person_seated.jpg`, `person_side_pose.jpg`).
   - **ETA Calculation**: Estimates processing latency based on GPU/CPU environment.
