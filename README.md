# XIPL SDE Intern Technical Assessment — Open-Source Virtual Try-On Pipeline

An end-to-end, open-source AI virtual try-on system designed to run cleanly on free-tier Google Colab (T4 GPU, 15GB VRAM) without relying on any paid or hosted APIs.

---

## 📌 Architecture & Model Roster

| Phase | Question | Model Choice | License | VRAM Target | Key Outputs |
|---|---|---|---|---|---|
| **Q1** | VLM Understanding | **Florence-2-large** | MIT | ~1.5 GB | Structured JSON attribute descriptions for 10 test + 3 edge-case images |
| **Q2** | Parsing & Segmentation | **SCHP** (human parsing) + **rembg (RMBG-1.4)** (garment cutout) | Open-Source | ~2.5 GB | 20-class parsing maps, upper-body agnostic images, clean garment masks |
| **Q3** | Try-On Inference | **CatVTON** | Non-Commercial Research | ~4.0 GB | 5 high-resolution virtual try-on images (`pair_01`–`pair_05`) + `constraints_log.md` |
| **Q4** | Automated Evaluation | **OpenCLIP** (garment fidelity) + **InsightFace** (identity) + **Florence-2** (VLM judge) | Open-Source / Research | ~3.0 GB | 3-axis quality scores exported to `evaluation_template_q4.csv` |
| **Q5** | Interactive Web Demo | **Gradio** | Apache 2.0 | Integrated | Live try-on UI with guardrails (`no_person` reject, pose warnings, ETA estimate) |

---

## 📁 Repository Structure

```
├── docs/                        # Planning documentation & specs
│   ├── PRD.md                   # Product requirements document
│   ├── DECISIONS_AND_RULES.md   # Locked technical decisions & deviation log
│   ├── IMPLEMENTATION_PLAN.md   # Phase-by-phase implementation roadmap
│   ├── PROMPT_GUIDE.md          # Structured prompts & rubrics
│   └── AGENTS.md                # Agent roles & boundaries
├── Q1_vlm_understanding/        # Phase 1: Garment & body VLM understanding
│   ├── config.py                # Configuration & hyperparameters
│   ├── requirements.txt         # Phase 1 dependencies
│   ├── README.md                # Phase 1 documentation
│   └── Q1_vlm_understanding.ipynb
├── Q2_parsing_segmentation/     # Phase 2: Human parsing & garment segmentation
│   ├── config.py                # Configuration & hyperparameters
│   ├── requirements.txt         # Phase 2 dependencies
│   ├── README.md                # Phase 2 documentation
│   └── Q2_parsing_segmentation.ipynb
├── Q3_tryon_inference/          # Phase 3: CatVTON inference pipeline
│   ├── config.py                # Configuration & hyperparameters
│   ├── requirements.txt         # Phase 3 dependencies
│   ├── README.md                # Phase 3 documentation
│   └── Q3_tryon_inference.ipynb
├── Q4_evaluation/               # Phase 4: 3-Axis automated quality evaluation
│   ├── config.py                # Configuration & hyperparameters
│   ├── requirements.txt         # Phase 4 dependencies
│   ├── README.md                # Phase 4 documentation
│   └── Q4_evaluation.ipynb
├── Q5_web_demo/                 # Phase 5: Gradio web application with guardrails
│   ├── config.py                # Configuration & hyperparameters
│   ├── requirements.txt         # Phase 5 dependencies
│   ├── README.md                # Phase 5 documentation
│   └── Q5_web_demo.ipynb
└── README.md                    # Root repository index
```

---

## ⚙️ Key Technical Rules & Decisions

- **No Paid APIs**: 100% open-source forward passes on local model weights.
- **Independent Requirements**: Dependencies are isolated per phase folder to prevent Colab conflicts between `transformers` (Florence-2) and `diffusers` (CatVTON).
- **Sequential Model Execution**: Models are loaded sequentially and explicitly freed (`del model; torch.cuda.empty_cache()`) to prevent VRAM cumulative pressure.
- **Drive Checkpointing**: All intermediate maps, masks, and outputs are saved to Google Drive to handle ephemeral Colab disconnections.
- **Strict Schema & CSV Compliance**: JSON outputs in Q1 are schema-validated; Q4 evaluation scores strictly match the required template headers.

---

## 📖 Planning Documentation Links

- [PRD.md](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/docs/PRD.md)
- [DECISIONS_AND_RULES.md](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/docs/DECISIONS_AND_RULES.md)
- [IMPLEMENTATION_PLAN.md](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/docs/IMPLEMENTATION_PLAN.md)
- [PROMPT_GUIDE.md](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/docs/PROMPT_GUIDE.md)
- [AGENTS.md](file:///home/anirudhs/Documents/Boredom/Xapien_Assesment/docs/AGENTS.md)
