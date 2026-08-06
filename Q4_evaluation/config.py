"""
Q4 Automated Quality Evaluation - Configuration module.
Contains model specs, paths, metric normalization rules, and CSV output specs.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Drive & Checkpoint directories
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/xipl_tryon_checkpoints/Q4"
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Input Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "V5", "sample_files")
TRYON_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Q3_tryon_inference", "outputs")
CSV_TEMPLATE_PATH = os.path.join(DATA_DIR, "evaluation_template_q4.csv")
FINAL_CSV_PATH = os.path.join(BASE_DIR, "evaluation_template_q4.csv")

# Model IDs & Specs
OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
INSIGHTFACE_MODEL_NAME = "buffalo_l"
VLM_JUDGE_MODEL_ID = "microsoft/Florence-2-large"

# Hardware & Precision Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Metric Normalization Settings
CLIP_FIDELITY_RANGE = (0.0, 1.0)
FACE_IDENTITY_RANGE = (0.0, 1.0)
VLM_JUDGE_SCALE = (1, 10)  # Raw 1-10 rating scale reported alongside 0-1 metrics (Rule D9)
