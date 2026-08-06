"""
Q5 Web Demo - Configuration module.
Contains Gradio interface parameters, guardrail test paths, and ETA calculation factors.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Drive & Checkpoint directories
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/xipl_tryon_checkpoints/Q5"
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Edge Case Test File Paths (Mandatory Rule A.5)
DATA_DIR = os.path.join(PROJECT_ROOT, "V5", "sample_files")
EDGE_CASES_DIR = os.path.join(DATA_DIR, "edge_cases")

NO_PERSON_TEST_FILE = os.path.join(EDGE_CASES_DIR, "no_person.jpg")
PERSON_SEATED_TEST_FILE = os.path.join(EDGE_CASES_DIR, "person_seated.jpg")
PERSON_SIDE_POSE_TEST_FILE = os.path.join(EDGE_CASES_DIR, "person_side_pose.jpg")

# Hardware & Precision Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Gradio Interface Settings
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860
SHARE_GRADIO = True

# Guardrail Thresholds
PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.5
ETA_BASE_SECONDS = 15.0  # Base processing time per pair at 768x1024
ETA_FALLBACK_FACTOR = 1.8 # Multiplier if resolution fallback triggered
