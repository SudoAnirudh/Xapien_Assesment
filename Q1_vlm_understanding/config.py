"""
Q1 VLM Understanding - Configuration module.
Contains paths, model parameters, precision, and validation settings for Florence-2 inference.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Drive & Checkpoint directories
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/xipl_tryon_checkpoints/Q1"
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "V5", "sample_files")
TEST_PAIRS_DIR = os.path.join(DATA_DIR, "test_pairs")
EDGE_CASES_DIR = os.path.join(DATA_DIR, "edge_cases")
SCHEMA_PATH = os.path.join(DATA_DIR, "sample_output_q1.json")

# Model settings
MODEL_ID = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Inference parameters
MAX_NEW_TOKENS = 512
NUM_BEAMS = 3
MAX_RETRIES = 2

# Allowed Vocabulary / Value Enums
POSE_CATEGORIES = ["front-facing", "side", "seated"]
GARMENT_TYPES = ["upper-body", "lower-body", "dress", "outerwear"]
