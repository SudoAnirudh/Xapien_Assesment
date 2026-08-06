"""
Q3 Try-On Inference - Configuration module.
Contains model IDs, resolution settings, OOM fallback levels, and inference parameters for CatVTON.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Drive & Checkpoint directories
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/xipl_tryon_checkpoints/Q3"
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Input Data Directories (Preprocessed by Q2)
DATA_DIR = os.path.join(PROJECT_ROOT, "V5", "sample_files")
AGNOSTIC_IMAGES_DIR = os.path.join(PROJECT_ROOT, "Q2_parsing_segmentation", "outputs", "agnostic")
GARMENT_MASKS_DIR = os.path.join(PROJECT_ROOT, "Q2_parsing_segmentation", "outputs", "garment_masks")
MANIFEST_PATH = os.path.join(DATA_DIR, "test_pairs", "pairs_manifest.csv")

# Model Settings
MODEL_ID = "zhengchong/CatVTON"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Resolution Settings & OOM Fallback hierarchy
NATIVE_RESOLUTION = (768, 1024)  # Width, Height
FALLBACK_RESOLUTION = (512, 512)  # Reduced resolution fallback if full-res OOMs

# Inference Settings
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 2.5
SEED = 42

# Memory Optimization Flags (OOM Fallback sequence)
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
ENABLE_CPU_OFFLOAD = False  # Enabled if VRAM threshold exceeded
