"""
Q2 Parsing & Segmentation - Configuration module.
Contains paths, class mappings for SCHP, rembg parameters, and resolution settings.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Drive & Checkpoint directories
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/xipl_tryon_checkpoints/Q2"
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "V5", "sample_files")
PERSON_IMAGES_DIR = os.path.join(DATA_DIR, "test_pairs", "person")
GARMENT_IMAGES_DIR = os.path.join(DATA_DIR, "test_pairs", "garment")
EDGE_CASES_DIR = os.path.join(DATA_DIR, "edge_cases")

# Model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Output Resolutions
DEFAULT_RESOLUTION = (768, 1024)  # Width, Height for CatVTON pipeline

# SCHP 20-Class Labels & Target Mask Classes
SCHP_CLASSES = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses",
    5: "Upper-clothes", 6: "Dress", 7: "Coat", 8: "Socks", 9: "Pants",
    10: "Jumpsuits", 11: "Scarf", 12: "Skirt", 13: "Face", 14: "Left-arm",
    15: "Right-arm", 16: "Left-leg", 17: "Right-leg", 18: "Left-shoe", 19: "Right-shoe"
}

# Classes to mask out for upper-body agnostic image generation
UPPER_BODY_MASK_CLASSES = [5, 6, 7]  # Upper-clothes, Dress, Coat
PRESERVED_CLASSES = [2, 13, 14, 15]   # Hair, Face, Left-arm, Right-arm (prevent swallowing hair/arms)
