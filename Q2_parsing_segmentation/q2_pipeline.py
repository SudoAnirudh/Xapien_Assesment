"""
Q2 Parsing & Segmentation Pipeline
Generates 20-class human parsing maps, upper-body agnostic images, and rembg garment cutout masks.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Import config
import config

def get_parsing_model():
    """
    Loads human parsing model using HuggingFace SegFormer b2 clothes model (20-class LIP/SCHP schema).
    """
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    model_id = "mattmdjaga/segformer_b2_clothes"
    print(f"[INFO] Loading Human Parsing Model: {model_id} on {config.DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(config.DEVICE)
    model.eval()
    return processor, model

def parse_human_image(image_path, processor, model):
    """
    Runs 20-class human parsing on a person image.
    Returns:
        image: Original PIL Image
        parsing_map: numpy array (H, W) with class IDs 0..19
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Upsample logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    
    return image, pred_seg

def create_agnostic_image(image, parsing_map, mask_classes=[5, 6, 7], preserve_classes=[2, 13, 14, 15]):
    """
    Creates an upper-body agnostic person image by masking upper clothes (5: Upper-clothes, 6: Dress, 7: Coat),
    while explicitly preserving hair (2), face (13), and arms (14, 15) to prevent swallowing hair/arms over shoulders.
    The masked region is filled with neutral gray (128, 128, 128).
    """
    img_np = np.array(image).copy()
    
    # Initial mask: upper clothing classes
    garment_mask = np.isin(parsing_map, mask_classes)
    
    # Exclude preserved classes (hair, face, left arm, right arm)
    preserved_mask = np.isin(parsing_map, preserve_classes)
    final_mask = garment_mask & (~preserved_mask)
    
    # Morphological dilation on the mask for seamless try-on boundary
    kernel = np.ones((7, 7), np.uint8)
    final_mask_dilated = cv2.dilate(final_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    
    # Preserve hair/face/arms even after dilation
    final_mask_dilated = final_mask_dilated & (~preserved_mask)
    
    # Fill upper body clothing region with neutral gray (128, 128, 128)
    img_np[final_mask_dilated] = [128, 128, 128]
    
    agnostic_img = Image.fromarray(img_np)
    binary_mask_img = Image.fromarray((final_mask_dilated * 255).astype(np.uint8))
    
    return agnostic_img, binary_mask_img

def run_rembg_garment_cutout(garment_path):
    """
    Runs rembg (RMBG-1.4 / u2net backend) on a garment image to extract clean alpha matte & garment mask.
    """
    from rembg import remove, new_session
    
    garment_img = Image.open(garment_path).convert("RGB")
    
    # Remove background -> RGBA image
    session = new_session("isnet-general-use")
    output_rgba = remove(garment_img, session=session)
    
    # Extract alpha channel as binary garment mask
    np_rgba = np.array(output_rgba)
    alpha_channel = np_rgba[:, :, 3]
    garment_mask = (alpha_channel > 10).astype(np.uint8) * 255
    garment_mask_img = Image.fromarray(garment_mask)
    
    return output_rgba, garment_mask_img

def run_q2_pipeline():
    """
    Main Q2 execution pipeline.
    """
    print("[INFO] Starting Q2 Human Parsing & Garment Segmentation Pipeline...")
    os.makedirs(config.LOCAL_OUTPUT_DIR, exist_ok=True)
    
    # Drive checkpoint directory if mounted
    drive_checkpoint = getattr(config, "DRIVE_CHECKPOINT_DIR", "/content/drive/MyDrive/xipl_tryon_checkpoints/Q2")
    
    # Step 1: Load parsing model
    processor, model = get_parsing_model()
    
    # Gather person images
    person_tasks = []
    if os.path.exists(config.PERSON_IMAGES_DIR):
        for fname in os.listdir(config.PERSON_IMAGES_DIR):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                person_tasks.append((fname, os.path.join(config.PERSON_IMAGES_DIR, fname)))
    
    if os.path.exists(config.EDGE_CASES_DIR):
        for fname in os.listdir(config.EDGE_CASES_DIR):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and "person" in fname:
                person_tasks.append((fname, os.path.join(config.EDGE_CASES_DIR, fname)))
                
    print(f"[INFO] Found {len(person_tasks)} person images for parsing & agnostic generation.")
    
    for fname, fpath in person_tasks:
        base_name = os.path.splitext(fname)[0]
        print(f" -> Processing Person Image: {fname}...")
        
        orig_img, parse_map = parse_human_image(fpath, processor, model)
        agnostic_img, agnostic_mask = create_agnostic_image(orig_img, parse_map)
        
        # Save color parsing map visual
        parse_vis = (parse_map * (255 // 20)).astype(np.uint8)
        Image.fromarray(parse_vis).save(os.path.join(config.LOCAL_OUTPUT_DIR, f"{base_name}_parse.png"))
        
        # Save agnostic image & mask
        agnostic_img.save(os.path.join(config.LOCAL_OUTPUT_DIR, f"{base_name}_agnostic.png"))
        agnostic_mask.save(os.path.join(config.LOCAL_OUTPUT_DIR, f"{base_name}_agnostic_mask.png"))
        
    # Step 2: Garment background removal via rembg
    garment_tasks = []
    if os.path.exists(config.GARMENT_IMAGES_DIR):
        for fname in os.listdir(config.GARMENT_IMAGES_DIR):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                garment_tasks.append((fname, os.path.join(config.GARMENT_IMAGES_DIR, fname)))
                
    print(f"[INFO] Found {len(garment_tasks)} garment images for rembg segmentation.")
    
    for fname, fpath in garment_tasks:
        base_name = os.path.splitext(fname)[0]
        print(f" -> Processing Garment Cutout: {fname}...")
        
        rgba_img, mask_img = run_rembg_garment_cutout(fpath)
        rgba_img.save(os.path.join(config.LOCAL_OUTPUT_DIR, f"{base_name}_garment_cutout.png"))
        mask_img.save(os.path.join(config.LOCAL_OUTPUT_DIR, f"{base_name}_garment_mask.png"))
        
    print(f"\n[SUCCESS] Q2 Pipeline finished successfully! Outputs saved to: {config.LOCAL_OUTPUT_DIR}")
    
    # Backup to Drive if available
    try:
        from google.colab import drive
        import shutil
        os.makedirs(drive_checkpoint, exist_ok=True)
        shutil.copytree(config.LOCAL_OUTPUT_DIR, drive_checkpoint, dirs_exist_ok=True)
        print(f"[INFO] Q2 outputs successfully backed up to Google Drive: {drive_checkpoint}")
    except Exception:
        pass
    
    # Clean up GPU memory
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU memory cleared.")

if __name__ == "__main__":
    run_q2_pipeline()
