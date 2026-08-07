# Q2 — Human Parsing & Garment Segmentation (20 marks)

## What's implemented
- **Human Parsing**: 20-class semantic parsing using SCHP / SegFormer-B2 Clothes model (`mattmdjaga/segformer_b2_clothes` trained on LIP/SCHP dataset).
- **Agnostic Person Representation**: Upper-body clothing mask generator that neutral-gray fills upper-garment regions (classes 5: Upper-clothes, 6: Dress, 7: Coat) while preserving hair (class 2), face (class 13), and arms (classes 14, 15).
- **Garment Segmentation**: Background removal using `rembg` (RMBG-1.4 / ISNet backend) for clean alpha matte & binary mask extraction from product photos.
- **Full Execution Coverage**: Batch processing across all person images (1–5), garment photos (1–5), and edge cases (`person_crossed_arms.jpg`, `person_seated.jpg`, `person_side_pose.jpg`, `garment_03`).

## Models & Technical Architecture
1. **Human Parsing Model**: `mattmdjaga/segformer_b2_clothes`
   - *Classes*: 20 LIP/SCHP categories (0: Background, 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses, 5: Upper-clothes, 6: Dress, 7: Coat, 8: Socks, 9: Pants, 10: Torso-skin, 11: Scarf, 12: Skirt, 13: Face, 14: Left-arm, 15: Right-arm, 16: Left-leg, 17: Right-leg, 18: Left-shoe, 19: Right-shoe).
   - *Precision*: FP32 / FP16 CUDA acceleration.
2. **Garment Background Removal**: `rembg` with `isnet-general-use` (RMBG-1.4 architecture)
   - Extracts high-resolution alpha mattes preserving fine features (thin straps, tassels, transparent meshes).

## Edge Case Handling Strategy
- **Hair over shoulders (`person_02.png`, `person_03.png`)**:
  - Mask construction explicitly subtracts `preserve_classes=[2, 13, 14, 15]` (hair, face, left arm, right arm) before and after morphological mask dilation, preventing hair strands over shoulders from being erroneously grayed out or erased.
- **Strappy sleeveless garment (`garment_03.jpg`)**:
  - Fine alpha thresholding (`alpha > 10`) combined with ISNet deep feature extraction retains delicate shoulder straps without erosion.
- **Crossed arms (`person_crossed_arms.jpg`)**:
  - Arm classes (14, 15) are explicitly excluded from upper-body garment mask, keeping crossed forearm skin intact over the torso area in `person_crossed_arms_agnostic.png` and `person_crossed_arms_parse.png`.

## Output Files Inventory
Located in `Q2_parsing_segmentation/outputs_extracted/outputs/`:
- `person_*_parse.png`: Colorized 20-class human parsing maps.
- `person_*_agnostic.png`: Upper-body clothing masked agnostic person images for diffusion try-on.
- `person_*_agnostic_mask.png`: Binary upper-body mask (white = masked garment region, black = preserved person).
- `garment_*_garment_cutout.png`: RGBA segmented garment product photos.
- `garment_*_garment_mask.png`: Binary garment silhouette masks.

## Constraints & Memory Optimization
- Models run sequentially with explicit CUDA cache flushes (`torch.cuda.empty_cache()`), maintaining peak GPU VRAM usage under **2.5 GB**.

