# Q2 — Human Parsing & Garment Segmentation

## What's implemented
- SCHP (Self-Correction Human Parsing) model integration for 20-class human parsing maps.
- Agnostic image generator (masks upper-clothes class, preserves face/hair/arms, gray-fills region).
- rembg (RMBG-1.4 backend) wrapper for clean garment cutout mask generation.
- Processing pipeline for all 5 person images + `person_crossed_arms.jpg`, and all 5 garment images incl. `garment_03`.

## How it works
- **Human Parsing**: SCHP generates 20-class pixel-wise segmentation map.
- **Agnostic Image**: Upper-clothing classes (5, 6, 7) are masked out; preserved classes (hair, face, arms) are protected.
- **Garment Cutout**: rembg executes alpha matte extraction on product images.
- **Output**: Parsing map, agnostic image, and binary garment mask saved for Q3 try-on input.

## Why it took this long
- Setup & SCHP model weight download (~20 min).
- Mask logic verification for hair-over-shoulder and crossed arms (~2 hrs).
- rembg integration and alpha matte tuning (~1 hr).
- Pipeline validation across test pairs and edge cases (~1.5 hrs).

## Gotchas / edge cases
- `person_02` / `person_03` (hair over shoulders): Ensuring SCHP hair class (2) is preserved and not swallowed into upper-clothes mask.
- `person_crossed_arms.jpg`: Validating arm class (14, 15) preservation across crossed posture.
- `garment_03` (strappy/sleeveless): Verifying fine strap detail retention during rembg background removal.

## Constraints hit
- **Memory**: SCHP and rembg run sequentially; total peak VRAM remains under ~2.5 GB.
- **Errors**: None encountered during initial setup.
