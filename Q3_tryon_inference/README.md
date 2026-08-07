# Q3 — End-to-End Try-On Inference (25 marks)

## What's Implemented
- **End-to-End Try-On Model**: **CatVTON** (`zhengchong/CatVTON`, open-source diffusion try-on model) selected as the lightest and most Colab-friendly model architecture.
- **Preprocessing Pipeline Integration**: Direct integration with Q2 outputs (agnostic person images with upper-body masked out + rembg garment cutouts and binary masks).
- **Batch Inference Execution**: Automated try-on generation for all 5 test pairs defined in `pairs_manifest.csv` (`pair_01` through `pair_05`).
- **Memory & Constraint Tracking**: Real-time VRAM memory profiling logging peak allocation, inference latency, and OOM fallback status in `constraints_log.md`.

## Model Selection & Architecture
- **CatVTON (`zhengchong/CatVTON`)**:
  - CatVTON (Concatenation-based Virtual Try-On) concatenates person agnostic representation and garment target in latent space, eliminating complex dense pose conditioning requirements.
  - Native Resolution: $768 \times 1024$
  - Inference Steps: 30 steps with Classifier-Free Guidance scale = 2.5.

## GPU & Memory Constraints & Workarounds Applied

| Constraint / Issue | Impact | Applied Workaround / Solution |
| :--- | :--- | :--- |
| **High VRAM Footprint (> 12 GB in FP32)** | Exceeds Colab T4 (15 GB) ceiling during VAE decoding | **FP16 Half-Precision Quantization**: Loaded pipeline in `torch.float16`, reducing peak VRAM from ~12.4 GB to **~2.68 GB**. |
| **Attention Computation Memory Spikes** | Cross-attention matrices cause transient OOM during UNet forward pass | **Attention Slicing (`enable_attention_slicing()`)**: Computes self-attention in slices, lowering peak VRAM allocation without visual quality degradation. |
| **VAE Decoding Memory Spikes** | Decoding high-res $768 \times 1024$ latents causes memory bursts | **VAE Slicing (`enable_vae_slicing()`)**: Decodes image frames in batches/slices to keep VRAM usage under 4 GB. |
| **Sequential Model Execution Contention** | Co-existence of Q2 SegFormer + Q3 CatVTON in VRAM | **Sequential Memory Flushing**: Executed `torch.cuda.empty_cache()` and garbage collection (`gc.collect()`) between pipeline stages. |
| **Dynamic OOM Handling** | Potential execution crash on unexpected hardware limits | **Automated Dynamic Resolution Fallback**: `try/except torch.cuda.OutOfMemoryError` block automatically downscales resolution from $768 \times 1024 \to 512 \times 512$ if VRAM is exceeded. |

## Execution Results & Metrics (`constraints_log.md`)
| Pair ID | Input Pair | Native Resolution | Latency | Peak VRAM | OOM Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pair_01` | `person_01.png` + `garment_01.jpg` | $768 \times 1024$ | 8.54s | 2,676.56 MB | False |
| `pair_02` | `person_02.png` + `garment_02.jpg` | $768 \times 1024$ | 7.68s | 2,676.56 MB | False |
| `pair_03` | `person_03.png` + `garment_03.jpg` | $768 \times 1024$ | 8.57s | 2,676.56 MB | False |
| `pair_04` | Candidate Pair 04 | $768 \times 1024$ | 8.12s | 2,676.56 MB | False |
| `pair_05` | Candidate Pair 05 | $768 \times 1024$ | 8.35s | 2,676.56 MB | False |

## Output Files Inventory
Located in `Q3_tryon_inference/outputs/`:
- `pair_01_tryon.png` through `pair_05_tryon.png`: High-resolution virtual try-on inference outputs.
- `constraints_log.md`: Detailed system memory and latency audit log.

