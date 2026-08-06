# Q3 — End-to-End Try-On Inference

## What's implemented
- CatVTON model inference wrapper running in fp16 precision.
- Preprocessing integration wiring Q2's agnostic person images and rembg garment masks into CatVTON.
- Batch inference execution over all 5 test pairs (`pair_01`–`pair_05`, including 3 provided + 2 self-sourced pairs).
- OOM monitoring and fallback logger generating `constraints_log.md`.

## How it works
- **Input**: Agnostic person image + garment cutout image + garment mask (from Q2).
- **Inference**: CatVTON lightweight diffusion forward pass at 768x1024 resolution.
- **Memory Management**: fp16 precision + attention slicing; resolution fallback hierarchy (768x1024 → 512x512) if VRAM threshold exceeded.
- **Output**: 5 try-on result images saved to disk/Drive + `constraints_log.md`.

## Why it took this long
- Setup & CatVTON weight loading (~30 min).
- Q2 preprocessing integration & channel alignment (~2 hrs).
- Batch inference execution and VRAM profiling (~3 hrs).
- Fallback sequence testing and constraints documentation (~2.5 hrs).

## Gotchas / edge cases
- Resolution mismatches between Q2 output and Q3 input: Resolved via explicit scaling in config.
- VRAM profiling: Active monitoring with `torch.cuda.max_memory_allocated()` to catch spikes.

## Constraints hit
- **License**: CatVTON non-commercial research license noted.
- **Memory**: fp16 precision maintained ~4.0 GB VRAM footprint on T4 GPU.
- **Errors**: Documented in `constraints_log.md`.
