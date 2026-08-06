# Decisions & Rules

This file is the single source of truth for locked technical decisions. If a phase needs to
deviate from something here, update this file in the same commit and explain why in the
"Deviation log" at the bottom — don't silently drift.

## A. Hard rules (non-negotiable, from the assessment brief)
1. **No paid/hosted inference APIs anywhere in the pipeline.** Every model call must be a local
   forward pass on weights downloaded from Hugging Face / GitHub releases.
2. **Every model used must have its license named in the README**, including non-commercial
   research licenses (IDM-VTON/OOTDiffusion/CatVTON) — acceptable for this assignment but must
   not be silently omitted.
3. **All 10 provided images run through Q1.** Not a sample — all 5 person + 5 garment images.
4. **The 3 specific edge cases named in the brief must appear by name in outputs/README**:
   `person_02`/`person_03` (hair over shoulders), `garment_03` (strappy sleeveless),
   `person_crossed_arms.jpg`. Skipping or silently averaging over them is a scoring risk.
5. **Q5 guardrails must be demonstrated on the exact 3 named files**: `no_person.jpg` → reject,
   `person_seated.jpg` → warning, `person_side_pose.jpg` → warning. No substitutes.
6. **A failure is documentation, not a bug to hide.** If a model OOMs or a pair fails try-on,
   the run must still produce a log entry (model, input, error, attempted fix) rather than being
   dropped from the outputs silently.

## B. Locked technical decisions
| # | Decision | Rationale | Owner phase |
|---|---|---|---|
| D1 | VLM = Florence-2-large, reused for Q1 and Q4c | Smallest of the 4 allowed VLMs; avoids loading two large VLMs on a single T4 | Q1, Q4 |
| D2 | Parsing = SCHP over SAM+GroundingDINO | Single model, purpose-built 20-class parser vs. a 2-model chain that still needs manual class-mapping to get "upper clothes" | Q2 |
| D3 | Garment cutout = rembg (RMBG-1.4) | One dependency, no bounding-box prompting needed since garment photos are already flat/isolated-background | Q2 |
| D4 | Try-on model = CatVTON | Brief explicitly names it as lightest/most Colab-friendly; confirmed ~4GB VRAM footprint at 768x1024 | Q3 |
| D5 | Precision = fp16 everywhere inference runs on GPU | T4 has 15GB VRAM; fp32 risks OOM on CatVTON's diffusion UNet + SCHP + Florence-2 loaded in the same session | Q2, Q3 |
| D6 | Models are loaded **sequentially, not concurrently** — load, run phase, `del` + `torch.cuda.empty_cache()`, load next | Prevents cumulative VRAM pressure across Q1→Q2→Q3→Q4 when run in one notebook | All |
| D7 | Intermediate outputs (parsing maps, agnostic images, masks) are saved to disk/Drive after every phase | Colab free tier disconnects unpredictably; re-running from scratch each time is wasted budget | All |
| D8 | JSON outputs are schema-validated in code (not just eyeballed) against `sample_output_q1.json` | VLM free-text generation drifts from schema; a validator catches this before it becomes a Q1 grading deduction | Q1 |
| D9 | Q4 CLIP/InsightFace scores are normalized to 0–1 and reported alongside the raw VLM 1–10 judge score, not blended into one number | Keeps the three axes independently inspectable, matches the brief's "three axes" framing | Q4 |
| D10 | Web demo = Gradio, not Streamlit | Easier to guarantee deterministic guardrail behavior on 3 named test files, cleaner screen-recording flow for the demo video | Q5 |

## C. Coding conventions
- One folder per question (`Q1_vlm_understanding/`, `Q2_parsing_segmentation/`, ...), matching
  the repo layout required in §6 of the brief.
- Every phase folder gets its own `README.md` with: what's implemented, how it works, why it
  took the time it did, and a "gotchas" section — see the template in IMPLEMENTATION_PLAN.md.
- No function longer than ~40 lines without a comment breaking it into named steps — this is
  what makes "explain every line in the interview" actually survivable.
- Config (paths, model names, resolution, thresholds) lives in one `config.py` per phase, not
  scattered as magic strings — makes the "reduced resolution" OOM fallback a one-line change.

## D. Deviation log
*(append here if any decision above changes mid-build, with date + reason)*
- — none yet —
