# Implementation Plan

Budget: 3 days (~18–20 working hours) against the brief's own per-question estimates
(Q1 0.5d, Q2 0.75d, Q3 1d, Q4 0.5d, Q5 0.75d = 3.5d nominal — compressed by running Q1/Q2 setup
in parallel with environment prep, see Phase 0).

Every phase below ends with a **per-phase README** filled from the template in §8. That template
is the "what / how / why-it-took-this-long / minute details" documentation the brief rewards
under "code quality and documented trade-offs" (20% of each question's score).

---

## Phase 0 — Environment & Repo Scaffold (2 hrs)
**Goal:** a Colab notebook + GitHub repo that can install every dependency for all 5 phases
without conflict, before any model code is written.

**Tasks**
1. Create repo with folders: `Q1_vlm_understanding/`, `Q2_parsing_segmentation/`,
   `Q3_tryon_inference/`, `Q4_evaluation/`, `Q5_web_demo/`, `docs/`, root `README.md`.
2. Unzip and inspect `sample_files.zip` — confirm all 10 test images, `pairs_manifest.csv`,
   `edge_cases_manifest.csv`, `sample_output_q1.json`, `evaluation_template_q4.csv` are present
   and match the brief's description exactly (catches a corrupted/incomplete zip early).
3. Pin dependency versions per phase in separate `requirements.txt` files — sharing one global
   file across CatVTON (diffusers-pinned) and Florence-2 (transformers-pinned) is a common
   version-conflict trap on Colab.
4. Source `pair_04`/`pair_05` (self-sourced person+garment images) — front-facing, upper body
   visible, royalty-free — and add them to `test_pairs/` with matching numbering.
5. Mount Google Drive in the Colab notebook for checkpointing (D7 in DECISIONS_AND_RULES.md).

**Why this takes 2 hrs:** dependency pinning across 4 different model families (transformers,
diffusers, detectron-style parsing, ONNX-based rembg) is where Colab sessions actually die —
front-loading this avoids re-debugging it inside every later phase.

---

## Phase 1 — Q1: VLM Garment & Body Understanding (4 hrs)
**Goal:** Florence-2 script → schema-valid JSON for all 10 test images + 3 edge-case person
images (side, seated — crossed-arms is Q2's edge case, not Q1's).

**Tasks**
1. Load Florence-2-large in fp16, confirm VRAM footprint on T4 (~1.5GB expected).
2. Write the structured-extraction prompt (see PROMPT_GUIDE.md §1) — few-shot against
   `sample_output_q1.json`'s exact key names and value vocabulary (e.g. pose category must be
   one of `front-facing`/`side`/`seated`, not free text).
3. Write a JSON schema validator; on validation failure, retry once with a stricter
   "return ONLY valid JSON matching these exact keys" repair prompt before failing loudly.
4. Run on all 13 images, save one JSON per image + `results_summary.json`.
5. Manually spot-check 3–4 outputs against the actual images; note any misclassification in the
   phase README's "gotchas" section rather than silently re-running until it looks right.

**Why 4 hrs, not the brief's 0.5 day (~4h) — matches:** prompt iteration to hit exact schema
compliance is the time sink here, not inference speed (Florence-2 is fast).

---

## Phase 2 — Q2: Human Parsing & Garment Segmentation (6 hrs)
**Goal:** parsing map + agnostic image per person image (incl. `person_02`, `person_03`,
`person_crossed_arms.jpg`), clean garment mask per garment image (incl. `garment_03`).

**Tasks**
1. Load SCHP, run on all 5 person images + `person_crossed_arms.jpg` → 20-class parsing maps.
2. Build the "agnostic" image: mask out the `upper-clothes` (+ `dress`/`coat` if present) class
   from the parsing map, gray-fill or inpaint the region — confirm this matches the standard
   diffusion try-on input format CatVTON expects (check its own repo's agnostic-image spec).
3. Handle hair-over-shoulder (`person_02`, `person_03`): verify SCHP's `hair` class doesn't get
   swallowed into the masked garment region — this is the actual edge case being tested, not
   just "does it run."
4. Handle crossed arms: verify the arm doesn't get misclassified as part of the masked-out
   upper-clothes region (a known SCHP failure mode) — document whatever the actual result is.
5. Run rembg on all 5 garment images, confirm clean alpha matte on `garment_03` (strappy/
   sleeveless — thin straps are the actual stress case for background removal).
6. Checkpoint all outputs to Drive.

**Why 6 hrs, above the brief's 0.75d (~6h) estimate — matches:** the 3 named edge cases each
need a manual before/after visual check, not just "the script exited 0."

---

## Phase 3 — Q3: End-to-End Try-On Inference (8 hrs)
**Goal:** CatVTON run on all 5 official pairs, Q2 outputs wired in as preprocessing, every
constraint hit logged.

**Tasks**
1. Load CatVTON, run a single pair first as a smoke test before batching all 5.
2. Wire Q2's parsing map + agnostic image + garment mask as CatVTON's expected inputs — this is
   the actual integration risk (format/resolution mismatches between Q2 and Q3's assumptions).
3. Run all 5 pairs at native 768x1024; monitor VRAM via `torch.cuda.max_memory_allocated()`.
4. If OOM: apply fp16 (if not already) → attention slicing → CPU offload for VAE, in that
   order, re-running only the failed pair each time, logging what each step changed.
5. If still OOM at full resolution: fall back to 512x512, document the visible quality loss.
6. Write `constraints_log.md`: every constraint hit, exact error message, fix applied, and
   before/after (even "no constraints hit" is a documented line, not silence).

**Why 8 hrs, above the brief's 1d (~8h) — matches:** this is the highest-risk integration point
(two independently-developed pipelines meeting), budgeted at parity with the brief's own
weighting (25/100 marks, the largest single question).

---

## Phase 4 — Q4: Automated Quality Evaluation (4 hrs)
**Goal:** CSV scoring all 5 Q3 outputs on 3 axes, rubric prompt documented.

**Tasks**
1. OpenCLIP: embed input garment image + garment region of output (crop via Q2's mask or a
   fresh detection on the output), cosine similarity → fidelity score.
2. InsightFace: face-region crop from input person + output, SSIM and embedding distance →
   identity score.
3. Write the VLM-judge rubric prompt (PROMPT_GUIDE.md §2) covering fit realism, artifacts,
   texture transfer; run through Florence-2 (reused from Q1), parse 1–10 + reasons.
4. Fill `evaluation_template_q4.csv` for all 5 pairs, matching its existing column headers
   exactly — don't rename or reorder columns even if a different name reads better.
5. Sanity-check: does the pair with the worst Q3 constraints-log outcome (if any) also score
   worst here? If not, note the discrepancy rather than smoothing it over.

**Why 4 hrs, matches brief's 0.5d (~4h):** three independent, off-the-shelf scoring methods —
no novel model work, mostly wiring + one rubric-prompt iteration cycle.

---

## Phase 5 — Q5: Mini Try-On Web Demo (6 hrs)
**Goal:** Gradio app wrapping Q2→Q3→Q4, guardrails demonstrated on the 3 named edge-case files.

**Tasks**
1. Build the pipeline function: person image + garment image in → parsing/agnostic/mask (Q2) →
   try-on output (Q3) → 3-axis scores + Q1 attributes (Q4/Q1) out, as one callable.
2. Guardrail 1 — no person detected: run SCHP (or a lightweight person-detector) first; if no
   person class pixels found, reject before running Q3 at all (saves the ETA, not just cosmetic).
3. Guardrail 2 — unsuitable pose: reuse Q1's pose classification; if `side` or `seated`, show a
   warning but allow continue (per brief: "warn," not "reject," for these two).
4. ETA estimate: a rough function of image resolution + which fallback path Q3's constraints
   log showed was needed (full-res vs 512 fallback), not a hardcoded number.
5. Test all 3 named files live (`no_person.jpg`, `person_seated.jpg`, `person_side_pose.jpg`)
   and confirm expected behavior before recording.
6. Record the ≤5 min demo video: pipeline running end-to-end on one normal pair, then all 3
   guardrail files firing.

**Why 6 hrs, matches brief's 0.75d (~6h):** UI wiring is fast; the time is in guardrail testing
and getting a clean, un-edited demo recording of all 3 required cases.

---

## Phase 6 — Submission Packaging (2 hrs)
**Tasks**
1. Copy `SUBMISSION_TEMPLATE.md` → root `README.md`, fill every section (model choices +
   licenses, per-question summary, known issues, Colab links).
2. Confirm `evaluation_template_q4.csv` is committed with real data, not the template stub.
3. Final clean-checkout test: clone fresh, follow only the README, confirm Q1–Q5 all run.
4. Upload demo video, link Colab notebooks with outputs saved (not just code).

---

## §8 — Per-phase README template (used for Q1–Q5, filled by the Docs Agent)

```markdown
# <Question N> — <short title>

## What's implemented
- Bullet list of concrete deliverables (scripts, outputs, files) — no vague "handled X."

## How it works
- Pipeline walkthrough: input → each transform step → output, naming the actual model/function
  at each step. Assume the reader is the interviewer who will ask "walk me through this."

## Why it took this long
- Honest breakdown: setup/dependency time vs. prompt-or-parameter iteration vs. debugging a
  specific edge case vs. actual inference runtime. Ties back to IMPLEMENTATION_PLAN.md's estimate.

## Gotchas / edge cases
- Each named edge case from the brief, what happened when it was tested, and the fix (or the
  honest "not fully solved, here's why" if it wasn't).

## Constraints hit
- GPU/memory/timeout/license issues, exact error text, workaround applied. "None" is a valid,
  explicitly-stated entry — don't omit the section if nothing broke.
```
