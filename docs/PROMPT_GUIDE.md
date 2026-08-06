# Prompt Guide

Two categories of prompts in this project: (1) prompts sent to Florence-2 at inference time
(Q1 extraction, Q4 judge) — these are the ones graded directly by the brief. (2) prompts for
driving AI coding assistants per-phase — these shape what gets built, not model outputs.

---

## 1. Q1 — VLM structured-extraction prompt (Florence-2)

Florence-2 responds best to short task-prefixed prompts rather than long conversational
instructions. Use its `<DETAILED_CAPTION>` / region-grounding modes to get raw description,
then a second deterministic pass to force it into schema, rather than trusting one free-form
generation to hit exact JSON. Two-stage beats one-stage here because Florence-2's native output
mode is captioning, not JSON — asking it to "invent" a JSON string from scratch drifts on key
names over many images.

**Stage A — grounded description (per image pair):**
```
<DETAILED_CAPTION>
```
(Florence-2 task token; run once on the garment image, once on the person image.)

**Stage B — schema extraction (text-only follow-up, on Stage A's captions):**
```
You are extracting structured attributes from two image captions: one of a garment, one of a
person. Return ONLY a single JSON object, no prose before or after, matching exactly this shape:

{
  "garment_type": string,        // one of: shirt, t-shirt, blouse, dress, jacket, sweater, tank_top, other
  "sleeve_length": string,       // one of: sleeveless, short, three_quarter, long, unknown
  "neckline": string,            // one of: crew, v_neck, collar, scoop, halter, strappy, unknown
  "primary_color": string,       // one lowercase color word
  "pattern": string,             // one of: solid, striped, floral, printed, checked, other
  "pose_category": string,       // one of: front-facing, side, seated
  "upper_body_visible": boolean,
  "lower_body_visible": boolean
}

Garment caption: "<Stage A garment caption>"
Person caption: "<Stage A person caption>"

If a field cannot be determined from the captions, use "unknown" (or false for booleans).
Return the JSON object now.
```

**Post-processing (mandatory, not optional):**
- Parse with `json.loads`; on failure, strip anything before the first `{` / after the last `}`
  and retry once.
- Validate keys against `sample_output_q1.json`'s key set exactly — extra/missing keys = fail
  this image, log it, move to the next rather than silently coercing.
- If retry also fails, log the raw model output verbatim in `results_summary.json` under a
  `"error"` field for that image — this is what turns a silent failure into documented partial
  credit per Rule A.6.

---

## 2. Q4c — VLM-as-judge rubric prompt (Florence-2, reused instance)

```
You are a strict but fair quality judge for an AI virtual try-on result. You will see a
description of the original garment, the original person photo, and the try-on output image.
Score the output from 1 (unusable) to 10 (indistinguishable from a real photo) on these three
combined factors:

1. Fit realism — does the garment drape and fit the body naturally, matching pose and proportions?
2. Artifacts — visible blending seams, warped textures, duplicated limbs, color bleeding?
3. Texture transfer — did the garment's actual pattern, color, and material carry over accurately,
   or was it simplified/distorted?

Original garment description: "<garment caption>"
Original person description: "<person caption>"
Try-on output description: "<Stage A caption run on the Q3 output image>"

Return ONLY this JSON:
{
  "score": integer 1-10,
  "reasons": {
    "fit_realism": string,
    "artifacts": string,
    "texture_transfer": string
  }
}
```

Same post-processing rules as §1 apply (parse, validate, log-and-continue on failure). Keep this
rubric prompt **verbatim in the Q4 phase README** — the brief explicitly asks for the rubric
prompt to be committed, not just described.

---

## 3. Coding-assistant prompts (per phase, for use with Claude Code / similar)

These are the prompts Anirudh gives an AI coding assistant at the start of each phase session,
per the one-agent-per-phase rule in AGENTS.md. Each one should be pasted along with the relevant
sections of PRD.md and DECISIONS_AND_RULES.md so the assistant isn't inventing its own model
choices.

**Phase 0 (Scaffold Agent):**
```
Set up the repo skeleton for this project per docs/PRD.md and docs/DECISIONS_AND_RULES.md §C.
Create the folder structure, one requirements.txt per phase (do not merge dependencies across
phases), a Colab notebook skeleton per phase with TODO markers, and Drive-mount boilerplate for
checkpointing per rule D7. Do not write any model inference code yet.
```

**Phase 1 (VLM Agent):**
```
Implement Q1 per docs/IMPLEMENTATION_PLAN.md Phase 1 and the prompt in docs/PROMPT_GUIDE.md §1.
Load Florence-2-large in fp16 (decision D1, D5). Build the two-stage caption→JSON pipeline,
including the retry-and-log validation logic — do not skip the validator even if outputs look
fine on the first few images. Run on all 10 test_pairs images plus the 3 edge_cases person
images. Output one JSON per image plus results_summary.json.
```

**Phase 2 (Vision Pipeline Agent):**
```
Implement Q2 per docs/IMPLEMENTATION_PLAN.md Phase 2. Use SCHP for parsing (D2) and rembg for
garment cutout (D3), fp16 where applicable (D5), sequential model loading with cache clearing
between SCHP and rembg (D6), checkpoint every output to Drive (D7). Explicitly test and report
on person_02, person_03 (hair over shoulders), person_crossed_arms.jpg, and garment_03
(strappy) — these are graded edge cases, not incidental inputs.
```

**Phase 3 (Try-On Agent):**
```
Implement Q3 per docs/IMPLEMENTATION_PLAN.md Phase 3. Use CatVTON (D4), wire in Q2's parsing
map + agnostic image + garment mask as its input. Run a single-pair smoke test before batching
all 5 pairs from pairs_manifest.csv. Monitor VRAM with torch.cuda.max_memory_allocated() and
apply the OOM fallback ladder in that order: fp16 -> attention slicing -> VAE CPU offload ->
512x512 resolution. Write constraints_log.md documenting every step tried, whether it was
needed, and the result — including if nothing broke.
```

**Phase 4 (Evaluation Agent):**
```
Implement Q4 per docs/IMPLEMENTATION_PLAN.md Phase 4. OpenCLIP cosine similarity for garment
fidelity, InsightFace SSIM + embedding distance for identity, and the exact rubric prompt in
docs/PROMPT_GUIDE.md §2 run through the same Florence-2 instance from Phase 1 (do not reload).
Fill evaluation_template_q4.csv for all 5 pairs, matching its existing columns exactly (do not
rename/reorder). Report scores per decision D9 — three separate axes, not one blended score.
```

**Phase 5 (Demo Agent):**
```
Implement Q5 per docs/IMPLEMENTATION_PLAN.md Phase 5. Build a Gradio app chaining Q2 -> Q3 ->
Q4/Q1 as one pipeline function. Implement guardrails: reject if no person detected (run parsing
first, check for person-class pixels before calling Q3 at all), warn (not reject) if pose is
side or seated per Q1's classification, and show a resolution-based ETA estimate. Before
recording any demo video, run and confirm expected behavior on exactly these three files:
edge_cases/no_person.jpg, edge_cases/person_seated.jpg, edge_cases/person_side_pose.jpg.
```

**Phase 6 (Docs Agent, run once per completed phase):**
```
Write the phase README for <phase> using the template in docs/IMPLEMENTATION_PLAN.md §8. Base
"what's implemented" strictly on the actual files/outputs present in this phase's folder right
now — do not describe planned-but-unbuilt functionality. Base "why it took this long" on the
actual iteration/debugging that happened this session, not the plan's original estimate.
```

## 4. Prompt-writing rules (apply to all prompts above)
- Model-facing prompts always specify the exact output shape and explicitly forbid prose
  wrapping the JSON — Florence-2 will add a leading sentence unless told not to.
- Never send a coding-assistant prompt without attaching the relevant DECISIONS_AND_RULES.md
  rows — otherwise the assistant will re-derive its own (possibly different) model choices.
- Every coding-assistant prompt above ends its scope at one phase folder — this is deliberate,
  matching AGENTS.md's "no agent writes outside its own phase" boundary.
