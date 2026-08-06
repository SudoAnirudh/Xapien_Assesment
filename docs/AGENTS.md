# AGENTS.md

This assessment explicitly allows AI coding assistants, on the condition that Anirudh can
explain every line in a follow-up interview. This file defines how AI agents are used here —
as scoped, single-phase pair-programmers with a hard boundary, not as an autonomous builder.
**Anirudh reviews and can explain every line before it's committed** — that review step is not
delegated to any agent.

## Operating principle
One agent session per phase (Q1–Q5), each restarted fresh with only that phase's context. This
keeps each agent's output small enough to read in full and matches the repo's one-folder-per-
question structure. No agent has write access outside its own phase folder + its own phase
README.

## Agent roster

### 1. Scaffold Agent (Phase 0)
- **Owns:** repo structure, `requirements.txt` per phase, Colab notebook skeletons, shared
  `config.py` pattern, `.gitignore`, README index.
- **Does not:** write any model-inference code.
- **Input:** DECISIONS_AND_RULES.md §C (coding conventions).
- **Output:** empty-but-runnable folder skeleton + notebook cells with `# TODO(phaseN)` markers.

### 2. VLM Agent (Q1, and reused for Q4c judge prompt)
- **Owns:** Florence-2 loading/inference wrapper, prompt template for structured JSON output,
  JSON schema validator against `sample_output_q1.json`.
- **Constraint it must respect:** D1, D8 (fp16, sequential loading, schema validation).
- **Must produce:** one JSON file per input image + a `results_summary.json` aggregating all 13
  (10 test + 3 edge-case) outputs, plus a short note on any image where classification looked
  wrong on manual spot-check.

### 3. Vision Pipeline Agent (Q2)
- **Owns:** SCHP parsing wrapper, agnostic-image generator (mask upper-clothes class → inpaint/
  gray-fill), rembg garment cutout wrapper.
- **Constraint it must respect:** D2, D3, D5, D6, D7.
- **Must produce:** parsing map + agnostic image for every person image (incl. the 3 provided
  edge cases named in Rule A.4) and a clean-background mask for every garment image.

### 4. Try-On Agent (Q3)
- **Owns:** CatVTON inference wrapper, wiring Q2 outputs as its input, constraints log.
- **Constraint it must respect:** D4, D5, D6, D7; must actively try the OOM fallback in
  PRD §6 and log the outcome, not just assume it'll be needed.
- **Must produce:** 5 output images (pair_01–pair_05) + `constraints_log.md` documenting every
  GPU/memory/timeout issue hit and the fix applied, even if nothing broke (say so explicitly).

### 5. Evaluation Agent (Q4)
- **Owns:** OpenCLIP fidelity scorer, InsightFace identity scorer, VLM-judge rubric prompt
  (reusing the VLM Agent's Florence-2 wrapper), CSV writer matching
  `evaluation_template_q4.csv` columns exactly.
- **Constraint it must respect:** D9.
- **Must produce:** completed CSV for all 5 pairs + the exact rubric prompt text committed to
  the README (not just described).

### 6. Demo Agent (Q5)
- **Owns:** Gradio app wiring Q2→Q3→Q4 as one callable pipeline, guardrail logic (no-person
  reject, pose warning, ETA estimate), UI layout.
- **Constraint it must respect:** D10, Rule A.5 — guardrails must be shown firing on the exact
  3 named edge-case files.
- **Must produce:** `app.py` + a checklist confirming each of the 3 guardrail files was tested
  live before the demo video was recorded.

### 7. Docs Agent (continuous, runs after every other agent)
- **Owns:** per-phase README (what/how/why-it-took-this-long/gotchas), root README from
  `SUBMISSION_TEMPLATE.md`, keeping DECISIONS_AND_RULES.md's deviation log current.
- **Does not:** touch inference code — only writes about it, and only after reading the actual
  diff/output from that phase (never speculative documentation of unbuilt code).

## Handoff rule
An agent's phase is "done" only when the Docs Agent's per-phase README exists and Anirudh has
manually run the phase's README instructions on a clean checkout once. This is what makes the
"50% runs from README" grading criterion checkable before submission, not discovered after.
