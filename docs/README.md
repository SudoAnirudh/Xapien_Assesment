# XIPL SDE Intern Assessment — Virtual Try-On Pipeline: Planning Docs

This is the planning scaffold for the assessment, built before any model code. It is **not**
the final submission README — that gets built from `SUBMISSION_TEMPLATE.md` in Phase 6 (see
`docs/IMPLEMENTATION_PLAN.md`) once every question's outputs actually exist.

## Read in this order
1. **`docs/PRD.md`** — scope, hard constraints, locked model choices + why, risks
2. **`docs/DECISIONS_AND_RULES.md`** — the single source of truth for every technical decision;
   check here before deviating from anything in the PRD
3. **`docs/IMPLEMENTATION_PLAN.md`** — phase-by-phase build order with time budgets, plus the
   per-phase documentation template every phase README must follow
4. **`docs/PROMPT_GUIDE.md`** — the exact model-facing prompts (Q1 extraction, Q4 judge) and
   the coding-assistant prompts to run per phase
5. **`AGENTS.md`** — which agent/session owns which phase, and the boundary each one works within
6. **`docs/ACHIEVED_GOALS.md`** — checklist to update as each phase closes; the "are we
   submission-ready" source of truth

## Folder layout
```
Q1_vlm_understanding/     Florence-2 extraction script + JSON outputs
Q2_parsing_segmentation/  SCHP parsing + agnostic images + rembg garment masks
Q3_tryon_inference/       CatVTON pipeline + 5 output images + constraints_log.md
Q4_evaluation/            OpenCLIP/InsightFace/VLM-judge scoring + evaluation_template_q4.csv
Q5_web_demo/              Gradio app + guardrail tests
docs/                     PRD, plan, prompt guide, decisions, goals tracker (this folder)
AGENTS.md                 AI agent roles and boundaries
README.md                 this file (becomes submission README in Phase 6)
```

## Status
Not started — see `docs/ACHIEVED_GOALS.md` for live progress once Phase 0 begins.
