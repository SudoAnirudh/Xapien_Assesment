# PRD — XIPL SDE Intern Technical Assessment: Open-Source Virtual Try-On Pipeline

## 1. Purpose
Build and document a working, end-to-end AI virtual try-on system using only open-source,
free-tier-compatible models. The deliverable is not a product — it's evidence of the ability to:
read research repos, pick the right model for a constraint (Colab T4, 15GB VRAM, no paid APIs),
chain independent pipelines together, evaluate output objectively, and communicate trade-offs
honestly, including where things break.

Grading is 50% "does the code run from the README," 30% output quality, 20% code quality +
documented trade-offs. This PRD treats **reproducibility and documentation as first-class
deliverables**, not an afterthought — a broken pipeline with a clear postmortem scores higher
than a silent, undocumented one.

## 2. Scope (what's in / out)
**In scope**
- Q1: VLM script → structured JSON (garment + person attributes) for 10 test images + 3 edge-case person images
- Q2: Human parsing map + agnostic (upper-body-masked) image + garment segmentation, incl. 3 named edge cases
- Q3: End-to-end try-on inference on the 5 official pairs (3 provided + 2 self-sourced), Q2 wired in as preprocessing
- Q4: 3-axis automated scoring (garment fidelity, identity preservation, VLM-judge) for all 5 Q3 outputs
- Q5: Gradio/Streamlit demo wrapping Q2–Q4 with guardrails (no-person reject, pose warning, ETA)
- README (from SUBMISSION_TEMPLATE.md), demo video (≤5 min), Colab links, evaluation_template_q4.csv

**Out of scope**
- Training or fine-tuning any model
- Any paid/hosted inference API (OpenAI, Replicate paid tiers, etc.)
- Mobile/production deployment, auth, persistence, multi-user support
- Garment types beyond upper-body (per Q2's "upper-body clothing masked out" scope)

## 3. Constraints (hard rules, from the brief)
| Constraint | Detail |
|---|---|
| Compute | Google Colab **T4 GPU**, free tier — 15GB VRAM, session timeouts, ephemeral disk |
| Licensing | Try-on weights (IDM-VTON/OOTDiffusion/CatVTON) are non-commercial research licenses — acceptable, must be noted in README |
| Models | Open-source, freely downloadable weights only, for **every** model in the chain (VLM, parser, segmenter, try-on, evaluator) |
| AI assistants | Allowed, but every line must be explainable in a follow-up interview → keep code simple, avoid unexplainable boilerplate |
| Partial credit | Undocumented broken code = 0. A documented failure with root cause + attempted fixes still scores. |

## 4. Model selection (locked decisions — see DECISIONS_AND_RULES.md for full rationale)
| Role | Model | Why |
|---|---|---|
| VLM (Q1, reused in Q4c) | **Florence-2-large** (MIT) | Smallest footprint of the four options (~0.77B params), runs comfortably on T4 without quantization, strong at structured/grounded outputs, fast enough to reuse as the judge in Q4 without a second load |
| Human parsing (Q2a) | **SCHP** (Self-Correction Human Parsing) | Purpose-built for exactly this: 20-class human part segmentation incl. upper-clothes/hair — directly gives the mask Q2 needs, lighter than SAM+GroundingDINO two-model chain |
| Garment segmentation (Q2b) | **rembg (RMBG-1.4 backend)** | One-line inference, robust on flat product photos, actively maintained, trivial to explain in an interview |
| Try-on inference (Q3) | **CatVTON** | Explicitly flagged in the brief as the lightest/most Colab-friendly; ~4GB VRAM at inference, no UNet duplication like OOTDiffusion, simpler pipeline to reason about and explain |
| Garment fidelity (Q4a) | **OpenCLIP (ViT-B/32, laion2b)** | Off-the-shelf embedding similarity, small, fast, no training needed |
| Identity preservation (Q4b) | **InsightFace (buffalo_l)** | Standard face-embedding toolkit, gives both a face crop for SSIM and an embedding distance |
| VLM-judge (Q4c) | Florence-2 (same instance as Q1) reused with a rubric prompt | Avoids loading a second large model; Florence-2's grounding ability helps it reason about specific image regions when given the rubric |
| Web demo (Q5) | **Gradio** | Faster to wire to guardrail logic + video-recordable UI than Streamlit; native support for multi-image + JSON output blocks |

## 5. Success criteria (self-grading rubric, mirrors §7 of the brief)
For each of Q1–Q5:
- **Runs clean from README instructions** on a fresh Colab runtime, no manual patching
- **Output matches spec** (JSON schema for Q1 matches `sample_output_q1.json`; CSV matches `evaluation_template_q4.csv` columns)
- **All named edge cases handled and called out explicitly** (crossed arms, hair-over-shoulder, side/seated pose, no-person, strappy garment)
- **Every constraint hit (OOM, timeout, license, resolution) is logged with the workaround**, not silently patched
- **Code is short enough to defend line-by-line in an interview** — no copy-pasted boilerplate the author can't explain

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| T4 OOM during CatVTON inference | fp16 + attention slicing + CPU offload for the VAE; documented fallback to reduced resolution (768→512) if still OOM |
| Colab session disconnect mid-run | Checkpoint intermediate outputs (parsing maps, agnostic images) to Drive after each phase so re-runs are incremental |
| Florence-2 JSON output drifts from schema | Constrained decoding via explicit few-shot prompt + a Python-side JSON-repair/validation pass (see PROMPT_GUIDE.md) |
| SCHP/rembg license or repo availability changes | Pin exact commit hashes / HF revision IDs in requirements, noted in README |
| Self-sourced pair_04/pair_05 don't match front-facing/upper-body requirement | Source from a fixed royalty-free set (Unsplash/Pexels) and manually verify pose before committing |

## 7. Deliverable map
See IMPLEMENTATION_PLAN.md for the phase-by-phase build order and AGENTS.md for who (which
agent/role) owns which phase when using AI coding assistance.
