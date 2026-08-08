# Achieved Goals Tracker

Update this after each phase closes (per AGENTS.md's handoff rule: a phase is "done" only once
its README exists and the README instructions have been run on a clean checkout). This is the
single place to check "are we submission-ready" without re-reading every phase README.

## Phase 0 — Scaffold
- [x] Repo structure created matching one-folder-per-question
- [x] Per-phase requirements.txt files pinned, no cross-phase conflicts confirmed
- [x] sample_files.zip contents verified complete against brief's file table
- [x] pair_04 / pair_05 sourced and added (front-facing, upper body visible)
- [x] Drive checkpointing wired

## Phase 1 — Q1 (15 marks)
- [x] Florence-2 loaded, fp16 confirmed on T4
- [x] JSON schema validator implemented and tested against sample_output_q1.json
- [x] All 10 test_pairs images processed
- [x] All 3 edge_cases person images processed, side/seated pose classified correctly
- [x] Phase README written (what/how/why/gotchas)

## Phase 2 — Q2 (20 marks)
- [x] SCHP parsing maps generated for all 5 person images + person_crossed_arms.jpg
- [x] Agnostic images generated, upper-clothes region correctly masked
- [x] hair-over-shoulder edge case (person_02, person_03) verified visually
- [x] crossed-arms edge case verified visually
- [x] rembg garment masks generated for all 5 garments, garment_03 (strappy) verified clean
- [x] Phase README written

## Phase 3 — Q3 (25 marks)
- [x] Single-pair smoke test passed
- [x] All 5 official pairs run through CatVTON
- [x] VRAM monitored, OOM fallback ladder tested/documented (even if unused)
- [x] constraints_log.md written with every constraint hit + fix, or explicit "none hit"
- [x] Phase README written

## Phase 4 — Q4 (20 marks)
- [x] OpenCLIP fidelity scores computed for all 5 pairs
- [x] InsightFace identity scores (SSIM + embedding distance) computed for all 5 pairs
- [x] VLM-judge rubric run for all 5 pairs, rubric prompt committed verbatim to README
- [x] evaluation_template_q4.csv filled, columns match template exactly
- [x] Phase README written

## Phase 5 — Q5 (20 marks)
- [x] Gradio app chains Q2 -> Q3 -> Q4/Q1 as one pipeline
- [x] No-person guardrail tested live on edge_cases/no_person.jpg
- [x] Seated-pose warning tested live on edge_cases/person_seated.jpg
- [x] Side-pose warning tested live on edge_cases/person_side_pose.jpg
- [x] ETA estimate implemented (resolution/fallback-aware, not hardcoded)
- [x] Demo video recorded (≤5 min), shows all 3 guardrails firing
- [x] Phase README written

## Phase 6 — Submission Packaging
- [x] Root README.md filled from SUBMISSION_TEMPLATE.md, every section complete
- [x] All model licenses listed (incl. non-commercial research licenses, per Rule A.2)
- [x] evaluation_template_q4.csv committed with real data
- [x] Clean-checkout test passed: clone fresh, follow only README, Q1-Q5 all run
- [x] Demo video uploaded/linked
- [x] Colab notebook links included, outputs saved (not just code)
- [x] Repo set to public or invite-shared as required

