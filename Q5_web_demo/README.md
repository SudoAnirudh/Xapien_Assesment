# Q5 — Mini Try-On Web Demo

## What's implemented
- Gradio web application (`app.py`) integrating Q2→Q3→Q4 into a single pipeline.
- Guardrail 1: Person presence verification (`no_person.jpg` → immediate rejection).
- Guardrail 2: Non-standard pose warning (`person_seated.jpg` & `person_side_pose.jpg` → warning notice).
- Dynamic ETA estimator function based on image resolution and hardware mode.
- Verification test suite confirming guardrail execution on the 3 required test files.

## How it works
- **Pipeline Wiring**: Upload Person & Garment → Guardrails Check → SCHP & rembg Preprocessing (Q2) → CatVTON Inference (Q3) → 3-Axis Evaluation (Q4) → UI Display.
- **Guardrail Logic**: Fast SCHP person detection pass checks for human pixel classes before invoking CatVTON.
- **UI Design**: Multi-column layout with real-time status log, ETA countdown, and scoring table.

## Why it took this long
- Setup & Gradio interface structure (~45 min).
- Guardrail logic implementation & validation (~2 hrs).
- Pipeline integration across Q2, Q3, and Q4 (~2 hrs).
- Live testing on mandatory edge case files & demo recording (~1.25 hrs).

## Gotchas / edge cases
- Guardrail 1 (`no_person.jpg`): Prevents unnecessary CatVTON GPU allocation when no human subject is present.
- Guardrail 2 (`person_seated.jpg`, `person_side_pose.jpg`): Displays clear UI warnings while allowing users to proceed.

## Constraints hit
- **Memory**: Models initialized sequentially; Gradio interface runs smoothly on Colab T4.
- **Errors**: None encountered during initial setup.
