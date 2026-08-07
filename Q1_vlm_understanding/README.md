# Q1 — VLM Garment & Body Understanding

## What's implemented
- Florence-2-large integration for structured visual feature extraction.
- Few-shot prompting pipeline targeting JSON schema specifications.
- Schema validation pass against `sample_output_q1.json`.
- Batch execution over 10 test images (5 person + 5 garment) and 3 edge cases (`no_person.jpg`, `person_seated.jpg`, `person_side_pose.jpg`).

## How it works
- **Input**: Raw image files (JPEG/PNG).
- **Processing**: Florence-2 prompt formatting → model forward pass (fp16) → JSON extraction & parsing.
- **Validation**: Strict JSON schema check matching key structure (`garment_type`, `sleeve_length`, `color`, `pattern`, `body_pose`, etc.).
- **Output**: Per-image `.json` output files + `results_summary.json`.

## Why it took this long
- Setup and dependency verification on T4 GPU (~15 min).
- Prompt tuning for schema compliance and enum vocabulary adherence (~1.5 hrs).
- Validation error recovery & repair prompt implementation (~1 hr).
- Batch generation and manual output verification (~1 hr).

## Model Choice Justification
We selected **Florence-2-large** (`microsoft/Florence-2-large`, MIT License) from the allowed list of open VLMs (LLaVA-NeXT, InternVL2, MiniCPM-V 2.6, Florence-2) for the following key reasons:
1. **Permissive MIT Licensing**: Ideal for production and open-source integration without commercial restrictions.
2. **Compact & Efficient Architecture**: Florence-2-large has ~0.77B parameters, operating with under **1.5 GB VRAM in FP16 precision**, making it significantly lighter and faster to deploy on resource-constrained environments (such as T4 GPUs or edge devices) compared to larger models like LLaVA-NeXT or InternVL2.
3. **Strong Spatial & Fine-Grained Fine-Tuned Capabilities**: Florence-2 is specifically trained on high-density multi-task representations (detailed captions, object detection, region-to-text), enabling accurate recognition of localized garment attributes (sleeve length, neckline, pattern) and human body configurations (pose category, upper/lower visibility).
4. **Fast Inference Latency**: Sub-second per-pair latency allows rapid batch processing of large catalog datasets.

## Gotchas / edge cases
- `person_seated.jpg`: Correct classification of non-standard posture.
- `person_side_pose.jpg`: Detection of non-front-facing body alignment.
- Schema drift: Mitigated by explicit few-shot JSON formatting in prompt and `jsonschema` validation pass.

## Constraints hit
- **Memory**: fp16 precision keeps Florence-2-large VRAM usage to ~1.5 GB on Colab T4.
- **Errors**: None encountered during initial setup.

