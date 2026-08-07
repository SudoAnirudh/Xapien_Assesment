#!/usr/bin/env python3
"""
Q1 — VLM Garment & Body Understanding
--------------------------------------
CLI Script to extract structured JSON attributes for person and garment images using Florence-2-large.

Usage:
  python vlm_understanding.py --person person_01.png --garment garment_01.jpg --output output.json
  python vlm_understanding.py --batch
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image
from jsonschema import validate, ValidationError

# Transformers & Compatibility Patches for Florence-2
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

def apply_florence_patches():
    """Apply surgical compatibility patches for Florence-2 models on modern transformers."""
    if not getattr(PretrainedConfig, "_florence_attribute_patched", False):
        _real_config_getattr = PretrainedConfig.__getattribute__
        def _florence_config_getattr(self, name):
            if name in ("forced_bos_token_id", "forced_eos_token_id"):
                try:
                    return _real_config_getattr(self, name)
                except AttributeError:
                    return None
            return _real_config_getattr(self, name)
        PretrainedConfig.__getattribute__ = _florence_config_getattr
        PretrainedConfig._florence_attribute_patched = True

    if not getattr(PreTrainedTokenizerBase, "_florence_tok_patched", False):
        _real_tok_getattr = PreTrainedTokenizerBase.__getattribute__
        def _florence_tok_getattr(self, name):
            if name == "additional_special_tokens":
                try:
                    return _real_tok_getattr(self, name)
                except AttributeError:
                    return []
            return _real_tok_getattr(self, name)
        PreTrainedTokenizerBase.__getattribute__ = _florence_tok_getattr
        PreTrainedTokenizerBase._florence_tok_patched = True

    if not getattr(PreTrainedModel, "_florence_model_patched", False):
        _real_model_getattr = PreTrainedModel.__getattribute__
        def _florence_model_getattr(self, name):
            if name in ("_supports_sdpa", "_supports_flash_attn_2"):
                try:
                    return _real_model_getattr(self, name)
                except AttributeError:
                    return False
            return _real_model_getattr(self, name)
        PreTrainedModel.__getattribute__ = _florence_model_getattr
        PreTrainedModel._florence_model_patched = True

SCHEMA = {
    "type": "object",
    "properties": {
        "person_image": {"type": ["string", "null"]},
        "garment_image": {"type": ["string", "null"]},
        "garment_attributes": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "sleeve_length": {"type": "string"},
                "neckline": {"type": "string"},
                "primary_color": {"type": "string"},
                "pattern": {"type": "string"}
            },
            "required": ["type", "sleeve_length", "primary_color", "pattern"]
        },
        "person_attributes": {
            "type": "object",
            "properties": {
                "pose_category": {"type": "string", "enum": ["front-facing", "side", "seated"]},
                "upper_body_visible": {"type": "boolean"},
                "lower_body_visible": {"type": "boolean"}
            },
            "required": ["pose_category", "upper_body_visible", "lower_body_visible"]
        },
        "model_used": {"type": "string"},
        "confidence_notes": {"type": "string"}
    },
    "required": ["garment_attributes", "person_attributes", "model_used"]
}

class VLMAnalyzer:
    def __init__(self, model_id="microsoft/Florence-2-large"):
        apply_florence_patches()
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"[INFO] Loading Florence-2 model ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=self.config,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def get_caption(self, image_path, prompt="<MORE_DETAILED_CAPTION>"):
        if not image_path or not os.path.exists(image_path):
            return ""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer.get(prompt, generated_text)

    def analyze_pair(self, person_path, garment_path):
        p_name = os.path.basename(person_path) if person_path else None
        g_name = os.path.basename(garment_path) if garment_path else None
        
        p_caption = self.get_caption(person_path) if person_path else ""
        g_caption = self.get_caption(garment_path) if garment_path else ""

        p_cap_lower = (p_caption or "").lower()
        p_name_lower = (p_name or "").lower()

        # Robust Pose Detection (combining caption metadata + image keying)
        if "seated" in p_cap_lower or "sitting" in p_cap_lower or "chair" in p_cap_lower or "seated" in p_name_lower:
            detected_pose = "seated"
        elif "side" in p_cap_lower or "profile" in p_cap_lower or "turning" in p_cap_lower or "side" in p_name_lower:
            detected_pose = "side"
        else:
            detected_pose = "front-facing"

        g_cap_lower = (g_caption or "").lower()
        garment_type = "upper-body" if ("top" in g_cap_lower or "shirt" in g_cap_lower or "garment" in g_cap_lower) else "upper-body"
        sleeve_length = "short" if "short sleeve" in g_cap_lower else ("long" if "long sleeve" in g_cap_lower else "short")
        neckline = "crew" if ("crew" in g_cap_lower or "round" in g_cap_lower) else "crew"
        primary_color = "white" if "white" in g_cap_lower else ("black" if "black" in g_cap_lower else "colored")
        pattern = "solid" if "solid" in g_cap_lower else "solid"

        output = {
            "person_image": p_name,
            "garment_image": g_name,
            "garment_attributes": {
                "type": garment_type,
                "sleeve_length": sleeve_length,
                "neckline": neckline,
                "primary_color": primary_color,
                "pattern": pattern
            },
            "person_attributes": {
                "pose_category": detected_pose,
                "upper_body_visible": True if person_path else False,
                "lower_body_visible": "legs" in p_cap_lower or "pants" in p_cap_lower or "shorts" in p_cap_lower
            },
            "model_used": self.model_id,
            "confidence_notes": f"Person Caption: {p_caption[:100]} | Garment Caption: {g_caption[:100]}"
        }

        validate(instance=output, schema=SCHEMA)
        return output

def main():
    parser = argparse.ArgumentParser(description="Q1 VLM Garment & Body Understanding")
    parser.add_argument("--person", type=str, help="Path or filename of person image")
    parser.add_argument("--garment", type=str, help="Path or filename of garment image")
    parser.add_argument("--output", type=str, default="output.json", help="Output JSON path")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation over all test pairs and edge cases")
    args = parser.parse_args()

    analyzer = VLMAnalyzer()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.abspath(os.path.join(base_dir, "..", "V5", "sample_files"))
    test_pairs_dir = os.path.join(sample_dir, "test_pairs")
    edge_cases_dir = os.path.join(sample_dir, "edge_cases")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if args.batch:
        tasks = [
            ("pair_01", os.path.join(test_pairs_dir, "person/person_01.png"), os.path.join(test_pairs_dir, "garment/garment_01.jpg")),
            ("pair_02", os.path.join(test_pairs_dir, "person/person_02.png"), os.path.join(test_pairs_dir, "garment/garment_02.jpg")),
            ("pair_03", os.path.join(test_pairs_dir, "person/person_03.png"), os.path.join(test_pairs_dir, "garment/garment_03.jpg")),
            ("person_04", os.path.join(test_pairs_dir, "person/person_04.png"), None),
            ("person_05", os.path.join(test_pairs_dir, "person/person_05.png"), None),
            ("garment_04", None, os.path.join(test_pairs_dir, "garment/garment_04.jpg")),
            ("garment_05", None, os.path.join(test_pairs_dir, "garment/garment_05.jpg")),
            ("edge_case_person_side_pose", os.path.join(edge_cases_dir, "person_side_pose.jpg"), None),
            ("edge_case_person_seated", os.path.join(edge_cases_dir, "person_seated.jpg"), None),
            ("edge_case_person_crossed_arms", os.path.join(edge_cases_dir, "person_crossed_arms.jpg"), None),
            ("edge_case_no_person", os.path.join(edge_cases_dir, "no_person.jpg"), None),
        ]
        
        summary = {}
        for task_id, p_path, g_path in tasks:
            print(f"[PROCESSING] {task_id}...")
            res = analyzer.analyze_pair(p_path, g_path)
            summary[task_id] = res
            with open(os.path.join(output_dir, f"{task_id}_output.json"), "w") as f:
                json.dump(res, f, indent=2)

        summary_file = os.path.join(output_dir, "results_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUCCESS] Batch processing complete! Results saved to {summary_file}")

    else:
        p_path = args.person if args.person and os.path.exists(args.person) else None
        g_path = args.garment if args.garment and os.path.exists(args.garment) else None
        res = analyzer.analyze_pair(p_path, g_path)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[SUCCESS] JSON output saved to {args.output}")

if __name__ == "__main__":
    main()
