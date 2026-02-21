from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.dataset import EIDSEG_CLASSES, _BRIDGE_CLASSES, _ROAD_CLASSES

# Colours for EIDSeg 6-class segmentation overlay
_EIDSEG_COLORS: list[tuple[int, int, int]] = [
    (144, 238, 144),  # Undamaged Building — light green
    (255, 165,   0),  # Damaged Building   — orange
    (220,  20,  60),  # Destroyed Building — crimson
    ( 70, 130, 180),  # Undamaged Road     — steel blue
    (255,  99,  71),  # Damaged Road       — tomato
    (200, 200, 200),  # Background         — light grey
]

_ROAD_DAMAGE_NAMES: dict[str, str] = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack (fatigue cracking)",
    "D40": "pothole",
}

# 19 visually distinct (R, G, B) colours — one per bridge defect class
_CLASS_COLORS: list[tuple[int, int, int]] = [
    (220,  20,  60),  # Crack         — crimson
    (255, 165,   0),  # ACrack        — orange
    ( 50, 205,  50),  # Wetspot       — lime green
    (  0, 191, 255),  # Efflorescence — deep sky blue
    (238, 130, 238),  # Rust          — violet
    (255, 215,   0),  # Rockpocket    — gold
    (  0, 128, 128),  # Hollowareas   — teal
    (255,  99,  71),  # Cavity        — tomato
    (100, 149, 237),  # Spalling      — cornflower blue
    (144, 238, 144),  # Graffiti      — light green
    (255, 182, 193),  # Weathering    — light pink
    ( 64, 224, 208),  # Restformwork  — turquoise
    (218, 165,  32),  # ExposedRebars — goldenrod
    (173, 216, 230),  # Bearing       — light blue
    (250, 128, 114),  # EJoint        — salmon
    (152, 251, 152),  # Drainage      — pale green
    (135, 206, 235),  # PEquipment    — sky blue
    (255, 160, 122),  # JTape         — light salmon
    (176, 196, 222),  # WConccor      — light steel blue
]


class BridgeVLMInference:
    """
    Post-segmentation VLM reasoning.

    Usage:
        vlm = BridgeVLMInference(adapter_path="checkpoints/vlm_lora/final_adapter")
        result = vlm.describe(pil_image, seg_probs_tensor)
        # result = {"report": str, "detected": list[str], "overlay_image": PIL.Image}
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------
    # Mask overlay
    # ------------------------------------------------------------------

    def render_mask_overlay(
        self,
        image: Image.Image,
        seg_probs: torch.Tensor,
        threshold: float = 0.5,
        alpha: float = 0.45,
    ) -> tuple[Image.Image, list[str]]:
        """
        Alpha-composite 19-class colour masks onto the original image.

        Args:
            image:     original PIL image.
            seg_probs: (19, H, W) float tensor of class probabilities.
            threshold: probability threshold for a pixel to be considered active.
            alpha:     mask opacity.

        Returns:
            (overlay PIL image, list of detected class names)
        """
        H, W = seg_probs.shape[1], seg_probs.shape[2]
        base = image.resize((W, H), Image.LANCZOS).convert("RGBA")
        composite = Image.new("RGBA", base.size, (0, 0, 0, 0))
        detected: list[str] = []

        for i, cls_name in enumerate(_BRIDGE_CLASSES):
            prob_map = seg_probs[i].cpu().numpy()  # (H, W)
            binary   = (prob_map > threshold).astype(np.uint8)
            if binary.sum() == 0:
                continue
            detected.append(cls_name)
            r, g, b   = _CLASS_COLORS[i % len(_CLASS_COLORS)]
            fill_layer = Image.new("RGBA", base.size, (r, g, b, 0))
            mask_layer = Image.fromarray(
                (binary * int(255 * alpha)).astype(np.uint8), mode="L"
            )
            fill_layer.putalpha(mask_layer)
            composite = Image.alpha_composite(composite, fill_layer)

        result = Image.alpha_composite(base, composite).convert("RGB")
        return result, detected

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_inference_prompt(
        self,
        detected_classes: list[str],
        coverage_map: dict[str, float],
    ) -> str:
        if not detected_classes:
            return (
                "This bridge image shows no detected defects from automated segmentation. "
                "Provide a brief expert assessment confirming the apparent structural integrity."
            )
        lines = [
            f"- {cls}: {coverage_map.get(cls, 0.0):.2f}% image coverage"
            for cls in detected_classes
        ]
        defect_summary = "\n".join(lines)
        return (
            "You are a licensed bridge structural engineer. "
            "Automated segmentation has detected the following defects in this bridge image "
            "(colour overlays indicate defect locations):\n\n"
            f"{defect_summary}\n\n"
            "Write a concise 3-5 sentence expert inspection report describing the observed "
            "defects, their likely structural implications, and recommended next steps."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def describe(
        self,
        image: Image.Image,
        seg_probs: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """
        Full post-segmentation reasoning pipeline.

        Args:
            image:     original PIL image.
            seg_probs: (19, H, W) float tensor from SegFormer.
            threshold: probability threshold for class detection.

        Returns:
            {"report": str, "detected": list[str], "overlay_image": PIL.Image}
        """
        overlay_img, detected = self.render_mask_overlay(image, seg_probs, threshold)

        total_pixels = seg_probs.shape[1] * seg_probs.shape[2]
        coverage_map = {
            _BRIDGE_CLASSES[i]: round(
                float((seg_probs[i] > threshold).sum()) / total_pixels * 100, 2
            )
            for i in range(len(_BRIDGE_CLASSES))
        }

        prompt = self._build_inference_prompt(detected, coverage_map)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": overlay_img},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[overlay_img],
            return_tensors="pt",
        ).to(self.model.device)

        generated  = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        report     = self.processor.decode(output_ids, skip_special_tokens=True).strip()

        return {
            "report":        report,
            "detected":      detected,
            "overlay_image": overlay_img,
        }


# ---------------------------------------------------------------------------
# Road VLM inference
# ---------------------------------------------------------------------------

class RoadVLMInference:
    """
    Post-segmentation VLM reasoning for the road domain.

    Receives road classification probabilities + EIDSeg segmentation map and
    produces an expert pavement condition report.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    def render_eidseg_overlay(
        self,
        image: Image.Image,
        seg_map: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Alpha-composite EIDSeg 6-class segmentation map onto the image.

        Args:
            image:   original PIL image.
            seg_map: (H, W) int numpy array with class indices 0-5.
            alpha:   overlay opacity.

        Returns:
            Overlay PIL image (RGB).
        """
        H, W = seg_map.shape
        base = image.resize((W, H), Image.LANCZOS).convert("RGBA")
        composite = Image.new("RGBA", base.size, (0, 0, 0, 0))

        for cls_idx, (r, g, b) in enumerate(_EIDSEG_COLORS):
            binary = (seg_map == cls_idx).astype(np.uint8)
            if binary.sum() == 0:
                continue
            fill_layer = Image.new("RGBA", base.size, (r, g, b, 0))
            mask_layer = Image.fromarray(
                (binary * int(255 * alpha)).astype(np.uint8), mode="L"
            )
            fill_layer.putalpha(mask_layer)
            composite = Image.alpha_composite(composite, fill_layer)

        return Image.alpha_composite(base, composite).convert("RGB")

    def _build_inference_prompt(
        self,
        detected: list[str],
        damage_probs: dict[str, float],
        severity_score: float,
        passability: str,
    ) -> str:
        severity_label = (
            "minor" if severity_score < 0.2 else
            "moderate" if severity_score < 0.5 else
            "severe"
        )
        if not detected:
            return (
                f"This road image shows no detected damage. "
                f"Overall severity is assessed as {severity_label}; "
                f"passability: {passability}. "
                "Provide a brief expert assessment confirming the road condition."
            )
        lines = [
            f"- {_ROAD_DAMAGE_NAMES.get(cls, cls)}: confidence {damage_probs.get(cls, 0):.2f}"
            for cls in detected
        ]
        damage_summary = "\n".join(lines)
        return (
            "You are a licensed road infrastructure engineer. "
            "Automated analysis has detected the following pavement damage "
            "(colour overlays indicate affected areas):\n\n"
            f"{damage_summary}\n\n"
            f"Overall severity: {severity_label} (score {severity_score:.2f}). "
            f"Assessed passability: {passability}.\n\n"
            "Write a concise 3-5 sentence expert pavement condition report describing "
            "the observed damage, its likely structural and safety implications, "
            "and recommended maintenance priority and actions."
        )

    @torch.inference_mode()
    def describe(
        self,
        image: Image.Image,
        damage_probs: dict[str, float],
        seg_map: np.ndarray,
        severity_score: float,
        passability: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Full post-processing VLM reasoning for road images.

        Args:
            image:         original PIL image.
            damage_probs:  dict {class_code: probability} for road damage classes.
            seg_map:       (H, W) int numpy array from EIDSeg.
            severity_score: float severity score from the road model.
            passability:   passability label string.
            threshold:     probability threshold for class detection.

        Returns:
            {"report": str, "detected": list[str], "overlay_image": PIL.Image}
        """
        detected = [cls for cls, p in damage_probs.items() if p >= threshold and cls != "NoDefect"]
        overlay_img = self.render_eidseg_overlay(image, seg_map)
        prompt = self._build_inference_prompt(detected, damage_probs, severity_score, passability)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": overlay_img},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[overlay_img],
            return_tensors="pt",
        ).to(self.model.device)

        generated  = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        report     = self.processor.decode(output_ids, skip_special_tokens=True).strip()

        return {
            "report":        report,
            "detected":      detected,
            "overlay_image": overlay_img,
        }
