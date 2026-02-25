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

# 3-class EIDSeg bridge colours
_CLASS_COLORS: list[tuple[int, int, int]] = [
    ( 50, 205,  50),  # Undamaged — lime green
    (255, 165,   0),  # Damaged   — orange
    (220,  20,  60),  # Destroyed — crimson
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
        base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base = AutoModelForImageTextToText.from_pretrained(
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
        class_map: torch.Tensor,
        alpha: float = 0.45,
    ) -> tuple[Image.Image, list[str]]:
        """
        Alpha-composite 3-class EIDSeg colour masks onto the original image.

        Args:
            image:     original PIL image.
            class_map: (H, W) long tensor with values 0=Undamaged,1=Damaged,2=Destroyed.
            alpha:     mask opacity.

        Returns:
            (overlay PIL image, list of detected class names)
        """
        H, W  = class_map.shape
        base  = image.resize((W, H), Image.LANCZOS).convert("RGBA")
        composite = Image.new("RGBA", base.size, (0, 0, 0, 0))
        detected: list[str] = []
        cm_np = class_map.cpu().numpy()

        for i, cls_name in enumerate(_BRIDGE_CLASSES):
            if i == 0:
                continue  # skip Undamaged overlay — show only damage
            binary = (cm_np == i).astype(np.uint8)
            if binary.sum() == 0:
                continue
            detected.append(cls_name)
            r, g, b    = _CLASS_COLORS[i]
            fill_layer = Image.new("RGBA", base.size, (r, g, b, 0))
            mask_layer = Image.fromarray(
                (binary * int(255 * alpha)).astype(np.uint8), mode="L"
            )
            fill_layer.putalpha(mask_layer)
            composite  = Image.alpha_composite(composite, fill_layer)

        result = Image.alpha_composite(base, composite).convert("RGB")
        return result, detected

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_inference_prompt(
        self,
        detected_classes: list[str],
        coverage_map: dict[str, float],
        severity_score: float,
        passability: str,
    ) -> str:
        severity_label = (
            "Ringan" if severity_score < 0.2 else
            "Sedang" if severity_score < 0.5 else
            "Berat"
        )
        if not detected_classes:
            return (
                "You are a field assessor conducting emergency post-disaster structural triage. "
                "Automated analysis of this bridge image detected no structural damage. "
                f"Severity: {severity_label}. Passability: {passability}. "
                "Provide a brief triage report confirming the bridge is safe to cross."
            )
        lines = [
            f"- {cls}: {coverage_map.get(cls, 0.0):.2f}% coverage"
            for cls in detected_classes
        ]
        defect_summary = "\n".join(lines)
        return (
            "You are a field assessor conducting emergency post-disaster structural triage. "
            "Your report will be used by emergency response teams to make immediate "
            "access and evacuation decisions.\n\n"
            "Look at the image carefully. Colour overlays mark the detected damage zones: "
            "orange = Damaged, red = Destroyed.\n\n"
            "Automated analysis results:\n"
            f"{defect_summary}\n\n"
            f"Severity: {severity_label} ({severity_score:.2f}). "
            f"Passability: {passability}.\n\n"
            "Based on what you see in the image and the analysis above, write a detailed triage report covering:\n"
            "1. What the visible damage looks like and how extensive it is\n"
            "2. Immediate safety risks for responders and civilians\n"
            "3. Clear access recommendation with reasoning\n"
            "4. Suggested next steps for the response team\n\n"
            "Be thorough — 4 to 6 sentences."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def describe(
        self,
        image: Image.Image,
        class_map: torch.Tensor,
        severity_score: float,
        passability: str,
        coverage: dict[str, float],
    ) -> dict:
        """
        Full post-segmentation reasoning pipeline.

        Args:
            image:          original PIL image.
            class_map:      (H, W) long tensor with values 0=Undamaged,1=Damaged,2=Destroyed.
            severity_score: float severity score from compute_bridge_severity().
            passability:    passability tier string from compute_bridge_passability().
            coverage:       pre-computed coverage dict {class_name: pct}.

        Returns:
            {"report": str, "detected": list[str], "overlay_image": PIL.Image}
        """
        overlay_img, detected = self.render_mask_overlay(image, class_map)

        prompt = self._build_inference_prompt(detected, coverage, severity_score, passability)

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
        base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base = AutoModelForImageTextToText.from_pretrained(
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
            "Ringan" if severity_score < 0.2 else
            "Sedang" if severity_score < 0.5 else
            "Berat"
        )
        if not detected:
            return (
                "You are a field assessor conducting emergency post-disaster road triage. "
                "Automated analysis of this road image detected no damage. "
                f"Severity: {severity_label}. Passability: {passability}. "
                "Provide a brief triage report confirming the road is passable."
            )
        lines = [
            f"- {_ROAD_DAMAGE_NAMES.get(cls, cls)}: confidence {damage_probs.get(cls, 0):.2f}"
            for cls in detected
        ]
        damage_summary = "\n".join(lines)
        return (
            "You are a field assessor conducting emergency post-disaster road triage. "
            "Your report will be used by emergency response teams to make immediate "
            "routing and evacuation decisions. "
            "Automated analysis detected the following road damage "
            "(colour overlays indicate affected areas):\n\n"
            f"{damage_summary}\n\n"
            f"Severity: {severity_label} ({severity_score:.2f}). "
            f"Passability: {passability}.\n\n"
            "Write a concise 2-3 sentence triage report describing the damage, its immediate "
            "safety risk, and the routing status. Use field-ready language "
            "(e.g., 'road closed', 'motorcycles only', 'avoid heavy vehicles', 'passable')."
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
