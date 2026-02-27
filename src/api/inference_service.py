"""Inference service — wraps JatanMTL + VLM for bridge domain.

Loaded once at startup and reused across all API requests.
"""
from __future__ import annotations

import io
import torch
import numpy as np
from loguru import logger
from PIL import Image
from torchvision import transforms

from src.dataset import _BRIDGE_CLASSES
from src.model import JatanMTL


_SEVERITY_MAP: dict[str, str] = {"Ringan": "minor", "Sedang": "moderate", "Berat": "severe"}
_PASSABILITY_MAP: dict[str, str] = {"Bisa": "possible", "Roda-2": "bike_only", "Tidak Bisa": "impossible"}

# Overlay colors (RGB) for visualization
_OVERLAY_COLORS = {
    "Undamaged": (0, 255, 0),       # Green
    "Damaged": (255, 165, 0),        # Orange
    "Destroyed": (255, 0, 0),        # Red
}


def _severity_label(score: float) -> str:
    if score < 0.2:
        return "minor"
    elif score < 0.5:
        return "moderate"
    return "severe"


def _aggregate_passability(labels: list[str]) -> str:
    if "impossible" in labels:
        return "impossible"
    if "bike_only" in labels:
        return "bike_only"
    return "possible"


class InferenceService:
    """Singleton inference service. Instantiate once and reuse."""

    def __init__(
        self,
        bridge_checkpoint: str = "checkpoints/bridge_seg_best.pt",
        adapter_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading JatanMTL (bridge_seg_checkpoint={})", bridge_checkpoint)
        self.model = JatanMTL(
            bridge_seg_checkpoint=bridge_checkpoint,
        ).to(self.device)
        self.model.eval()

        self._bridge_tfm = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # VLM adapter (optional)
        self._bridge_vlm = None
        if adapter_path:
            from src.vlm.inference import BridgeVLMInference
            self._bridge_vlm = BridgeVLMInference(
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
            )

    def run(
        self,
        images: list[Image.Image],
        use_vlm: bool = True,
        threshold: float = 0.5,
    ) -> dict:

        per_image = []
        all_severities: list[float] = []
        all_passability: list[str] = []

        for img in images:
            entry = self._run_bridge(img, use_vlm, threshold)
            all_severities.append(entry["severity"]["score"])
            all_passability.append(entry["passability"])
            per_image.append(entry)

        agg_severity = max(all_severities)
        return {
            "images": per_image,
            "aggregate": {
                "severity":    {"score": round(agg_severity, 4), "label": _severity_label(agg_severity)},
                "passability": _aggregate_passability(all_passability),
            },
        }

    def generate_overlay(
        self,
        img: Image.Image,
        class_map: torch.Tensor,
        alpha: float = 0.4,
    ) -> Image.Image:
        """Generate overlay image with colored segmentation mask.

        Args:
            img: Original PIL image
            class_map: Class map from segmentation (H, W)
            alpha: Transparency of overlay (0-1)

        Returns:
            PIL Image with overlay
        """
        # Resize original image to match class_map size
        img_resized = img.resize((class_map.shape[1], class_map.shape[0]))
        img_array = np.array(img_resized)

        # Create overlay array
        overlay = np.zeros_like(img_array)
        class_map_np = class_map.cpu().numpy()

        for class_idx, class_name in enumerate(_BRIDGE_CLASSES):
            color = _OVERLAY_COLORS[class_name]
            mask = class_map_np == class_idx
            overlay[mask] = color

        # Blend original with overlay
        result = img_array * (1 - alpha) + overlay * alpha
        return Image.fromarray(result.astype(np.uint8))

    def _run_bridge(self, img: Image.Image, use_vlm: bool, threshold: float) -> dict:
        x = self._bridge_tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.segment_bridge(x, with_depth=True)

        presence_mask = out["presence"][0].cpu()
        probs_map     = out["probs"][0].cpu()
        total_pixels  = probs_map.shape[1] * probs_map.shape[2]

        detected_classes = [
            _BRIDGE_CLASSES[i] for i, p in enumerate(presence_mask.tolist())
            if p and _BRIDGE_CLASSES[i] != "Undamaged"
        ]
        coverage = {
            _BRIDGE_CLASSES[i]: round(float((probs_map[i] > threshold).sum()) / total_pixels * 100, 2)
            for i in range(len(_BRIDGE_CLASSES))
        }

        severity_score = float(self.model.compute_bridge_severity(
            out["class_map"], out["depth_map"]
        )[0].cpu())
        passability = _PASSABILITY_MAP[self.model.compute_bridge_passability(severity_score)]

        entry: dict = {
            "seg": {
                "presence": detected_classes,
                "coverage": coverage,
            },
            "severity":    {"score": round(severity_score, 4), "label": _severity_label(severity_score)},
            "passability": passability,
        }

        if use_vlm and self._bridge_vlm is not None:
            vlm_result = self._bridge_vlm.describe(
                img, out["class_map"][0].cpu(), severity_score, passability, coverage,
            )
            entry["reasoning"] = {
                "report":   vlm_result["report"],
                "detected": vlm_result["detected"],
            }

        return entry

    def run_overlay(
        self,
        images: list[Image.Image],
        threshold: float = 0.5,
        alpha: float = 0.4,
    ) -> list[bytes]:
        """Generate overlay images for input images.

        Args:
            images: List of PIL images
            threshold: Detection confidence threshold
            alpha: Overlay transparency (0-1)

        Returns:
            List of overlay images as bytes (PNG format)
        """
        overlays = []
        for img in images:
            x = self._bridge_tfm(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model.segment_bridge(x, with_depth=True)

            overlay_img = self.generate_overlay(img, out["class_map"][0], alpha)

            # Convert to bytes
            buffer = io.BytesIO()
            overlay_img.save(buffer, format="PNG")
            overlays.append(buffer.getvalue())

        return overlays
