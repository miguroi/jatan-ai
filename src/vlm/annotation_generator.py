from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.dataset import BridgeDataset, RoadDataset, _BRIDGE_CLASSES, _ROAD_CLASSES

_ROAD_DAMAGE_NAMES: dict[str, str] = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack (fatigue cracking)",
    "D40": "pothole",
}

_STRUCTURAL_CLASSES = {
    "Crack", "ACrack", "Spalling", "ExposedRebars",
    "Cavity", "Hollowareas", "Rockpocket",
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize(tensor: torch.Tensor) -> Image.Image:
    """Convert normalized (3,H,W) float tensor to PIL Image."""
    t = tensor.cpu().float() * _IMAGENET_STD + _IMAGENET_MEAN
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _extract_defect_metadata(mask: np.ndarray, class_names: list[str]) -> list[dict]:
    """
    mask: (N_classes, H, W) float32 binary mask.
    Returns list of per-class dicts sorted by pixel area descending.
    """
    _, H, W = mask.shape
    detections: list[dict] = []
    for i, name in enumerate(class_names):
        m = mask[i]
        pixel_area = int((m > 0.5).sum())
        if pixel_area == 0:
            continue
        coverage = round(pixel_area / (H * W) * 100, 2)
        ys, xs = np.where(m > 0.5)
        cy = float(ys.mean()) / H
        cx = float(xs.mean()) / W
        vert  = "upper" if cy < 0.33 else "lower" if cy > 0.67 else "middle"
        horiz = "left"  if cx < 0.33 else "right"  if cx > 0.67 else "center"
        detections.append({
            "class":       name,
            "pixel_area":  pixel_area,
            "coverage_pct": coverage,
            "region":      f"{vert}-{horiz}",
            "structural":  name in _STRUCTURAL_CLASSES,
        })
    detections.sort(key=lambda d: d["pixel_area"], reverse=True)
    return detections


def _compute_severity_score(detections: list[dict]) -> float:
    """
    Compute severity score (0-1) from detected bridge defects.

    Structural defects have higher weight. Coverage % is factored in.
    Formula: sum(structural_weight * coverage_pct) / 100, capped at 1.0
    """
    if not detections:
        return 0.0

    # Structural defects have weight 1.0, cosmetic have weight 0.2
    total = 0.0
    for d in detections:
        weight = 1.0 if d["structural"] else 0.2
        total += weight * d["coverage_pct"]

    return min(total / 100.0, 1.0)


def _severity_label(score: float) -> str:
    """Convert severity score to Indonesian label."""
    if score < 0.2:
        return "Ringan"
    elif score < 0.5:
        return "Sedang"
    return "Berat"


def _passability_from_severity(severity_score: float) -> str:
    """Map severity score to passability label (bridge and road)."""
    if severity_score < 0.3:
        return "Bisa"
    elif severity_score < 0.6:
        return "Roda-2"
    return "Tidak Bisa"


def _build_annotation_prompt(
    detections: list[dict],
    severity_score: float,
    passability: str,
) -> str:
    """Build annotation prompt with severity and passability context."""
    if not detections:
        return (
            "This bridge image shows no visible defects. "
            "Severity: Ringan (0.0). Passability: Bisa. "
            "Write a 1-2 sentence expert inspection report confirming no defects are observed."
        )

    lines = []
    for d in detections:
        tag = " [STRUCTURAL]" if d["structural"] else ""
        lines.append(
            f"- {d['class']}{tag}: {d['coverage_pct']}% coverage, "
            f"located in the {d['region']} region"
        )
    defect_list = "\n".join(lines)
    severity_label = _severity_label(severity_score)

    return (
        "You are a licensed bridge structural engineer conducting a visual inspection. "
        "The following defects have been detected by automated segmentation:\n\n"
        f"{defect_list}\n\n"
        f"Severity: {severity_label} ({severity_score:.2f}). "
        f"Passability: {passability}.\n\n"
        "Write a 1-2 sentence, actionable inspection report. "
        "Describe the observed damage and state the recommended action "
        "(e.g., 'immediate closure', 'schedule repair', 'monitor'). "
        "Be direct and concise. Do not repeat the defect list verbatim."
    )


class AnnotationGenerator:
    """Generates VLM training annotations from dacl10k masks via an OpenAI-compatible API."""

    def __init__(
        self,
        data_root: str = "data/dacl10k",
        output_path: str = "data/vlm_annotations.jsonl",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
        split: str = "train",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        request_delay: float = 1.0,
    ) -> None:
        self.data_root     = data_root
        self.output_path   = output_path
        self.api_key       = api_key
        self.base_url      = base_url
        self.model         = model
        self.split         = split
        self.max_retries   = max_retries
        self.retry_delay   = retry_delay
        self.request_delay = request_delay

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_existing_ids(self) -> set[str]:
        existing: set[str] = set()
        if not os.path.exists(self.output_path):
            return existing
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass
        return existing

    def _get_raw_image(self, dataset: BridgeDataset, idx: int) -> Image.Image:
        """3-tier fallback to obtain a PIL Image for the given dataset index."""
        inner = dataset._inner

        # Tier 1: image_files attribute
        if hasattr(inner, "image_files"):
            try:
                return Image.open(inner.image_files[idx]).convert("RGB")
            except Exception:
                pass

        # Tier 2: samples attribute
        if hasattr(inner, "samples"):
            try:
                s = inner.samples[idx]
                path = s[0] if isinstance(s, (tuple, list)) else s
                return Image.open(path).convert("RGB")
            except Exception:
                pass

        # Tier 3: denormalize prefetched tensor
        if getattr(inner, "use_prefetched_data", False) and inner.prefetched_data is not None:
            img_tensor, _ = inner.prefetched_data[idx]
            return _denormalize(img_tensor)

        # Final fallback: denormalize from dataset item
        return _denormalize(dataset[idx]["image"])

    def _image_to_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_api(self, img: Image.Image, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        b64 = self._image_to_base64(img)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "API error (attempt {}/{}): {}. Retrying in {:.0f}s...",
                        attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        existing_ids = self._load_existing_ids()
        logger.info("Loaded {} existing annotations — resuming.", len(existing_ids))

        dataset = BridgeDataset(split=self.split, data_root=self.data_root)
        logger.info("Dataset size: {} images (split={})", len(dataset), self.split)

        with open(self.output_path, "a") as out_f:
            for idx in range(len(dataset)):
                record_id = f"{self.split}_{idx:06d}"
                if record_id in existing_ids:
                    continue

                item = dataset[idx]
                mask = item["mask"].numpy()  # (19, H, W)
                detections = _extract_defect_metadata(mask, _BRIDGE_CLASSES)

                # Compute severity and passability
                severity_score = _compute_severity_score(detections)
                passability    = _passability_from_severity(severity_score)

                prompt     = _build_annotation_prompt(detections, severity_score, passability)
                defect_names = [d["class"] for d in detections]

                try:
                    img    = self._get_raw_image(dataset, idx)
                    report = self._call_api(img, prompt)
                except Exception as exc:
                    logger.error("Failed at index {}: {}", idx, exc)
                    continue

                # Resolve image path for the JSONL record
                inner = dataset._inner
                image_path = ""
                if hasattr(inner, "image_files"):
                    try:
                        image_path = str(inner.image_files[idx])
                    except Exception:
                        pass
                if not image_path and hasattr(inner, "samples"):
                    try:
                        s = inner.samples[idx]
                        image_path = str(s[0] if isinstance(s, (tuple, list)) else s)
                    except Exception:
                        pass

                record = {
                    "id":         record_id,
                    "image_path": image_path,
                    "defects":    defect_names,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": report},
                    ],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if (idx + 1) % 50 == 0:
                    logger.info("Progress: {}/{}", idx + 1, len(dataset))

                time.sleep(self.request_delay)

        logger.success("Annotation generation complete. Output: {}", self.output_path)


# ---------------------------------------------------------------------------
# Road annotation generator
# ---------------------------------------------------------------------------

def _compute_road_severity_score(detected_codes: list[str]) -> float:
    """
    Compute severity score (0-1) from detected road damage codes.

    D40 (pothole) = highest severity (1.0)
    D20 (alligator crack) = high (0.8)
    D10 (transverse crack) = medium (0.5)
    D00 (longitudinal crack) = low (0.3)
    Multiple defects: take max, then add 0.1 per additional defect (capped at 1.0)
    """
    if not detected_codes:
        return 0.0

    severity_map = {
        "D00": 0.3,
        "D10": 0.5,
        "D20": 0.8,
        "D40": 1.0,
    }
    max_severity = max(severity_map.get(c, 0.0) for c in detected_codes)
    additional = min(0.1 * (len(detected_codes) - 1), 0.2)
    return min(max_severity + additional, 1.0)


def _build_road_annotation_prompt(
    detected_names: list[str],
    severity_score: float,
    passability: str,
) -> str:
    """Build an expert prompt for road damage annotation."""
    if not detected_names:
        return (
            "This road image shows no visible damage. "
            f"Severity: Ringan ({0.0:.2f}). Passability: Bisa. "
            "Write a 1-2 sentence expert road condition report confirming no defects are observed."
        )
    damage_list = "\n".join(f"- {name}" for name in detected_names)
    severity_label = _severity_label(severity_score)

    return (
        "You are a licensed road infrastructure engineer conducting a pavement condition survey. "
        "The following damage types have been detected in this road image:\n\n"
        f"{damage_list}\n\n"
        f"Severity: {severity_label} ({severity_score:.2f}). "
        f"Passability: {passability}.\n\n"
        "Write a 1-2 sentence, actionable condition report. "
        "State the recommended action based on severity and passability "
        "(e.g., 'close to heavy vehicles', 'patch within 48h', 'monitor'). "
        "Be direct and concise."
    )


class RoadAnnotationGenerator:
    """Generates VLM training annotations from RDD2022 damage labels via an OpenAI-compatible API."""

    def __init__(
        self,
        data_root: str = "data/raw",
        output_path: str = "data/vlm_road_annotations.jsonl",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
        split: str = "train",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        request_delay: float = 1.0,
    ) -> None:
        self.data_root     = data_root
        self.output_path   = output_path
        self.api_key       = api_key
        self.base_url      = base_url
        self.model         = model
        self.split         = split
        self.max_retries   = max_retries
        self.retry_delay   = retry_delay
        self.request_delay = request_delay

    def _load_existing_ids(self) -> set[str]:
        existing: set[str] = set()
        if not os.path.exists(self.output_path):
            return existing
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass
        return existing

    def _image_to_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_api(self, img: Image.Image, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        b64 = self._image_to_base64(img)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "API error (attempt {}/{}): {}. Retrying in {:.0f}s...",
                        attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    def run(self) -> None:
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        existing_ids = self._load_existing_ids()
        logger.info("Loaded {} existing road annotations — resuming.", len(existing_ids))

        dataset = RoadDataset(split=self.split, data_root=self.data_root)
        logger.info("Road dataset size: {} images (split={})", len(dataset), self.split)

        with open(self.output_path, "a") as out_f:
            for idx in range(len(dataset)):
                record_id = f"road_{self.split}_{idx:06d}"
                if record_id in existing_ids:
                    continue

                item = dataset[idx]
                damage = item["damage"].numpy()  # (5,) binary: D00 D10 D20 D40 NoDefect

                detected_codes = [
                    _ROAD_CLASSES[i] for i, v in enumerate(damage)
                    if v > 0.5 and _ROAD_CLASSES[i] != "NoDefect"
                ]
                detected_names = [_ROAD_DAMAGE_NAMES[c] for c in detected_codes]

                # Compute severity and passability
                severity_score = _compute_road_severity_score(detected_codes)
                passability    = _passability_from_severity(severity_score)

                prompt = _build_road_annotation_prompt(detected_names, severity_score, passability)

                image_path = dataset._samples[idx]
                try:
                    img    = Image.open(image_path).convert("RGB")
                    report = self._call_api(img, prompt)
                except Exception as exc:
                    logger.error("Failed at index {}: {}", idx, exc)
                    continue

                record = {
                    "id":         record_id,
                    "image_path": str(image_path),
                    "defects":    detected_names,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": report},
                    ],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if (idx + 1) % 100 == 0:
                    logger.info("Progress: {}/{}", idx + 1, len(dataset))

                time.sleep(self.request_delay)

        logger.success("Road annotation generation complete. Output: {}", self.output_path)
