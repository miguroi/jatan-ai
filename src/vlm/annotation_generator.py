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

from src.dataset import BridgeDataset, EIDSegDataset, RoadDataset, _BRIDGE_CLASSES, _ROAD_CLASSES

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
    cot: bool = False,
) -> str:
    """Build annotation prompt with post-disaster emergency triage context."""
    if not detections:
        if cot:
            return (
                "You are a field assessor conducting emergency post-disaster structural triage. "
                "Automated analysis of this bridge image detected no structural damage. "
                "Severity: Ringan (0.0). Passability: Bisa.\n\n"
                "Think step by step through the visible structural condition, then write a "
                "1-2 sentence triage report for emergency response teams. "
                "Format your response exactly as:\n"
                "<think>\n[your step-by-step reasoning]\n</think>\n"
                "[your final 1-2 sentence triage report]"
            )
        return (
            "You are a field assessor conducting emergency post-disaster structural triage. "
            "Automated analysis of this bridge image detected no structural damage. "
            "Severity: Ringan (0.0). Passability: Bisa. "
            "Write a 1-2 sentence triage report confirming the bridge is safe to cross."
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

    if cot:
        return (
            "You are a field assessor conducting emergency post-disaster structural triage. "
            "Your report will be used by emergency response teams to make immediate "
            "access and evacuation decisions. "
            "Automated analysis detected the following bridge damage:\n\n"
            f"{defect_list}\n\n"
            f"Severity: {severity_label} ({severity_score:.2f}). "
            f"Passability: {passability}.\n\n"
            "Think step by step: assess each damage type for immediate safety risk, "
            "consider load-bearing implications, and determine the access status. "
            "Then write a 1-2 sentence triage report.\n\n"
            "Format your response exactly as:\n"
            "<think>\n[your step-by-step reasoning]\n</think>\n"
            "[your final 1-2 sentence triage report with immediate access recommendation "
            "(e.g., 'do not cross', 'motorcycles only', 'proceed with caution', 'safe to cross')]"
        )

    return (
        "You are a field assessor conducting emergency post-disaster structural triage. "
        "Your report will be used by emergency response teams to make immediate "
        "access and evacuation decisions. "
        "Automated analysis detected the following bridge damage:\n\n"
        f"{defect_list}\n\n"
        f"Severity: {severity_label} ({severity_score:.2f}). "
        f"Passability: {passability}.\n\n"
        "Write a 1-2 sentence triage report stating the immediate access status and action. "
        "Use field-ready language "
        "(e.g., 'do not cross', 'motorcycles only', 'proceed with caution', 'safe to cross'). "
        "Be direct. Do not repeat the defect list verbatim."
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
        cot: bool = False,
        max_samples: int | None = None,
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
        self.cot           = cot
        self.max_samples   = max_samples

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

    def _get_image_path(self, dataset: BridgeDataset, idx: int) -> str:
        """Get the image file path for the given dataset index."""
        inner = dataset._inner

        if hasattr(inner, "image_files"):
            try:
                return str(inner.image_files[idx])
            except Exception:
                pass

        if hasattr(inner, "samples"):
            try:
                s = inner.samples[idx]
                path = s[0] if isinstance(s, (tuple, list)) else s
                return str(path)
            except Exception:
                pass

        return f"<unknown index {idx}>"
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
        max_tokens = 1024 if self.cot else 512

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
                    max_tokens=max_tokens,
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

    @staticmethod
    def _split_cot_response(raw: str) -> tuple[str, str]:
        """Split a CoT response into (reasoning, annotation).

        Expects format: <think>...</think>[annotation]
        Falls back to ("", raw) if the tag is absent.
        """
        import re
        m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if m:
            reasoning  = m.group(1).strip()
            annotation = m.group(2).strip()
        else:
            reasoning  = ""
            annotation = raw
        return reasoning, annotation

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

        if self.max_samples is not None:
            logger.info("max_samples={} (including {} already written)", self.max_samples, len(existing_ids))

        n_written = len(existing_ids)

        with open(self.output_path, "a") as out_f:
            for idx in range(len(dataset)):
                if self.max_samples is not None and n_written >= self.max_samples:
                    logger.info("Reached max_samples={} — stopping.", self.max_samples)
                    break

                record_id = f"{self.split}_{idx:06d}"
                if record_id in existing_ids:
                    continue

                item = dataset[idx]
                mask = item["mask"].numpy()  # (19, H, W)
                detections = _extract_defect_metadata(mask, _BRIDGE_CLASSES)

                # Compute severity and passability
                severity_score = _compute_severity_score(detections)
                passability    = _passability_from_severity(severity_score)

                prompt     = _build_annotation_prompt(detections, severity_score, passability, cot=self.cot)
                defect_names = [d["class"] for d in detections]

                try:
                    img    = self._get_raw_image(dataset, idx)
                    raw    = self._call_api(img, prompt)
                except Exception as exc:
                    logger.error("Failed at index {}: {}", idx, exc)
                    continue

                if self.cot:
                    cot_reasoning, report = self._split_cot_response(raw)
                else:
                    cot_reasoning, report = "", raw

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

                record: dict = {
                    "id":         record_id,
                    "image_path": image_path,
                    "defects":    defect_names,
                    "severity":   round(severity_score, 4),
                    "passability": passability,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": report},
                    ],
                }
                if self.cot and cot_reasoning:
                    record["cot_reasoning"] = cot_reasoning
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1

                if n_written % 50 == 0:
                    limit_str = f"/{self.max_samples}" if self.max_samples is not None else ""
                    logger.info("Progress: {}{}", n_written, limit_str)

                time.sleep(self.request_delay)

        logger.success("Annotation generation complete. {} annotations written. Output: {}", n_written, self.output_path)


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
    cot: bool = False,
) -> str:
    """Build a post-disaster emergency triage prompt for road damage annotation."""
    if not detected_names:
        if cot:
            return (
                "You are a field assessor conducting emergency post-disaster road triage. "
                "Automated analysis of this road image detected no damage. "
                f"Severity: Ringan ({0.0:.2f}). Passability: Bisa.\n\n"
                "Think step by step through the road surface condition, then write a "
                "1-2 sentence triage report for emergency response teams. "
                "Format your response exactly as:\n"
                "<think>\n[your step-by-step reasoning]\n</think>\n"
                "[your final 1-2 sentence triage report]"
            )
        return (
            "You are a field assessor conducting emergency post-disaster road triage. "
            "Automated analysis of this road image detected no damage. "
            f"Severity: Ringan ({0.0:.2f}). Passability: Bisa. "
            "Write a 1-2 sentence triage report confirming the road is passable."
        )
    damage_list = "\n".join(f"- {name}" for name in detected_names)
    severity_label = _severity_label(severity_score)

    if cot:
        return (
            "You are a field assessor conducting emergency post-disaster road triage. "
            "Your report will be used by emergency response teams to make immediate "
            "routing and evacuation decisions. "
            "Automated analysis detected the following road damage:\n\n"
            f"{damage_list}\n\n"
            f"Severity: {severity_label} ({severity_score:.2f}). "
            f"Passability: {passability}.\n\n"
            "Think step by step: assess each damage type for immediate safety risk, "
            "consider vehicle load implications, and determine the access status. "
            "Then write a 1-2 sentence triage report.\n\n"
            "Format your response exactly as:\n"
            "<think>\n[your step-by-step reasoning]\n</think>\n"
            "[your final 1-2 sentence triage report with immediate routing recommendation "
            "(e.g., 'road closed', 'motorcycles only', 'avoid heavy vehicles', 'passable')]"
        )

    return (
        "You are a field assessor conducting emergency post-disaster road triage. "
        "Your report will be used by emergency response teams to make immediate "
        "routing and evacuation decisions. "
        "Automated analysis detected the following road damage:\n\n"
        f"{damage_list}\n\n"
        f"Severity: {severity_label} ({severity_score:.2f}). "
        f"Passability: {passability}.\n\n"
        "Write a 1-2 sentence triage report stating the immediate routing status and action. "
        "Use field-ready language "
        "(e.g., 'road closed', 'motorcycles only', 'avoid heavy vehicles', 'passable'). "
        "Be direct. Do not repeat the damage list verbatim."
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
        cot: bool = False,
        max_samples: int | None = None,
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
        self.cot           = cot
        self.max_samples   = max_samples

    def _get_image_path(self, dataset, idx: int) -> str:
        """Get the image file path for the given dataset index."""
        return str(dataset._samples[idx])

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
        max_tokens = 1024 if self.cot else 512

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
                    max_tokens=max_tokens,
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

        n_written = len(existing_ids)
        limit_str = f"/{self.max_samples}" if self.max_samples is not None else ""

        with open(self.output_path, "a") as out_f:
            for idx in range(len(dataset)):
                if self.max_samples is not None and n_written >= self.max_samples:
                    logger.info("Reached max_samples={} — stopping.", self.max_samples)
                    break

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

                prompt = _build_road_annotation_prompt(detected_names, severity_score, passability, cot=self.cot)

                image_path = dataset._samples[idx]
                try:
                    img = Image.open(image_path).convert("RGB")
                    raw = self._call_api(img, prompt)
                except Exception as exc:
                    logger.error("Failed at index {}: {}", idx, exc)
                    continue

                if self.cot:
                    cot_reasoning, report = AnnotationGenerator._split_cot_response(raw)
                else:
                    cot_reasoning, report = "", raw

                record: dict = {
                    "id":         record_id,
                    "image_path": str(image_path),
                    "defects":    detected_names,
                    "severity":   round(severity_score, 4),
                    "passability": passability,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": report},
                    ],
                }
                if self.cot and cot_reasoning:
                    record["cot_reasoning"] = cot_reasoning
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1

                if n_written % 100 == 0:
                    logger.info("Progress: {}{}", n_written, limit_str)

                time.sleep(self.request_delay)

        logger.success("Road annotation generation complete. {} annotations written. Output: {}", n_written, self.output_path)


# ---------------------------------------------------------------------------
# EIDSeg annotation generator (bridge + road, unified 3-class pipeline)
# ---------------------------------------------------------------------------

def _build_eidseg_prompt(
    detected: list[str],
    coverage: dict[str, float],
    severity_score: float,
    passability: str,
    cot: bool = False,
) -> str:
    """Build triage prompt matching the BridgeVLMInference inference prompt format."""
    severity_label = (
        "Ringan" if severity_score < 0.2 else
        "Sedang" if severity_score < 0.5 else
        "Berat"
    )
    if not detected:
        if cot:
            return (
                "You are a field assessor conducting emergency post-disaster structural triage. "
                "Automated analysis of this image detected no damage. "
                f"Severity: {severity_label} (0.00). Passability: {passability}.\n\n"
                "Think step by step through the visible condition, then write a "
                "1-2 sentence triage report for emergency response teams. "
                "Format your response exactly as:\n"
                "<think>\n[your step-by-step reasoning]\n</think>\n"
                "[your final 1-2 sentence triage report]"
            )
        return (
            "You are a field assessor conducting emergency post-disaster structural triage. "
            "Automated analysis of this image detected no damage. "
            f"Severity: {severity_label} (0.00). Passability: {passability}. "
            "Provide a brief triage report confirming the infrastructure is safe to use."
        )

    lines = [
        f"- {cls}: {coverage.get(cls, 0.0):.2f}% coverage"
        for cls in detected
    ]
    defect_summary = "\n".join(lines)

    if cot:
        return (
            "You are a field assessor conducting emergency post-disaster structural triage. "
            "Your report will be used by emergency response teams to make immediate "
            "access and evacuation decisions. "
            "Automated analysis detected the following damage "
            "(colour overlays indicate affected areas):\n\n"
            f"{defect_summary}\n\n"
            f"Severity: {severity_label} ({severity_score:.2f}). "
            f"Passability: {passability}.\n\n"
            "Think step by step: assess the damage extent and immediate safety risk, "
            "then determine the access status. "
            "Format your response exactly as:\n"
            "<think>\n[your step-by-step reasoning]\n</think>\n"
            "[your final 1-2 sentence triage report with immediate access recommendation "
            "(e.g., 'do not cross', 'motorcycles only', 'proceed with caution', 'safe to cross')]"
        )

    return (
        "You are a field assessor conducting emergency post-disaster structural triage. "
        "Your report will be used by emergency response teams to make immediate "
        "access and evacuation decisions. "
        "Automated analysis detected the following damage "
        "(colour overlays indicate affected areas):\n\n"
        f"{defect_summary}\n\n"
        f"Severity: {severity_label} ({severity_score:.2f}). "
        f"Passability: {passability}.\n\n"
        "Write a 1-2 sentence triage report stating the immediate access status and action. "
        "Use field-ready language "
        "(e.g., 'do not cross', 'motorcycles only', 'proceed with caution', 'safe to cross'). "
        "Be direct. Do not repeat the damage list verbatim."
    )


class EIDSegAnnotationGenerator:
    """Generates VLM training annotations from EIDSeg ground-truth masks.

    Uses 3-class labels (Undamaged/Damaged/Destroyed) matching the unified
    inference pipeline. Works for both bridge and road domain images.
    """

    def __init__(
        self,
        data_root: str = "data/raw/eidseg",
        output_path: str = "data/vlm_annotations_eidseg.jsonl",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen3-vl-32b-instruct",
        split: str = "train",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        request_delay: float = 1.0,
        cot: bool = False,
        max_samples: int | None = None,
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
        self.cot           = cot
        self.max_samples   = max_samples

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
        max_tokens = 1024 if self.cot else 512

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
                    max_tokens=max_tokens,
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
        logger.info("Loaded {} existing annotations — resuming.", len(existing_ids))

        dataset = EIDSegDataset(split=self.split, data_root=self.data_root)
        logger.info("EIDSeg dataset size: {} images (split={})", len(dataset), self.split)

        n_written = len(existing_ids)

        with open(self.output_path, "a") as out_f:
            for idx in range(len(dataset)):
                if self.max_samples is not None and n_written >= self.max_samples:
                    logger.info("Reached max_samples={} — stopping.", self.max_samples)
                    break

                record_id = f"eidseg_{self.split}_{idx:06d}"
                if record_id in existing_ids:
                    continue

                sample     = dataset._samples[idx]
                image_path = str(sample["image_path"])

                item       = dataset[idx]
                mask       = item["mask"]  # (H, W) long, values 0/1/2/255

                # Coverage over valid (non-ignore) pixels
                valid        = mask != 255
                valid_pixels = int(valid.sum())
                if valid_pixels == 0:
                    continue

                coverage = {
                    cls: round(float((mask == i).sum()) / valid_pixels * 100, 2)
                    for i, cls in enumerate(_BRIDGE_CLASSES)
                }
                detected = [
                    cls for cls in ["Damaged", "Destroyed"] if coverage.get(cls, 0.0) > 0
                ]
                severity_score = min(
                    coverage.get("Destroyed", 0.0) * 2.0 / 100
                    + coverage.get("Damaged",   0.0) * 0.5 / 100,
                    1.0,
                )
                passability = (
                    "Bisa"       if severity_score < 0.3 else
                    "Roda-2"     if severity_score < 0.6 else
                    "Tidak Bisa"
                )

                prompt = _build_eidseg_prompt(detected, coverage, severity_score, passability, cot=self.cot)

                try:
                    img = Image.open(image_path).convert("RGB")
                    raw = self._call_api(img, prompt)
                except Exception as exc:
                    logger.error("Failed at index {}: {}", idx, exc)
                    continue

                if self.cot:
                    cot_reasoning, report = AnnotationGenerator._split_cot_response(raw)
                else:
                    cot_reasoning, report = "", raw

                record: dict = {
                    "id":          record_id,
                    "image_path":  image_path,
                    "defects":     detected,
                    "severity":    round(severity_score, 4),
                    "passability": passability,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": report},
                    ],
                }
                if self.cot and cot_reasoning:
                    record["cot_reasoning"] = cot_reasoning

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1

                if n_written % 50 == 0:
                    limit_str = f"/{self.max_samples}" if self.max_samples is not None else ""
                    logger.info("Progress: {}{}", n_written, limit_str)

                time.sleep(self.request_delay)

        logger.success(
            "EIDSeg annotation generation complete. {} annotations written. Output: {}",
            n_written, self.output_path,
        )
