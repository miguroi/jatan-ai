"""
Test annotation generation on a small sample (default: 2 images).

Prints the full pipeline for each sample so you can verify the output
is grounded and not hallucinating:

  [1] Detected defects (from mask / damage labels)
  [2] Extracted metadata (coverage %, region, structural flag)
  [3] Exact prompt sent to the API
  [4] Raw model response

Usage:
  # Bridge (dacl10k)
  python scripts/test_annotation.py \\
    --domain bridge \\
    --data-root data/dacl10k \\
    --n-samples 2

  # Road (RDD2022)
  python scripts/test_annotation.py \\
    --domain road \\
    --data-root data/raw \\
    --n-samples 2
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 72
    logger.info(f"\n{'─' * width}")
    logger.info(f"  {title}")
    logger.info(f"{'─' * width}")


def _wrapped(text: str, indent: int = 4) -> None:
    prefix = " " * indent
    for line in text.splitlines():
        wrapped = textwrap.fill(line, width=80, initial_indent=prefix,
                                subsequent_indent=prefix)
        logger.info(wrapped if wrapped.strip() else "")


# ── Bridge test ───────────────────────────────────────────────────────────────

def test_bridge(args: argparse.Namespace) -> None:
    from src.dataset import BridgeDataset, _BRIDGE_CLASSES
    from src.vlm.annotation_generator import (
        AnnotationGenerator,
        _build_annotation_prompt,
        _extract_defect_metadata,
        _compute_severity_score,
        _severity_label,
        _passability_from_severity,
    )

    logger.info(f"\n{'═' * 72}")
    logger.info("  BRIDGE ANNOTATION TEST")
    logger.info("  model : {}", args.model)
    logger.info("  split : {}  |  n_samples : {}", args.split, args.n_samples)
    logger.info(f"{'═' * 72}")

    logger.info("Loading BridgeDataset from {} (split={})", args.data_root, args.split)
    dataset = BridgeDataset(split=args.split, data_root=args.data_root)
    logger.success("BridgeDataset loaded: {} images", len(dataset))

    generator = AnnotationGenerator(
        data_root=args.data_root,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    for sample_idx in range(min(args.n_samples, len(dataset))):
        logger.info(f"\n\n{'█' * 72}")
        logger.info("  SAMPLE {} / {}", sample_idx + 1, args.n_samples)
        logger.info(f"{'█' * 72}")

        item = dataset[sample_idx]
        mask = item["mask"].numpy()  # (19, H, W)

        # ── Step 1: detected defects ──────────────────────────────────────
        _section("STEP 1 — Detected defects (from segmentation mask)")
        detections = _extract_defect_metadata(mask, _BRIDGE_CLASSES)
        if not detections:
            logger.info("    (none — all-zero mask)")
        for d in detections:
            flag = " [STRUCTURAL]" if d["structural"] else ""
            logger.info(f"    • {d['class']:<20}{flag}")
            logger.info("      coverage : {:.2f}%", d['coverage_pct'])
            logger.info("      region   : {}", d['region'])
            logger.info("      pixels   : {}", d['pixel_area'])

        # ── Step 2: raw image path ────────────────────────────────────────
        _section("STEP 2 — Image source")
        try:
            img = generator._get_raw_image(dataset, sample_idx)
            logger.info("    size   : {}  mode : {}", img.size, img.mode)
        except Exception as exc:
            logger.error("    ERROR loading image: {}", exc)
            continue

        # ── Step 3: severity, passability, prompt ─────────────────────────
        _section("STEP 3 — Severity & Passability")
        severity_score = _compute_severity_score(detections)
        passability    = _passability_from_severity(severity_score)
        severity_label = _severity_label(severity_score)
        logger.info("    severity score : {:.2f} / 1.0", severity_score)
        logger.info("    severity label : {}", severity_label)
        logger.info("    passability    : {}", passability)

        _section("STEP 4 — Prompt sent to API")
        prompt = _build_annotation_prompt(detections, severity_score, passability)
        _wrapped(prompt)

        # ── Step 5: API response ──────────────────────────────────────────
        _section("STEP 5 — Model response")
        try:
            logger.info("Calling API (this may take 10-30s)...")
            report = generator._call_api(img, prompt)
            _wrapped(report)
            logger.success("API call complete")
        except Exception as exc:
            logger.error("    ERROR calling API: {}", exc)
            continue

        # ── Grounding check hint ──────────────────────────────────────────
        _section("GROUNDING CHECK — Does the report mention these defects?")
        for d in detections:
            mentioned = d["class"].lower() in report.lower()
            tick = "✓" if mentioned else "✗ (not mentioned)"
            logger.info("    {}  {}", tick, d['class'])


# ── Road test ─────────────────────────────────────────────────────────────────

def test_road(args: argparse.Namespace) -> None:
    from src.dataset import RoadDataset, _ROAD_CLASSES
    from src.vlm.annotation_generator import (
        RoadAnnotationGenerator,
        _ROAD_DAMAGE_NAMES,
        _build_road_annotation_prompt,
        _compute_road_severity_score,
        _severity_label,
        _passability_from_severity,
    )
    from PIL import Image

    logger.info(f"\n{'═' * 72}")
    logger.info("  ROAD ANNOTATION TEST")
    logger.info("  model : {}", args.model)
    logger.info("  split : {}  |  n_samples : {}", args.split, args.n_samples)
    logger.info(f"{'═' * 72}")

    logger.info("Loading RoadDataset from {} (split={})", args.data_root, args.split)
    dataset = RoadDataset(split=args.split, data_root=args.data_root)
    logger.success("RoadDataset loaded: {} images", len(dataset))

    generator = RoadAnnotationGenerator(
        data_root=args.data_root,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    for sample_idx in range(min(args.n_samples, len(dataset))):
        logger.info(f"\n\n{'█' * 72}")
        logger.info("  SAMPLE {} / {}", sample_idx + 1, args.n_samples)
        logger.info(f"{'█' * 72}")

        item = dataset[sample_idx]
        damage = item["damage"].numpy()  # (5,)

        # ── Step 1: detected damage ───────────────────────────────────────
        _section("STEP 1 — Detected damage (from classification labels)")
        detected_codes = [
            _ROAD_CLASSES[i] for i, v in enumerate(damage)
            if v > 0.5 and _ROAD_CLASSES[i] != "NoDefect"
        ]
        detected_names = [_ROAD_DAMAGE_NAMES[c] for c in detected_codes]
        if not detected_codes:
            logger.info("    (no damage — NoDefect)")
        for code, name in zip(detected_codes, detected_names):
            logger.info("    • {}  →  {}", code, name)

        # ── Step 2: image path ────────────────────────────────────────────
        _section("STEP 2 — Image source")
        image_path = dataset._samples[sample_idx]
        logger.info("    path : {}", image_path)
        try:
            img = Image.open(image_path).convert("RGB")
            logger.info("    size : {}  mode : {}", img.size, img.mode)
        except Exception as exc:
            logger.error("    ERROR loading image: {}", exc)
            continue

        # ── Step 3: severity, passability, prompt ─────────────────────────
        _section("STEP 3 — Severity & Passability")
        severity_score = _compute_road_severity_score(detected_codes)
        passability    = _passability_from_severity(severity_score)
        severity_label = _severity_label(severity_score)
        logger.info("    severity score : {:.2f} / 1.0", severity_score)
        logger.info("    severity label : {}", severity_label)
        logger.info("    passability    : {}", passability)

        _section("STEP 4 — Prompt sent to API")
        prompt = _build_road_annotation_prompt(detected_names, severity_score, passability)
        _wrapped(prompt)

        # ── Step 5: API response ──────────────────────────────────────────
        _section("STEP 5 — Model response")
        try:
            logger.info("Calling API (this may take 10-30s)...")
            report = generator._call_api(img, prompt)
            _wrapped(report)
            logger.success("API call complete")
        except Exception as exc:
            logger.error("    ERROR calling API: {}", exc)
            continue

        # ── Grounding check ───────────────────────────────────────────────
        _section("GROUNDING CHECK — Does the report mention these damage types?")
        for name in detected_names:
            keyword = name.split()[0].lower()  # e.g. "longitudinal", "pothole"
            mentioned = keyword in report.lower()
            tick = "✓" if mentioned else "✗ (not mentioned)"
            logger.info("    {}  {}", tick, name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test annotation generation on a small sample."
    )
    parser.add_argument("--domain",     choices=["bridge", "road"], default="bridge")
    parser.add_argument("--data-root",  default="data/dacl10k",
                        help="data/dacl10k for bridge, data/raw for road")
    parser.add_argument("--api-key",    default=None,
                        help="API key (default: OPENROUTER_API_KEY from .env)")
    parser.add_argument("--base-url",   default="https://openrouter.ai/api/v1")
    parser.add_argument("--model",      default="qwen/qwen2-vl-72b-instruct")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--n-samples",  type=int, default=2,
                        help="Number of images to test (default: 2)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key:
        args.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.api_key:
        logger.error("No API key found. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        sys.exit(1)

    if args.domain == "bridge":
        test_bridge(args)
    else:
        test_road(args)


if __name__ == "__main__":
    main()
