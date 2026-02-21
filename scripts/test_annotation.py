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
  python scripts/test_annotation.py \
    --domain bridge \
    --data-root data/dacl10k \
    --api-key "$OPENROUTER_API_KEY" \
    --n-samples 2

  # Road (RDD2022)
  python scripts/test_annotation.py \
    --domain road \
    --data-root data/raw \
    --api-key "$OPENROUTER_API_KEY" \
    --n-samples 2
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _wrapped(text: str, indent: int = 4) -> None:
    prefix = " " * indent
    for line in text.splitlines():
        wrapped = textwrap.fill(line, width=80, initial_indent=prefix,
                                subsequent_indent=prefix)
        print(wrapped if wrapped.strip() else "")


# ── Bridge test ───────────────────────────────────────────────────────────────

def test_bridge(args: argparse.Namespace) -> None:
    from src.dataset import BridgeDataset, _BRIDGE_CLASSES
    from src.vlm.annotation_generator import (
        AnnotationGenerator,
        _build_annotation_prompt,
        _extract_defect_metadata,
    )

    print(f"\n{'═' * 72}")
    print("  BRIDGE ANNOTATION TEST")
    print(f"  model : {args.model}")
    print(f"  split : {args.split}  |  n_samples : {args.n_samples}")
    print(f"{'═' * 72}")

    dataset = BridgeDataset(split=args.split, data_root=args.data_root)
    generator = AnnotationGenerator(
        data_root=args.data_root,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    for sample_idx in range(min(args.n_samples, len(dataset))):
        print(f"\n\n{'█' * 72}")
        print(f"  SAMPLE {sample_idx + 1} / {args.n_samples}")
        print(f"{'█' * 72}")

        item = dataset[sample_idx]
        mask = item["mask"].numpy()  # (19, H, W)

        # ── Step 1: detected defects ──────────────────────────────────────
        _section("STEP 1 — Detected defects (from segmentation mask)")
        detections = _extract_defect_metadata(mask, _BRIDGE_CLASSES)
        if not detections:
            print("    (none — all-zero mask)")
        for d in detections:
            flag = " [STRUCTURAL]" if d["structural"] else ""
            print(f"    • {d['class']:<20}{flag}")
            print(f"      coverage : {d['coverage_pct']:.2f}%")
            print(f"      region   : {d['region']}")
            print(f"      pixels   : {d['pixel_area']}")

        # ── Step 2: raw image path ────────────────────────────────────────
        _section("STEP 2 — Image source")
        try:
            img = generator._get_raw_image(dataset, sample_idx)
            print(f"    size   : {img.size}  mode : {img.mode}")
        except Exception as exc:
            print(f"    ERROR loading image: {exc}")
            continue

        # ── Step 3: prompt ────────────────────────────────────────────────
        _section("STEP 3 — Prompt sent to API")
        prompt = _build_annotation_prompt(detections)
        _wrapped(prompt)

        # ── Step 4: API response ──────────────────────────────────────────
        _section("STEP 4 — Model response")
        try:
            report = generator._call_api(img, prompt)
            _wrapped(report)
        except Exception as exc:
            print(f"    ERROR calling API: {exc}")
            continue

        # ── Grounding check hint ──────────────────────────────────────────
        _section("GROUNDING CHECK — Does the report mention these defects?")
        for d in detections:
            mentioned = d["class"].lower() in report.lower()
            tick = "✓" if mentioned else "✗ (not mentioned)"
            print(f"    {tick}  {d['class']}")


# ── Road test ─────────────────────────────────────────────────────────────────

def test_road(args: argparse.Namespace) -> None:
    from src.dataset import RoadDataset, _ROAD_CLASSES
    from src.vlm.annotation_generator import (
        RoadAnnotationGenerator,
        _ROAD_DAMAGE_NAMES,
        _build_road_annotation_prompt,
    )
    from PIL import Image

    print(f"\n{'═' * 72}")
    print("  ROAD ANNOTATION TEST")
    print(f"  model : {args.model}")
    print(f"  split : {args.split}  |  n_samples : {args.n_samples}")
    print(f"{'═' * 72}")

    dataset = RoadDataset(split=args.split, data_root=args.data_root)
    generator = RoadAnnotationGenerator(
        data_root=args.data_root,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    for sample_idx in range(min(args.n_samples, len(dataset))):
        print(f"\n\n{'█' * 72}")
        print(f"  SAMPLE {sample_idx + 1} / {args.n_samples}")
        print(f"{'█' * 72}")

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
            print("    (no damage — NoDefect)")
        for code, name in zip(detected_codes, detected_names):
            print(f"    • {code}  →  {name}")

        # ── Step 2: image path ────────────────────────────────────────────
        _section("STEP 2 — Image source")
        image_path = dataset._samples[sample_idx]
        print(f"    path : {image_path}")
        try:
            img = Image.open(image_path).convert("RGB")
            print(f"    size : {img.size}  mode : {img.mode}")
        except Exception as exc:
            print(f"    ERROR loading image: {exc}")
            continue

        # ── Step 3: prompt ────────────────────────────────────────────────
        _section("STEP 3 — Prompt sent to API")
        prompt = _build_road_annotation_prompt(detected_names)
        _wrapped(prompt)

        # ── Step 4: API response ──────────────────────────────────────────
        _section("STEP 4 — Model response")
        try:
            report = generator._call_api(img, prompt)
            _wrapped(report)
        except Exception as exc:
            print(f"    ERROR calling API: {exc}")
            continue

        # ── Grounding check ───────────────────────────────────────────────
        _section("GROUNDING CHECK — Does the report mention these damage types?")
        for name in detected_names:
            keyword = name.split()[0].lower()  # e.g. "longitudinal", "pothole"
            mentioned = keyword in report.lower()
            tick = "✓" if mentioned else "✗ (not mentioned)"
            print(f"    {tick}  {name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test annotation generation on a small sample."
    )
    parser.add_argument("--domain",     choices=["bridge", "road"], default="bridge")
    parser.add_argument("--data-root",  default="data/dacl10k",
                        help="data/dacl10k for bridge, data/raw for road")
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--base-url",   default="https://openrouter.ai/api/v1")
    parser.add_argument("--model",      default="qwen/qwen2-vl-72b-instruct")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--n-samples",  type=int, default=2,
                        help="Number of images to test (default: 2)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.domain == "bridge":
        test_bridge(args)
    else:
        test_road(args)


if __name__ == "__main__":
    main()
