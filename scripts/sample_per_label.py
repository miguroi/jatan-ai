"""Pick one random sample per damage label from EIDSeg dataset.

Labels:
  0 — Undamaged
  1 — Damaged
  2 — Destroyed

A sample is assigned the label of its dominant class (highest pixel coverage,
excluding ignore pixels with value 255).

Usage:
    uv run python scripts/sample_per_label.py
    uv run python scripts/sample_per_label.py --split train --data-root data/raw/eidseg --seed 7
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from src.dataset import EIDSegDataset, _BRIDGE_CLASSES


def dominant_label(mask: np.ndarray) -> int:
    """Return the class index with the highest pixel count (ignoring 255)."""
    valid = mask[mask != 255]
    if len(valid) == 0:
        return 0
    counts = np.bincount(valid, minlength=len(_BRIDGE_CLASSES))
    return int(counts.argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one image per damage label.")
    parser.add_argument("--data-root", default="data/raw/eidseg")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("Loading EIDSeg {} split from {}", args.split, args.data_root)
    ds = EIDSegDataset(split=args.split, data_root=args.data_root)

    # Group indices by dominant label
    buckets: dict[int, list[int]] = {i: [] for i in range(len(_BRIDGE_CLASSES))}
    logger.info("Scanning {} samples...", len(ds))
    for idx, sample in enumerate(ds._samples):
        mask = ds._render_mask(sample)
        lbl = dominant_label(mask)
        buckets[lbl].append(idx)

    logger.info("Label distribution:")
    for lbl, indices in buckets.items():
        logger.info("  {} ({}): {} samples", lbl, _BRIDGE_CLASSES[lbl], len(indices))

    print()
    print("=== One random sample per label ===")
    for lbl in range(len(_BRIDGE_CLASSES)):
        indices = buckets[lbl]
        if not indices:
            logger.warning("No samples found for label {} ({})", lbl, _BRIDGE_CLASSES[lbl])
            continue
        idx = random.choice(indices)
        sample = ds._samples[idx]
        mask = ds._render_mask(sample)
        valid = mask[mask != 255]
        counts = np.bincount(valid, minlength=len(_BRIDGE_CLASSES))
        total = counts.sum()
        coverage = {_BRIDGE_CLASSES[i]: round(counts[i] / total * 100, 2) for i in range(len(_BRIDGE_CLASSES))}

        # Collect original polygon labels from XML
        orig_labels = sorted({p["label"] for p in sample["polygons"]})

        print(f"\nLabel {lbl} — {_BRIDGE_CLASSES[lbl]}")
        print(f"  Index         : {idx}")
        print(f"  Image path    : {sample['image_path']}")
        print(f"  Orig labels   : {orig_labels}")
        print(f"  Coverage      : {coverage}")


if __name__ == "__main__":
    main()
