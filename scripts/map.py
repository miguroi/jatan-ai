"""Generate segmentation overlay and depth map images for one or more input images.

Usage:
    uv run python scripts/map.py sample/jembatan_rusak_2.jpg
    uv run python scripts/map.py sample/*.jpg --output output/maps --checkpoint checkpoints/bridge_seg_best_focal_v2.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


_OVERLAY_COLORS = {
    0: (0, 255, 0),      # Undamaged — green
    1: (255, 165, 0),    # Damaged   — orange
    2: (255, 0, 0),      # Destroyed — red
}


def load_model(checkpoint: str, device: torch.device):
    from src.model import JatanMTL
    model = JatanMTL(bridge_seg_checkpoint=checkpoint).to(device)
    model.eval()
    return model


def process_image(
    img_path: str,
    model,
    device: torch.device,
    alpha: float = 0.5,
) -> tuple[Image.Image, Image.Image]:
    """Return (seg_overlay, depth_map) as PIL Images."""
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.segment_bridge(x, with_depth=True)

    # ── Segmentation overlay ─────────────────────────────────────────────────
    class_map = out["class_map"][0].cpu().numpy()   # [512, 512]
    img_resized = np.array(img.resize((512, 512)))

    color_mask = np.zeros_like(img_resized)
    for cls_idx, color in _OVERLAY_COLORS.items():
        mask = class_map == cls_idx
        color_mask[mask] = color

    blended = (img_resized * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    seg_overlay = Image.fromarray(blended)

    # ── Depth map ────────────────────────────────────────────────────────────
    depth = out["depth_map"][0].cpu().numpy()       # [0.1, 1.0]
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_norm)         # grayscale: brighter = farther

    return seg_overlay, depth_img


def get_random_dataset_images(data_root: str, split: str, n: int) -> list[str]:
    """Pick n random image paths from the EIDSeg dataset."""
    import random
    from src.dataset import EIDSegDataset
    ds = EIDSegDataset(split=split, data_root=data_root)
    samples = random.sample(ds._samples, min(n, len(ds._samples)))
    return [str(s["image_path"]) for s in samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation overlay and depth map images.")
    parser.add_argument("images", nargs="*", help="Input image path(s). Omit to sample from dataset.")
    parser.add_argument("--checkpoint", default="checkpoints/bridge_seg_best_focal_v2.pt")
    parser.add_argument("--output", default="output/maps", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency 0–1 (default: 0.5)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--data-root", default="data/raw/eidseg", help="EIDSeg data root (used with --random)")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to sample from (default: val)")
    parser.add_argument("--random", type=int, default=0, metavar="N", help="Sample N random images from the dataset")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    if args.random > 0:
        image_paths = get_random_dataset_images(args.data_root, args.split, args.random)
        print(f"Sampled {len(image_paths)} random image(s) from {args.split} split")
    elif args.images:
        image_paths = args.images
    else:
        parser.error("Provide image paths or use --random N to sample from the dataset.")

    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.checkpoint, device)

    for img_path in image_paths:
        stem = Path(img_path).stem
        print(f"Processing {img_path} ...")

        seg_overlay, depth_img = process_image(img_path, model, device, alpha=args.alpha)

        seg_path   = os.path.join(args.output, f"{stem}_seg.png")
        depth_path = os.path.join(args.output, f"{stem}_depth.png")

        seg_overlay.save(seg_path)
        depth_img.save(depth_path)

        print(f"  Segmentation → {seg_path}")
        print(f"  Depth map    → {depth_path}")

    print(f"\nDone. {len(image_paths)} image(s) processed → {args.output}/")


if __name__ == "__main__":
    main()
