"""Compare annotation quality across multiple VLM models on EIDSeg samples.

Models compared:
  1. Qwen2.5-VL-72B    (qwen/qwen2.5-vl-72b-instruct)
  2. InternVL3-78B     (opengvlab/internvl3-78b)
  3. Llama 3.2 90B     (meta-llama/llama-3.2-90b-vision-instruct)

Output:
  - data/vlm_comparison.jsonl   — one record per image, responses from all models
  - data/vlm_comparison.csv     — flat table for easy inspection

Usage:
    uv run python scripts/compare_vlm.py \
        --api-key $OPENROUTER_API_KEY \
        --samples 20 \
        --split val \
        --output data/vlm_comparison
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from loguru import logger

from src.dataset import EIDSegDataset, _BRIDGE_CLASSES
from src.vlm.annotation_generator import _build_eidseg_prompt


MODELS = {
    "qwen2.5-vl-72b":    "qwen/qwen2.5-vl-72b-instruct",
    "internvl3-78b":     "opengvlab/internvl3-78b",
    "llama-3.2-90b":     "meta-llama/llama-3.2-90b-vision-instruct",
}


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_model(
    img: Image.Image,
    prompt: str,
    model_id: str,
    api_key: str,
    base_url: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    b64 = _image_to_base64(img)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=512,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning("Model {} error (attempt {}/{}): {}. Retrying in {:.0f}s...",
                               model_id, attempt + 1, max_retries, exc, wait)
                time.sleep(wait)
            else:
                logger.error("Model {} failed after {} attempts: {}", model_id, max_retries, exc)
                return f"[ERROR: {exc}]"


def _compute_severity(mask, _DS_W: float = 0.75) -> tuple[float, str, str]:
    """Compute depth-free coverage-based severity and passability from mask."""
    valid = mask != 255
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        return 0.0, "Ringan", "Bisa"

    coverage = {
        cls: float((mask == i).sum()) / valid_pixels * 100
        for i, cls in enumerate(_BRIDGE_CLASSES)
    }
    detected = [cls for cls in ["Damaged", "Destroyed"] if coverage.get(cls, 0.0) > 0]

    # Simple coverage-based severity (no depth model needed for comparison script)
    score = min(
        (_DS_W * coverage.get("Damaged", 0.0) + coverage.get("Destroyed", 0.0)) / 100,
        1.0,
    )
    severity_label = "Ringan" if score < 0.2 else "Sedang" if score < 0.5 else "Berat"
    passability    = "Bisa" if score < 0.3 else "Roda-2" if score < 0.6 else "Tidak Bisa"
    return score, severity_label, passability, coverage, detected


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM annotation quality across models.")
    parser.add_argument("--api-key",   default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--base-url",  default="https://openrouter.ai/api/v1")
    parser.add_argument("--data-root", default="data/raw/eidseg")
    parser.add_argument("--split",     default="val", choices=["train", "val"])
    parser.add_argument("--samples",   type=int, default=10, help="Number of random samples (default: 10)")
    parser.add_argument("--output",    default="data/vlm_comparison", help="Output path prefix (no extension)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--request-delay", type=float, default=1.5, help="Delay between API calls (default: 1.5s)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or pass --api-key.")
        sys.exit(1)

    random.seed(args.seed)

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading EIDSeg {} split from {}", args.split, args.data_root)
    ds = EIDSegDataset(split=args.split, data_root=args.data_root)
    indices = random.sample(range(len(ds)), min(args.samples, len(ds)))
    logger.info("Selected {} samples (seed={})", len(indices), args.seed)

    # ── Prepare output ────────────────────────────────────────────────────────
    jsonl_path = args.output + ".jsonl"
    csv_path   = args.output + ".csv"
    os.makedirs(Path(jsonl_path).parent, exist_ok=True)

    csv_fields = ["id", "image_path", "detected", "severity_score", "passability"] + list(MODELS.keys())
    csv_rows   = []

    with open(jsonl_path, "w") as jf:
        for rank, idx in enumerate(indices, 1):
            sample     = ds._samples[idx]
            image_path = str(sample["image_path"])
            item       = ds[idx]
            mask       = item["mask"].numpy()

            score, severity_label, passability, coverage, detected = _compute_severity(mask)
            prompt = _build_eidseg_prompt(detected, coverage, score, passability, cot=False)

            logger.info("[{}/{}] {} | severity={:.2f} passability={}",
                        rank, len(indices), Path(image_path).name, score, passability)

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error("Cannot open {}: {}", image_path, e)
                continue

            record: dict = {
                "id":            f"{args.split}_{idx:06d}",
                "image_path":    image_path,
                "detected":      detected,
                "coverage":      {k: round(v, 2) for k, v in coverage.items()},
                "severity_score": round(score, 4),
                "severity_label": severity_label,
                "passability":   passability,
                "prompt":        prompt,
                "responses":     {},
            }

            csv_row = {
                "id":            record["id"],
                "image_path":    image_path,
                "detected":      ", ".join(detected) if detected else "none",
                "severity_score": round(score, 4),
                "passability":   passability,
            }

            for model_name, model_id in MODELS.items():
                logger.info("  Querying {} ...", model_name)
                response = _call_model(img, prompt, model_id, api_key, args.base_url)
                record["responses"][model_name] = response
                csv_row[model_name] = response
                time.sleep(args.request_delay)

            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()
            csv_rows.append(csv_row)

            logger.success("  Done [{}/{}]", rank, len(indices))

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.success("Done. {} samples compared.", len(csv_rows))
    logger.success("JSONL → {}", jsonl_path)
    logger.success("CSV  → {}", csv_path)


if __name__ == "__main__":
    main()
