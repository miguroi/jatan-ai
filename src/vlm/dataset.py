from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class VLMDataset(Dataset):
    """
    Dataset for Qwen2-VL LoRA finetuning on bridge inspection annotations.

    Loads ShareGPT-style JSONL produced by AnnotationGenerator and converts
    each record into tokenised inputs with label masking (human turn → -100).
    All images are resized to image_size×image_size before processor call so
    that pixel_values has a uniform shape across the batch.
    """

    def __init__(
        self,
        annotations_path: str,
        processor,
        split: str = "train",
        val_fraction: float = 0.1,
        image_size: int = 512,
    ) -> None:
        self.processor  = processor
        self.image_size = image_size

        records: list[dict] = []
        with open(annotations_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        n_val = max(1, int(len(records) * val_fraction))
        if split == "train":
            self.records = records[:-n_val]
        else:
            self.records = records[-n_val:]

        logger.info("VLMDataset split={} — {} records", split, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        # ── Load & resize image ──────────────────────────────────────────
        image_path = record.get("image_path", "")
        if image_path and Path(image_path).exists():
            img = (
                Image.open(image_path)
                .convert("RGB")
                .resize((self.image_size, self.image_size), Image.LANCZOS)
            )
        else:
            img = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        # ── Build chat messages ──────────────────────────────────────────
        human_text    = record["conversations"][0]["value"].replace("<image>\n", "")
        assistant_text = record["conversations"][1]["value"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": human_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )

        input_ids      = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values   = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values[0]

        # ── Label masking ────────────────────────────────────────────────
        # Mask everything up to (and including) the last <|im_start|> + "assistant\n"
        labels = input_ids.clone()
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions   = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

        if len(positions) > 0:
            last_pos   = positions[-1].item()
            mask_until = min(last_pos + 2, len(labels) - 1)
            labels[:mask_until] = -100
        else:
            # Fallback: mask first 80 % of tokens
            logger.debug("VLMDataset idx={}: <|im_start|> token not found; masking first 80%.", idx)
            labels[: int(len(labels) * 0.8)] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        return result
