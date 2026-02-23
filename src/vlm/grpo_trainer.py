"""RISE-R1 GRPO trainer for Qwen2-VL-2B on bridge/road inspection reasoning.

Reward signals (all in [0, 1]):
  - format_reward:      <think>...</think>[answer] structure present
  - passability_reward: predicted passability tier matches ground truth
  - grounding_reward:   fraction of ground-truth defect classes mentioned in <think>

Total reward = format_reward * (0.4 * passability_reward + 0.6 * grounding_reward)

Run with torchrun for multi-GPU:
  torchrun --nproc_per_node=3 -m src.vlm.grpo_trainer [args]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _to_text(completion) -> str:
    """Extract plain text from a completion that may be a string or message-dict list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(completion)


def _extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_answer(text: str) -> str:
    m = re.search(r"</think>(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def format_reward(completions: list[str], **_) -> list[float]:
    """1.0 if the completion contains <think>...</think> followed by non-empty answer."""
    completions = [_to_text(c) for c in completions]
    scores = []
    for c in completions:
        has_think = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        answer    = _extract_answer(c)
        scores.append(1.0 if (has_think and len(answer) > 10) else 0.0)
    return scores


_PASSABILITY_ALIASES: dict[str, str] = {
    "bisa":       "Bisa",
    "roda-2":     "Roda-2",
    "roda 2":     "Roda-2",
    "tidak bisa": "Tidak Bisa",
    "tidakbisa":  "Tidak Bisa",
}

def _normalise_passability(text: str) -> str | None:
    lower = text.lower()
    for alias, canonical in _PASSABILITY_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def passability_reward(completions: list[str], **kwargs) -> list[float]:
    """1.0 if the answer section predicts the correct passability tier."""
    completions = [_to_text(c) for c in completions]
    targets = kwargs.get("target", [{} for _ in completions])
    scores = []
    for c, t in zip(completions, targets):
        answer    = _extract_answer(c)
        predicted = _normalise_passability(answer)
        expected  = t.get("passability", "")
        scores.append(1.0 if predicted == expected else 0.0)
    return scores


def grounding_reward(completions: list[str], **kwargs) -> list[float]:
    """Recall of ground-truth defect class names mentioned anywhere in the completion."""
    completions = [_to_text(c) for c in completions]
    targets = kwargs.get("target", [{} for _ in completions])
    scores = []
    for c, t in zip(completions, targets):
        defects: list[str] = t.get("defects", [])
        if not defects:
            # No defects to ground — reward if model doesn't hallucinate damage
            scores.append(1.0)
            continue
        c_lower = c.lower()
        hits = sum(1 for d in defects if d.lower() in c_lower)
        scores.append(hits / len(defects))
    return scores


def combined_reward(completions: list[str], **kwargs) -> list[float]:
    fmt   = format_reward(completions)
    pass_ = passability_reward(completions, **kwargs)
    grnd  = grounding_reward(completions, **kwargs)
    return [
        f * (0.4 * p + 0.6 * g)
        for f, p, g in zip(fmt, pass_, grnd)
    ]


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def load_grpo_dataset(annotations_path: str):
    """Convert CoT JSONL to a HuggingFace Dataset for GRPOTrainer."""
    from datasets import Dataset
    from PIL import Image as PILImage

    records: list[dict] = []
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rows: list[dict[str, Any]] = []
    for r in records:
        image_path = r.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        img = PILImage.open(image_path).convert("RGB").resize((512, 512))
        problem = r["conversations"][0]["value"].replace("<image>\n", "")

        rows.append({
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": problem},
                    ],
                }
            ],
            "image":      img,
            "image_path": image_path,
            "answer":     r["conversations"][1]["value"],
            "target": {
                "passability": r.get("passability", ""),
                "defects":     r.get("defects", []),
                "severity":    r.get("severity", 0.0),
            },
        })

    logger.info("GRPO dataset: {} samples loaded from {}", len(rows), annotations_path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOVLMTrainer:
    """RISE-R1 GRPO trainer wrapping TRL's GRPOTrainer for Qwen2-VL."""

    def __init__(
        self,
        annotations_path: str = "data/vlm_cot_annotations.jsonl",
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        base_adapter: str | None = None,
        output_dir: str = "checkpoints/vlm_lora/grpo",
        num_generations: int = 8,
        max_prompt_length: int = 1024,
        max_new_tokens: int = 512,
        per_device_batch_size: int = 1,
        grad_accum: int = 2,
        learning_rate: float = 2e-6,
        max_steps: int = 200,
        save_steps: int = 10,
        val_fraction: float = 0.1,
    ) -> None:
        self.annotations_path      = annotations_path
        self.model_name            = model_name
        self.base_adapter          = base_adapter
        self.output_dir            = output_dir
        self.num_generations       = num_generations
        self.max_prompt_length     = max_prompt_length
        self.max_new_tokens        = max_new_tokens
        self.per_device_batch_size = per_device_batch_size
        self.grad_accum            = grad_accum
        self.learning_rate         = learning_rate
        self.max_steps             = max_steps
        self.save_steps            = save_steps
        self.val_fraction          = val_fraction
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from trl import GRPOConfig, GRPOTrainer

        # ── Load model ──────────────────────────────────────────────────
        logger.info("Loading base model: {}", self.model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        )

        if self.base_adapter:
            logger.info("Loading CoT SFT adapter from {}", self.base_adapter)
            model = PeftModel.from_pretrained(model, self.base_adapter, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # ── Dataset ─────────────────────────────────────────────────────
        full_ds = load_grpo_dataset(self.annotations_path)
        n_val   = max(1, int(len(full_ds) * self.val_fraction))
        train_ds = full_ds.select(range(len(full_ds) - n_val))
        eval_ds  = full_ds.select(range(len(full_ds) - n_val, len(full_ds)))
        logger.info("Train: {} | Eval: {}", len(train_ds), len(eval_ds))


        # ── GRPOConfig ───────────────────────────────────────────────────
        grpo_config = GRPOConfig(
            output_dir=self.output_dir,
            num_generations=self.num_generations,
            max_completion_length=self.max_new_tokens,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.learning_rate,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=self.save_steps,
            save_total_limit=2,
            max_steps=self.max_steps,
            report_to="wandb",
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            reward_funcs=[combined_reward],
        )

        logger.info("Starting GRPO training (max_steps={}, num_generations={})...",
                    self.max_steps, self.num_generations)
        trainer.train()

        final_dir = os.path.join(self.output_dir, "final_adapter")
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        logger.success("GRPO training complete. Adapter saved to {}", final_dir)


# ---------------------------------------------------------------------------
# CLI entry point (for torchrun)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RISE-R1 GRPO trainer for Qwen2-VL")
    p.add_argument("--annotations",    default="data/vlm_cot_annotations.jsonl")
    p.add_argument("--model-name",     default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--base-adapter",   default=None,
                   help="Path to CoT SFT LoRA adapter (Stage 2 output)")
    p.add_argument("--output-dir",     default="checkpoints/vlm_lora/grpo")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-new-tokens",  type=int, default=512)
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--grad-accum",     type=int, default=2)
    p.add_argument("--lr",             type=float, default=2e-6)
    p.add_argument("--max-steps",      type=int, default=200)
    p.add_argument("--save-steps",     type=int, default=10)
    p.add_argument("--val-fraction",   type=float, default=0.1)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    trainer = GRPOVLMTrainer(
        annotations_path=args.annotations,
        model_name=args.model_name,
        base_adapter=args.base_adapter,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        val_fraction=args.val_fraction,
    )
    trainer.run()
