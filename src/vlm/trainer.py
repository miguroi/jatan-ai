from __future__ import annotations

import os

from loguru import logger


class VLMTrainer:
    """
    LoRA finetuner for Qwen2-VL-2B-Instruct on bridge inspection annotations.

    LoRA targets only the LM attention projections (q/k/v/o_proj); the vision
    encoder stays frozen.  Training uses TRL SFTTrainer with bf16 + gradient
    checkpointing.  device_map="auto" distributes across all available GPUs.
    """

    def __init__(
        self,
        annotations_path: str = "data/vlm_annotations.jsonl",
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        output_dir: str = "checkpoints/vlm_lora",
        batch_size: int = 4,
        grad_accum: int = 4,
        epochs: int = 3,
        lr: float = 5e-5,
        val_fraction: float = 0.1,
        cot: bool = False,
    ) -> None:
        self.annotations_path  = annotations_path
        self.model_name        = model_name
        self.output_dir        = output_dir
        self.batch_size        = batch_size
        self.grad_accum        = grad_accum
        self.epochs            = epochs
        self.lr                = lr
        self.val_fraction      = val_fraction
        self.cot               = cot
        self.final_adapter_dir = os.path.join(output_dir, "final_adapter")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.final_adapter_dir, exist_ok=True)

    def run(self) -> None:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from trl import SFTConfig, SFTTrainer

        from src.vlm.dataset import VLMDataset

        # ── Load model & processor ───────────────────────────────────────
        logger.info("Loading processor and model: {}", self.model_name)
        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        # ── Apply LoRA (LM layers only; ~0.28% trainable) ───────────────
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

        # ── Datasets ─────────────────────────────────────────────────────
        logger.info("Building train/val datasets from {}", self.annotations_path)
        train_ds = VLMDataset(
            self.annotations_path,
            processor=processor,
            split="train",
            val_fraction=self.val_fraction,
            cot=self.cot,
        )
        val_ds = VLMDataset(
            self.annotations_path,
            processor=processor,
            split="val",
            val_fraction=self.val_fraction,
            cot=self.cot,
        )
        logger.info("Train: {} samples | Val: {} samples", len(train_ds), len(val_ds))

        # ── SFTConfig ────────────────────────────────────────────────────
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            gradient_checkpointing=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,  # required for multimodal
            logging_steps=10,
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        # ── Train ────────────────────────────────────────────────────────
        logger.info("Starting VLM LoRA finetuning...")
        trainer.train()

        # ── Save adapter ─────────────────────────────────────────────────
        logger.info("Saving final LoRA adapter to {}", self.final_adapter_dir)
        model.save_pretrained(self.final_adapter_dir)
        processor.save_pretrained(self.final_adapter_dir)
        logger.success("VLM training complete. Adapter saved to {}", self.final_adapter_dir)
