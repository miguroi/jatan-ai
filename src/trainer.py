from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, hamming_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import CombinedDamageDataset, get_transform


class Trainer:
    """
    Orchestrates two-phase MTL training with validation, checkpointing, and
    early stopping.

    Args:
        model:          JatanMTL instance.
        device:         torch.device.
        data_root:      Root directory for raw data.
        batch_size:     DataLoader batch size.
        epochs1:        Epochs for Phase 1 (frozen backbone).
        epochs2:        Epochs for Phase 2 (full fine-tune).
        checkpoint_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        data_root: str = "data/raw",
        batch_size: int = 32,
        epochs1: int = 10,
        epochs2: int = 20,
        checkpoint_dir: str = "checkpoints",
        patience: int = 5,
        num_workers: int = min(4, os.cpu_count() or 1),
    ) -> None:
        self.model = model
        self.device = device
        self.data_root = data_root
        self.batch_size = batch_size
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.num_workers = num_workers

        os.makedirs(checkpoint_dir, exist_ok=True)

        self._bce = nn.BCEWithLogitsLoss()

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def run(self) -> None:
        """Orchestrate Phase 1 → Phase 2."""
        self._build_loaders()

        # --- Phase 1: frozen backbone ---
        self.model.freeze_backbone()
        trainable = (
            list(self.model.shared_fc.parameters())
            + list(self.model.asset_head.parameters())
            + list(self.model.damage_head.parameters())
            + [self.model.logsigma]
        )
        optimizer1 = torch.optim.SGD(
            trainable, lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self._phase("phase1", self.epochs1, optimizer1)

        # --- Phase 2: full fine-tune ---
        phase1_ckpt = os.path.join(self.checkpoint_dir, "phase1_best.pt")
        ckpt = torch.load(phase1_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.unfreeze_backbone()

        optimizer2 = torch.optim.SGD(
            [
                {"params": self.model.backbone.parameters(), "lr": 1e-4},
                {
                    "params": (
                        list(self.model.shared_fc.parameters())
                        + list(self.model.asset_head.parameters())
                        + list(self.model.damage_head.parameters())
                        + [self.model.logsigma]
                    ),
                    "lr": 1e-3,
                },
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )
        self._phase("phase2", self.epochs2, optimizer2)

    def _build_loaders(self) -> None:
        train_ds = CombinedDamageDataset(
            split="train", data_root=self.data_root, transform=get_transform("train")
        )
        val_ds = CombinedDamageDataset(
            split="val", data_root=self.data_root, transform=get_transform("val")
        )
        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=train_ds.get_sampler(),
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _phase(self, phase_name: str, epochs: int, optimizer: torch.optim.Optimizer) -> None:
        """Run a full training phase with validation and checkpointing."""
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_loss = float("inf")
        patience_counter = 0

        ckpt_filename = (
            "phase1_best.pt" if phase_name == "phase1" else "best_model.pt"
        )
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_filename)

        for epoch in range(epochs):
            logger.info("[{}] Epoch {}/{}", phase_name, epoch + 1, epochs)

            train_metrics = self._train_epoch(self._train_loader, optimizer)
            val_metrics   = self.validate(self._val_loader)

            scheduler.step()

            val_loss = val_metrics["val_loss"]
            logger.info(
                "train_loss={:.4f} val_loss={:.4f} asset_acc={:.4f} damage_macro_f1={:.4f}",
                train_metrics["train_loss"],
                val_loss,
                val_metrics["asset_acc"],
                val_metrics["damage_macro_f1"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(ckpt_path, epoch, optimizer, val_metrics, phase_name)
                logger.success("Saved best checkpoint -> {}", ckpt_path)
            else:
                patience_counter += 1

            if phase_name == "phase2" and patience_counter >= self.patience:
                logger.warning("Early stopping triggered after {} epochs without improvement.", self.patience)
                break

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            images     = batch["image"].to(self.device)
            asset_tgt  = batch["asset_type"].to(self.device)
            damage_tgt = batch["damage_types"].to(self.device)

            optimizer.zero_grad()

            asset_logits, damage_logits = self.model(images)

            asset_loss  = self._bce(asset_logits, asset_tgt.unsqueeze(1).float())
            damage_loss = self._bce(damage_logits, damage_tgt)

            loss = self._compute_loss(asset_loss, damage_loss)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return {"train_loss": running_loss / max(n_batches, 1)}

    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        all_asset_preds:  list[np.ndarray] = []
        all_asset_tgts:   list[np.ndarray] = []
        all_damage_preds: list[np.ndarray] = []
        all_damage_tgts:  list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                images     = batch["image"].to(self.device)
                asset_tgt  = batch["asset_type"].to(self.device)
                damage_tgt = batch["damage_types"].to(self.device)

                asset_logits, damage_logits = self.model(images)

                asset_loss  = self._bce(asset_logits, asset_tgt.unsqueeze(1).float())
                damage_loss = self._bce(damage_logits, damage_tgt)
                loss = self._compute_loss(asset_loss, damage_loss)

                running_loss += loss.item()
                n_batches += 1

                asset_pred  = (torch.sigmoid(asset_logits) >= 0.5).long().squeeze(1)
                damage_pred = (torch.sigmoid(damage_logits) >= 0.5).float()

                all_asset_preds.append(asset_pred.cpu().numpy())
                all_asset_tgts.append(asset_tgt.cpu().numpy())
                all_damage_preds.append(damage_pred.cpu().numpy())
                all_damage_tgts.append(damage_tgt.cpu().numpy())

        asset_preds  = np.concatenate(all_asset_preds)
        asset_tgts   = np.concatenate(all_asset_tgts)
        damage_preds = np.concatenate(all_damage_preds, axis=0)
        damage_tgts  = np.concatenate(all_damage_tgts, axis=0)

        asset_acc = float((asset_preds == asset_tgts).mean())
        asset_f1  = float(f1_score(asset_tgts, asset_preds, average="binary", zero_division=0))

        damage_macro_f1 = float(f1_score(damage_tgts, damage_preds, average="macro", zero_division=0))
        damage_micro_f1 = float(f1_score(damage_tgts, damage_preds, average="micro", zero_division=0))
        damage_hamming  = float(hamming_loss(damage_tgts, damage_preds))
        per_class_f1    = f1_score(damage_tgts, damage_preds, average=None, zero_division=0).tolist()

        return {
            "val_loss":        running_loss / max(n_batches, 1),
            "asset_acc":       asset_acc,
            "asset_f1":        asset_f1,
            "damage_macro_f1": damage_macro_f1,
            "damage_micro_f1": damage_micro_f1,
            "damage_hamming":  damage_hamming,
            "per_class_f1":    per_class_f1,
        }

    def _compute_loss(
        self,
        asset_loss: torch.Tensor,
        damage_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Uncertainty-weighted loss (Kendall et al., 2018).

        Automatically balances the 1-term asset loss vs. 7-term damage loss
        via the learned logsigma parameters.
        """
        task_losses = [asset_loss, damage_loss]
        return sum(
            1 / (2 * torch.exp(self.model.logsigma[i])) * task_losses[i]
            + self.model.logsigma[i] / 2
            for i in range(2)
        )

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metrics: dict,
        phase: str,
    ) -> None:
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             metrics["val_loss"],
                "metrics":              metrics,
                "phase":                phase,
            },
            path,
        )

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer) -> int:
        """Load a checkpoint into the model and optimizer. Returns the saved epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]

