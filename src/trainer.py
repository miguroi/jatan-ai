from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BridgeDataset, CombinedDamageDataset, EIDSegDataset, get_transform


def custom_collate_fn(batch):
    images  = torch.stack([item["image"] for item in batch])
    domains = torch.stack([item["domain"] for item in batch])
    damages = [item.get("damage") for item in batch]
    masks   = [item.get("mask")   for item in batch]
    return {"image": images, "domain": domains, "damage": damages, "mask": masks}


class Trainer:
    """Road-only MTL trainer (ResNet50 → road_head)."""

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
        self._build_loaders()

        self.model.freeze_backbone()
        trainable = (
            list(self.model.shared_fc.parameters())
            + list(self.model.road_head.parameters())
        )
        optimizer1 = torch.optim.SGD(
            trainable, lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self._phase("phase1", self.epochs1, optimizer1)

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
                        + list(self.model.road_head.parameters())
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
            collate_fn=custom_collate_fn,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

    def _phase(self, phase_name: str, epochs: int, optimizer: torch.optim.Optimizer) -> None:
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
            val_metrics = self.validate(self._val_loader)

            scheduler.step()

            val_loss = val_metrics["val_loss"]
            logger.info(
                "train_loss={:.4f} val_loss={:.4f} road_f1={:.4f}",
                train_metrics["train_loss"],
                val_loss,
                val_metrics["road_macro_f1"],
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

        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            domains = batch["domain"]
            damages = batch["damage"]

            optimizer.zero_grad()

            road_idx = (domains == 0).nonzero(as_tuple=True)[0]

            loss = torch.tensor(0.0, device=self.device)

            if len(road_idx) > 0:
                road_imgs = images[road_idx]
                road_tgt = torch.stack([damages[i] for i in road_idx]).to(self.device)
                road_logits = self.model(road_imgs, "road")
                loss = loss + self._bce(road_logits, road_tgt)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": running_loss / max(n_batches, 1)}

    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        road_preds = []
        road_tgts = []

        pbar = tqdm(loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                domains = batch["domain"]
                damages = batch["damage"]

                road_idx = (domains == 0).nonzero(as_tuple=True)[0]

                loss = torch.tensor(0.0, device=self.device)

                if len(road_idx) > 0:
                    road_imgs = images[road_idx]
                    road_tgt = torch.stack([damages[i] for i in road_idx]).to(self.device)
                    road_logits = self.model(road_imgs, "road")
                    loss = loss + self._bce(road_logits, road_tgt)
                    road_pred = (torch.sigmoid(road_logits) >= 0.5).float()
                    road_preds.append(road_pred.cpu().numpy())
                    road_tgts.append(road_tgt.cpu().numpy())

                running_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        road_macro_f1 = 0.0
        if road_preds:
            road_preds = np.concatenate(road_preds, axis=0)
            road_tgts = np.concatenate(road_tgts, axis=0)
            road_macro_f1 = float(f1_score(road_tgts, road_preds, average="macro", zero_division=0))

        return {
            "val_loss": running_loss / max(n_batches, 1),
            "road_macro_f1": road_macro_f1,
        }

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
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": metrics["val_loss"],
                "metrics": metrics,
                "phase": phase,
            },
            path,
        )


class EIDSegBridgeTrainer:
    """Two-phase trainer for SegFormer-B2 bridge segmentation on EIDSeg (3 classes).

    Classes: 0=Undamaged, 1=Damaged, 2=Destroyed, 255=Ignore
    Loss: CrossEntropyLoss(ignore_index=255)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        data_root: str = "data/raw/eidseg",
        batch_size: int = 8,
        epochs1: int = 10,
        epochs2: int = 20,
        checkpoint_dir: str = "checkpoints",
        patience: int = 5,
        num_workers: int = min(4, os.cpu_count() or 1),
        resume_phase2: bool = False,
    ) -> None:
        self.model        = model
        self.device       = device
        self.data_root    = data_root
        self.batch_size   = batch_size
        self.epochs1      = epochs1
        self.epochs2      = epochs2
        self.checkpoint_dir = checkpoint_dir
        self.patience     = patience
        self.num_workers  = num_workers
        self.resume_phase2 = resume_phase2
        self.ckpt_path    = os.path.join(checkpoint_dir, "bridge_seg_best.pt")
        self._scaler      = torch.amp.GradScaler("cuda")
        self._ce          = nn.CrossEntropyLoss(ignore_index=255)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def run(self) -> None:
        train_ds = EIDSegDataset(split="train", data_root=self.data_root)
        val_ds   = EIDSegDataset(split="val",   data_root=self.data_root)

        self._train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        self._val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

        if self.resume_phase2:
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"--resume-phase2 requested but no checkpoint at {self.ckpt_path}"
                )
            self.model.bridge_seg_model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device), strict=False
            )
            logger.info("Loaded Phase 1 checkpoint — skipping Phase 1")
        else:
            self.model.freeze_bridge_seg_encoder()
            opt1 = torch.optim.AdamW(
                self.model.bridge_seg_model.decode_head.parameters(), lr=1e-3
            )
            logger.info("EIDSegBridgeTrainer Phase 1: decode_head only")
            self._train_phase(opt1, self.epochs1, patience=None)

        self.model.unfreeze_bridge_seg_encoder()
        opt2 = torch.optim.AdamW([
            {"params": self.model.bridge_seg_model.segformer.parameters(),    "lr": 1e-4},
            {"params": self.model.bridge_seg_model.decode_head.parameters(),  "lr": 1e-3},
        ])
        logger.info("EIDSegBridgeTrainer Phase 2: full fine-tune")
        self._train_phase(opt2, self.epochs2, patience=self.patience)

    def _train_phase(self, optimizer, epochs: int, patience: Optional[int]) -> None:
        scheduler       = CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_loss   = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            logger.info("Epoch {}/{}", epoch + 1, epochs)
            train_loss = self._train_epoch(optimizer)
            val_loss, val_miou = self._validate()
            scheduler.step()
            logger.info(
                "train_loss={:.4f} val_loss={:.4f} val_mIoU={:.4f}",
                train_loss, val_loss, val_miou,
            )
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(self.model.bridge_seg_model.state_dict(), self.ckpt_path)
                logger.success("Saved checkpoint → {}", self.ckpt_path)
            else:
                patience_counter += 1
            if patience is not None and patience_counter >= patience:
                logger.warning("Early stopping triggered.")
                break

    def _train_epoch(self, optimizer) -> float:
        self.model.bridge_seg_model.train()
        running_loss = 0.0
        n_batches    = 0
        pbar = tqdm(self._train_loader, desc="Training bridge seg", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)   # [B, 3, 512, 512]
            masks  = batch["mask"].to(self.device)    # [B, 512, 512] long

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out  = self.model.segment_bridge(images)
                loss = self._ce(out["seg_logits"], masks)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.bridge_seg_model.parameters(), max_norm=1.0
            )
            self._scaler.step(optimizer)
            self._scaler.update()

            running_loss += loss.item()
            n_batches    += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return running_loss / max(n_batches, 1)

    def _validate(self) -> tuple[float, float]:
        self.model.bridge_seg_model.eval()
        running_loss = 0.0
        running_miou = 0.0
        n_batches    = 0
        pbar = tqdm(self._val_loader, desc="Validating bridge seg", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)
                with torch.amp.autocast("cuda"):
                    out  = self.model.segment_bridge(images)
                    loss = self._ce(out["seg_logits"], masks)

                class_map = out["class_map"]  # [B, H, W]
                valid     = masks != 255
                ious = []
                for c in range(3):
                    pred_c   = (class_map == c) & valid
                    target_c = (masks == c) & valid
                    inter    = (pred_c & target_c).sum().float()
                    union    = (pred_c | target_c).sum().float()
                    if union > 0:
                        ious.append((inter / union).item())
                miou = float(np.mean(ious)) if ious else 0.0

                running_loss += loss.item()
                running_miou += miou
                n_batches    += 1
        return (
            running_loss / max(n_batches, 1),
            running_miou / max(n_batches, 1),
        )


class BridgeSegTrainer:
    """Two-phase trainer for SegFormer-B2 bridge segmentation (dacl10k, 19 classes).

    Phase 1: freeze encoder, train decode_head only (AdamW lr=1e-3).
    Phase 2: unfreeze encoder, differential LR + CosineAnnealingLR.

    Resume logic: if checkpoints/bridge_seg_best.pt exists and was saved during
    Phase 1, --resume-phase2 skips Phase 1 and loads those weights directly.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        data_root: str = "data/dacl10k",
        batch_size: int = 16,
        epochs1: int = 10,
        epochs2: int = 20,
        checkpoint_dir: str = "checkpoints",
        patience: int = 5,
        num_workers: int = min(4, os.cpu_count() or 1),
        resume_phase2: bool = False,
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
        self.resume_phase2 = resume_phase2
        self.ckpt_path = os.path.join(checkpoint_dir, "bridge_seg_best.pt")
        self._scaler = torch.amp.GradScaler("cuda")

        os.makedirs(checkpoint_dir, exist_ok=True)

    def run(self) -> None:
        train_ds = BridgeDataset(split="train", data_root=self.data_root)
        val_ds   = BridgeDataset(split="val",   data_root=self.data_root)

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

        if self.resume_phase2:
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"--resume-phase2 requested but no checkpoint found at {self.ckpt_path}"
                )
            state_dict = torch.load(self.ckpt_path, map_location=self.device)
            self.model.bridge_seg_model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded Phase 1 checkpoint from {} — skipping Phase 1", self.ckpt_path)
        else:
            # Phase 1: decode head only
            self.model.freeze_bridge_seg_encoder()
            optimizer1 = torch.optim.AdamW(
                self.model.bridge_seg_model.decode_head.parameters(), lr=1e-3
            )
            logger.info("BridgeSegTrainer Phase 1: training decode_head only")
            self._train_phase(optimizer1, self.epochs1, patience=None)

        # Phase 2: full fine-tune with differential LR
        self.model.unfreeze_bridge_seg_encoder()
        optimizer2 = torch.optim.AdamW(
            [
                {"params": self.model.bridge_seg_model.segformer.parameters(), "lr": 1e-4},
                {"params": self.model.bridge_seg_model.decode_head.parameters(), "lr": 1e-3},
            ]
        )
        logger.info("BridgeSegTrainer Phase 2: full fine-tune with CosineAnnealingLR")
        self._train_phase(optimizer2, self.epochs2, patience=self.patience)

    def _train_phase(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        patience: Optional[int],
    ) -> None:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            logger.info("Epoch {}/{}", epoch + 1, epochs)
            train_loss = self._train_epoch(optimizer)
            val_loss, val_miou = self._validate()
            scheduler.step()

            logger.info(
                "train_loss={:.4f} val_loss={:.4f} val_mIoU={:.4f}",
                train_loss, val_loss, val_miou,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.bridge_seg_model.state_dict(),
                    self.ckpt_path,
                )
                logger.success("Saved best bridge seg checkpoint -> {}", self.ckpt_path)
            else:
                patience_counter += 1

            if patience is not None and patience_counter >= patience:
                logger.warning("Early stopping triggered.")
                break

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.bridge_seg_model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(self._train_loader, desc="Training bridge seg", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks_list = batch["mask"]
            masks = torch.stack([m for m in masks_list if m is not None]).to(self.device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = self.model.segment_bridge(images)
                loss = F.binary_cross_entropy_with_logits(out["seg_logits"], masks)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.bridge_seg_model.parameters(), max_norm=1.0
            )
            self._scaler.step(optimizer)
            self._scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / max(n_batches, 1)

    def _validate(self) -> tuple[float, float]:
        self.model.bridge_seg_model.eval()
        running_loss = 0.0
        running_miou = 0.0
        n_batches = 0

        pbar = tqdm(self._val_loader, desc="Validating bridge seg", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks_list = batch["mask"]
                masks = torch.stack([m for m in masks_list if m is not None]).to(self.device)

                with torch.amp.autocast("cuda"):
                    out = self.model.segment_bridge(images)
                    loss = F.binary_cross_entropy_with_logits(out["seg_logits"], masks)

                preds   = (out["probs"] > 0.5).float()
                targets = masks
                inter   = (preds * targets).sum(dim=(2, 3))
                union   = ((preds + targets) > 0).float().sum(dim=(2, 3))
                present = union > 0
                iou     = inter[present] / union[present]
                miou    = iou.mean() if present.any() else torch.tensor(0.0, device=self.device)

                running_loss += loss.item()
                running_miou += miou.item()
                n_batches += 1

        return (
            running_loss / max(n_batches, 1),
            running_miou / max(n_batches, 1),
        )
