import torch
import torch.nn as nn
import torch.nn.functional as F

_DEPTH_PRETRAINED = "Intel/dpt-large"


class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation.

    Addresses class imbalance by down-weighting easy examples.
    FL(p_t) = -(1 - p_t)^γ * log(p_t)

    Paper: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha_tensor = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  [B, C, H, W] logits
            targets: [B, H, W] class indices (0, 1, 2, or ignore_index)

        Returns:
            scalar loss
        """
        B, C, H, W = inputs.shape

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        # targets_one_hot: [B, C, H, W]

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Create mask for ignored indices
        mask = targets != self.ignore_index  # [B, H, W]

        # Compute focal loss
        ce = -targets_one_hot * torch.log(probs.clamp(min=1e-8))
        weight = torch.pow(1 - probs, self.gamma)
        focal = weight * ce

        # Sum over classes, average over pixels
        loss = focal.sum(dim=1)  # [B, H, W]

        # Apply ignore mask
        loss = loss * mask

        # Average over valid pixels
        return loss.sum() / mask.sum().clamp(min=1)

_BRIDGE_SEG_PRETRAINED = "nvidia/mit-b2"
_BRIDGE_SEG_CHECKPOINT = "checkpoints/bridge_seg_best.pt"
_N_BRIDGE_SEG_CLASSES = 3

_DS_WEIGHT = 0.65

_BRIDGE_PASSABILITY_BISA  = 0.3  # bridge: severity below this → all vehicles
_BRIDGE_PASSABILITY_RODA2 = 0.6  # bridge: severity below this → motorcycles only


class JatanMTL(nn.Module):
    """Damage assessment model for bridge and road infrastructure.

    SegFormer-B2 (EIDSeg, 3 classes) for pixel-level damage segmentation.
    DPT-Large for monocular depth estimation (depth-weighted severity).
    """

    def __init__(
        self,
        bridge_seg_checkpoint: str = _BRIDGE_SEG_CHECKPOINT,
    ) -> None:
        super().__init__()

        self.depth_model = self._load_depth_model()
        self.freeze_depth_model()

        self.bridge_seg_model = self._load_bridge_seg_model(bridge_seg_checkpoint)

    @staticmethod
    def _load_depth_model():
        from transformers import DPTForDepthEstimation

        return DPTForDepthEstimation.from_pretrained(_DEPTH_PRETRAINED)

    @staticmethod
    def _load_bridge_seg_model(checkpoint_path: str):
        import os
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(
            _BRIDGE_SEG_PRETRAINED,
            num_labels=_N_BRIDGE_SEG_CLASSES,
            ignore_mismatched_sizes=True,
        )
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        return model

    def freeze_depth_model(self) -> None:
        for p in self.depth_model.parameters():
            p.requires_grad = False

    def unfreeze_depth_model(self) -> None:
        for p in self.depth_model.parameters():
            p.requires_grad = True

    def freeze_bridge_seg_encoder(self) -> None:
        for p in self.bridge_seg_model.segformer.parameters():
            p.requires_grad = False

    def unfreeze_bridge_seg_encoder(self) -> None:
        for p in self.bridge_seg_model.segformer.parameters():
            p.requires_grad = True

    def segment_bridge(self, x: torch.Tensor, with_depth: bool = False) -> dict:
        """Run SegFormer-B2 bridge segmentation (EIDSeg, 3 classes).

        Args:
            x:          [B, 3, H, W]
            with_depth: also run DPT-Large depth estimation (for inference).
                        Keep False during training to avoid unnecessary overhead.

        Returns:
            seg_logits: [B, 3, 512, 512]
            probs:      [B, 3, 512, 512]  per-pixel softmax probabilities
            class_map:  [B, 512, 512]     argmax class index
            presence:   [B, 3]            bool — class detected anywhere in image
            depth_map:  [B, 512, 512]     normalised depth (only if with_depth=True)
        """
        x_resized = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        out = self.bridge_seg_model(pixel_values=x_resized)
        seg_logits = F.interpolate(
            out.logits, size=(512, 512), mode="bilinear", align_corners=False
        )
        probs     = torch.softmax(seg_logits, dim=1)   # [B, 3, 512, 512]
        class_map = probs.argmax(dim=1)               # [B, 512, 512]
        presence  = torch.stack(
            [class_map == c for c in range(probs.shape[1])], dim=1
        ).any(dim=(2, 3))                             # [B, 3]

        result = {"seg_logits": seg_logits, "probs": probs, "class_map": class_map, "presence": presence}

        if with_depth:
            depth_out = self.depth_model(pixel_values=x_resized)
            depth_map = F.interpolate(
                depth_out.predicted_depth.unsqueeze(1),
                size=(512, 512), mode="bilinear", align_corners=False,
            ).squeeze(1)
            result["depth_map"] = self._normalize_depth(depth_map)

        return result

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        """Normalise depth per image to [0.1, 1.0] (far pixels → higher weight)."""
        B = depth.shape[0]
        flat = depth.view(B, -1)
        d_min = flat.min(dim=1).values.view(B, 1, 1)
        d_max = flat.max(dim=1).values.view(B, 1, 1)
        return 0.1 + (depth - d_min) / (d_max - d_min + 1e-8) * 0.9

    @staticmethod
    def compute_bridge_severity(class_map: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Depth-weighted bridge damage severity score.

        Follows the volume ratio algorithm:
            numerator   = DS_weight × Σ(Damaged×depth) + Σ(Destroyed×depth)
            denominator = Σ(Damaged×depth) + Σ(Destroyed×depth) + Σ(Undamaged×depth)
            score       = numerator / denominator, clamped to [0, 1]

        Destroyed pixels contribute fully to numerator (impassable).
        Damaged pixels are weighted by _DS_WEIGHT (0.65).
        Undamaged pixels only appear in denominator (reduce severity).

        Args:
            class_map: [B, H, W] long — 0=Undamaged, 1=Damaged, 2=Destroyed
            depth_map: [B, H, W] float — normalised depth in [0.1, 1.0]

        Returns:
            [B] float severity scores in [0, 1]
        """
        damaged_w   = (class_map == 1).float() * depth_map   # DS
        destroyed_w = (class_map == 2).float() * depth_map   # Debris
        undamaged_w = (class_map == 0).float() * depth_map   # US

        numerator   = _DS_WEIGHT * damaged_w.sum(dim=(1, 2)) + destroyed_w.sum(dim=(1, 2))
        denominator = (damaged_w + destroyed_w + undamaged_w).sum(dim=(1, 2)).clamp(min=1e-8)
        return (numerator / denominator).clamp(max=1.0)

    @staticmethod
    def compute_bridge_passability(severity_score: float) -> str:
        """Map bridge severity score to transportation passability tier.

        < 0.3  → Bisa      (all vehicles)
        0.3–0.6 → Roda-2  (motorcycles only)
        ≥ 0.6  → Tidak Bisa (impassable)
        """
        if severity_score < _BRIDGE_PASSABILITY_BISA:
            return "Bisa"
        elif severity_score < _BRIDGE_PASSABILITY_RODA2:
            return "Roda-2"
        return "Tidak Bisa"
