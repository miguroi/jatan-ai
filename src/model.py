import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

_SEG_CHECKPOINT = "checkpoints/segformer_best_model.pth"
_SEG_PRETRAINED = "nvidia/segformer-b5-finetuned-ade-640-640"
_DEPTH_PRETRAINED = "Intel/dpt-large"
_N_SEG_CLASSES = 6

_BRIDGE_SEG_PRETRAINED = "nvidia/mit-b2"
_BRIDGE_SEG_CHECKPOINT = "checkpoints/bridge_seg_best.pt"
_N_BRIDGE_SEG_CLASSES = 19

_DS_WEIGHT = 0.65

_PASSABILITY_BISA  = 0.6   # road: clear fraction above which all vehicles can pass
_PASSABILITY_RODA2 = 0.2   # road: clear fraction above which motorcycles can pass

_BRIDGE_STRUCTURAL_CLASSES = {
    "Crack", "ACrack", "Spalling", "ExposedRebars",
    "Cavity", "Hollowareas", "Rockpocket",
}
_BRIDGE_PASSABILITY_BISA  = 0.3  # bridge: severity below this → all vehicles
_BRIDGE_PASSABILITY_RODA2 = 0.6  # bridge: severity below this → motorcycles only


class JatanMTL(nn.Module):
    """Multi-task learning model for road and bridge damage detection.

    Uses a road head for RDD2022 road damage types and a separate SegFormer-B2
    for bridge damage segmentation (dacl10k, 19 classes).
    A frozen SegFormer-B5 + frozen DPT provide depth-weighted damage severity
    segmentation for road images (paper Eq. 1).
    """

    def __init__(
        self,
        seg_checkpoint: str = _SEG_CHECKPOINT,
        bridge_seg_checkpoint: str = _BRIDGE_SEG_CHECKPOINT,
    ) -> None:
        super().__init__()

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.shared_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.road_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )

        self.seg_model = self._load_seg_model(seg_checkpoint)
        self.freeze_seg_model()

        self.depth_model = self._load_depth_model()
        self.freeze_depth_model()

        self.bridge_seg_model = self._load_bridge_seg_model(bridge_seg_checkpoint)

    @staticmethod
    def _load_seg_model(checkpoint_path: str):
        from transformers import SegformerForSemanticSegmentation

        seg = SegformerForSemanticSegmentation.from_pretrained(
            _SEG_PRETRAINED,
            num_labels=_N_SEG_CLASSES,
            ignore_mismatched_sizes=True,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
        clean_sd = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        seg.load_state_dict(clean_sd, strict=False)
        return seg

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

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_seg_model(self) -> None:
        for p in self.seg_model.parameters():
            p.requires_grad = False

    def unfreeze_seg_model(self) -> None:
        for p in self.seg_model.parameters():
            p.requires_grad = True

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


    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 384, 384]
            domain: "road" (bridge now uses segment_bridge())

        Returns:
            damage logits [B, 5] for road
        """
        feats = self.backbone(x)
        feats = feats.flatten(1)
        shared = self.shared_fc(feats)

        if domain == "road":
            return self.road_head(shared)
        elif domain == "bridge":
            raise ValueError(
                "Bridge classification is no longer supported via forward(). "
                "Use model.segment_bridge(x) for bridge damage segmentation."
            )
        else:
            raise ValueError(f"Invalid domain: {domain}. Must be 'road'")


    def segment(self, x: torch.Tensor) -> dict:
        """Run SegFormer + DPT on any image and compute paper Eq. 1 severity.

        Args:
            x: [B, 3, H, W]

        Returns:
            seg_logits: [B, 6, 640, 640]
            seg_map:    [B, 640, 640]  integer class indices 0–5
            depth_map:  [B, 640, 640]  normalised to [0.1, 1.0]
            severity:   [B]            damage severity score
        """
        x_resized = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False)

        seg_out = self.seg_model(pixel_values=x_resized)
        seg_logits = F.interpolate(
            seg_out.logits, size=(640, 640), mode="bilinear", align_corners=False
        )
        seg_map = seg_logits.argmax(dim=1)

        depth_out = self.depth_model(pixel_values=x_resized)
        depth_map = F.interpolate(
            depth_out.predicted_depth.unsqueeze(1),
            size=(640, 640), mode="bilinear", align_corners=False,
        ).squeeze(1)
        depth_map = self._normalize_depth(depth_map)

        return {
            "seg_logits":  seg_logits,
            "seg_map":     seg_map,
            "depth_map":   depth_map,
            "severity":    self.compute_severity(seg_map, depth_map),
            "passability": self.compute_passability(seg_map, depth_map),
        }

    def segment_bridge(self, x: torch.Tensor) -> dict:
        """Run SegFormer-B2 bridge segmentation (dacl10k, 19 classes).

        Args:
            x: [B, 3, H, W]

        Returns:
            seg_logits: [B, 19, 512, 512]
            probs:      [B, 19, 512, 512]  per-pixel sigmoid probabilities
            presence:   [B, 19]            bool — class detected if max prob > 0.5
        """
        x_resized = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        out = self.bridge_seg_model(pixel_values=x_resized)
        seg_logits = F.interpolate(
            out.logits, size=(512, 512), mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(seg_logits)             # [B, 19, 512, 512]
        presence = probs.amax(dim=(2, 3)) > 0.5       # [B, 19]
        return {"seg_logits": seg_logits, "probs": probs, "presence": presence}

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        """Normalise depth per image to [0.1, 1.0] (far pixels → higher weight)."""
        B = depth.shape[0]
        flat = depth.view(B, -1)
        d_min = flat.min(dim=1).values.view(B, 1, 1)
        d_max = flat.max(dim=1).values.view(B, 1, 1)
        return 0.1 + (depth - d_min) / (d_max - d_min + 1e-8) * 0.9

    @staticmethod
    def compute_passability(seg_map: torch.Tensor, depth_map: torch.Tensor) -> list[str]:
        """Depth-weighted clear road fraction → transportation modal passability.

        Clear fraction = Σ undamaged_road×depth / (Σ all_road×depth + Σ debris×depth)

        > 0.6  → Bisa      (all vehicles)
        0.2–0.6 → Roda-2   (motorcycle only)
        < 0.2  → Tidak Bisa (blocked)

        Debris (class 2) included in denominator as impassable obstruction.
        Background (class 5) excluded — not infrastructure.
        """
        undamaged = (seg_map == 3).float() * depth_map   # Undamaged Road
        damaged   = (seg_map == 4).float() * depth_map   # Damaged Road
        debris    = (seg_map == 2).float() * depth_map   # Destroyed/debris

        clear = undamaged.sum(dim=(1, 2))
        total = (undamaged + damaged + debris).sum(dim=(1, 2)).clamp(min=1e-8)

        fraction = clear / total

        result = []
        for f in fraction.tolist():
            if f > _PASSABILITY_BISA:
                result.append("Bisa")
            elif f > _PASSABILITY_RODA2:
                result.append("Roda-2")
            else:
                result.append("Tidak Bisa")
        return result

    @staticmethod
    def compute_bridge_severity(detected_classes: list[str], coverage: dict[str, float]) -> float:
        """Coverage-weighted structural severity score for bridge images.

        Structural defects (cracks, spalling, exposed rebars, etc.) carry weight 1.0;
        cosmetic defects (graffiti, weathering, etc.) carry weight 0.2.

        score = Σ (weight × coverage_pct) / 100, clamped to [0, 1]
        """
        total = sum(
            (1.0 if cls in _BRIDGE_STRUCTURAL_CLASSES else 0.2) * coverage.get(cls, 0.0)
            for cls in detected_classes
        )
        return min(total / 100.0, 1.0)

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

    @staticmethod
    def compute_severity(seg_map: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Depth-weighted road damage severity score.

        score = (DS_weight × Σ Damaged_Road×Depth)
                / (Σ Damaged_Road×Depth  +  Σ Undamaged_Road×Depth  +  Σ Background×Depth)

        Building classes (0, 1, 2) are excluded — road-only assessment.
        Background (class 5) acts as the undamaged reference in the denominator.
        """
        damaged_road   = (seg_map == 4).float() * depth_map
        undamaged_road = (seg_map == 3).float() * depth_map
        background     = (seg_map == 5).float() * depth_map

        ds  = damaged_road.sum(dim=(1, 2))
        us  = undamaged_road.sum(dim=(1, 2))
        bg  = background.sum(dim=(1, 2))

        return (_DS_WEIGHT * ds) / (ds + us + bg).clamp(min=1e-8)
