import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class JatanMTL(nn.Module):
    """Multi-task learning model for road and bridge damage detection."""

    def __init__(self) -> None:
        super().__init__()

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B, 2048, 1, 1]

        self.shared_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.asset_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        self.damage_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, 384, 384]

        Returns:
            (asset_logits [B, 1], damage_logits [B, 7]) — raw logits, no sigmoid
        """
        feats = self.backbone(x)          # [B, 2048, 1, 1]
        feats = feats.flatten(1)          # [B, 2048]
        shared = self.shared_fc(feats)    # [B, 1024]

        asset_logits  = self.asset_head(shared)   # [B, 1]
        damage_logits = self.damage_head(shared)  # [B, 7]

        return asset_logits, damage_logits

    def shared_modules(self) -> list[nn.Module]:
        """Return modules shared across tasks (used for gradient management)."""
        return [self.backbone, self.shared_fc]

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (Phase 1: train heads only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (Phase 2: full fine-tune)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
