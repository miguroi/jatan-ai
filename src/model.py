import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class JatanMTL(nn.Module):
    """Multi-task learning model for road and bridge damage detection.

    Uses separate task heads for bridge (dacl1k) and road (RDD2022) damage types.
    Shared backbone learns general damage features across domains.
    """

    def __init__(self) -> None:
        super().__init__()

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.shared_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.bridge_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 6),
        )

        self.road_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )

    def forward(
        self, x: torch.Tensor, domain: str
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 384, 384]
            domain: "bridge" or "road" - which head to use

        Returns:
            damage_logits [B, 4] for bridge or [B, 7] for road
        """
        feats = self.backbone(x)
        feats = feats.flatten(1)
        shared = self.shared_fc(feats)

        if domain == "bridge":
            return self.bridge_head(shared)
        elif domain == "road":
            return self.road_head(shared)
        else:
            raise ValueError(f"Invalid domain: {domain}. Must be 'bridge' or 'road'")

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
