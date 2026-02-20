from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from bikit.datasets import BikitDataset as _BikitDataset

UNIFIED_CLASSES: List[str] = [
    "Crack",         # 0
    "Pothole",       # 1
    "Spalling",      # 2
    "Efflorescence", # 3
    "Rust",          # 4
    "ExposedBars",   # 5
    "NoDefect",      # 6
]
N_UNIFIED = len(UNIFIED_CLASSES)

_DACL1K_TO_UNIFIED: dict[str, int] = {
    "NoDamage":      6,
    "Crack":         0,
    "Efflorescence": 3,
    "Spalling":      2,
    "BarsExposed":   5,
    "Rust":          4,
}

_RDD_TO_UNIFIED: dict[str, int] = {
    "D00": 0,  # LongitudinalCrack  → Crack
    "D10": 0,  # TransverseCrack    → Crack
    "D20": 0,  # AlligatorCrack     → Crack
    "D40": 1,  # Pothole            → Pothole
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transform(split: str) -> A.Compose:
    """Return the albumentations pipeline for *split* ('train' | 'val' | 'test')."""
    if split == "train":
        return A.Compose([
            A.Resize(384, 384),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


class BridgeDataset(Dataset):
    """
    Wraps bikit's dacl1k dataset (bridge images).

    Each sample returns:
        {
            "image":        Tensor[3, 384, 384],
            "asset_type":   Tensor(1)  — bridge
            "damage_types": Tensor[7]  — binary multi-label (unified vocab)
        }

    Args:
        split:      'train', 'val', or 'test'
        transform:  albumentations Compose pipeline; defaults to get_transform(split)
        data_root:  root directory passed to BikitDataset as cache_dir
                    (dacl1k images live at data_root/dacl1k/…)
    """

    def __init__(
        self,
        split: str,
        transform: Optional[A.Compose] = None,
        data_root: str = "data/raw",
    ) -> None:
        self.transform = transform if transform is not None else get_transform(split)

        self._inner = _BikitDataset(
            name="dacl1k",
            split=split,
            cache_dir=data_root,
            img_type="pil",
            return_type="np",
        )
        self._class_names: List[str] = self._inner.class_names

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        img_np, label = self._inner[idx]  # HxWxC uint8, float32 [6]

        img_tensor: torch.Tensor = self.transform(image=img_np)["image"]  # [3, H, W]

        damage = torch.zeros(N_UNIFIED, dtype=torch.float32)
        for i, cls_name in enumerate(self._class_names):
            if label[i] == 1.0 and cls_name in _DACL1K_TO_UNIFIED:
                damage[_DACL1K_TO_UNIFIED[cls_name]] = 1.0

        return {
            "image":        img_tensor,
            "asset_type":   torch.tensor(1, dtype=torch.long),
            "damage_types": damage,
        }


# ---------------------------------------------------------------------------
# RoadDataset
# ---------------------------------------------------------------------------

class RoadDataset(Dataset):
    """
    Reads the RDD2022 dataset from a PascalVOC directory layout:

        data_root/rdd2022/{split}/images/*.jpg
        data_root/rdd2022/{split}/annotations/*.xml

    Each sample returns:
        {
            "image":        Tensor[3, 384, 384],
            "asset_type":   Tensor(0)  — road
            "damage_types": Tensor[7]  — binary multi-label (unified vocab)
        }

    Damage mapping:
        D00, D10, D20 → Crack (index 0)
        D40           → Pothole (index 1)
        (no boxes)    → NoDefect (index 6)

    Args:
        split:      'train' or 'val'
        transform:  albumentations Compose pipeline; defaults to get_transform(split)
        data_root:  root directory; RDD2022 lives at data_root/rdd2022/
    """

    def __init__(
        self,
        split: str,
        transform: Optional[A.Compose] = None,
        data_root: str = "data/raw",
    ) -> None:
        self.transform = transform if transform is not None else get_transform(split)

        if split not in ("train", "val"):
            raise ValueError(f"RoadDataset split must be 'train' or 'val', got '{split}'")
        rdd_split = split
        self._images_dir = Path(data_root) / "rdd2022" / "RDD_SPLIT" / rdd_split / "images"
        self._annots_dir = Path(data_root) / "rdd2022" / "RDD_SPLIT" / rdd_split / "annotations"

        self._samples: List[Path] = sorted(self._images_dir.glob("*.jpg"))
        if not self._samples:
            raise FileNotFoundError(
                f"No .jpg images found in {self._images_dir}.\n"
                "Run `uv run main.py download` to download RDD2022 first."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _parse_annotation(self, xml_path: Path) -> np.ndarray:
        """Parse a PascalVOC XML file → unified 7-class float32 binary vector."""
        damage = np.zeros(N_UNIFIED, dtype=np.float32)

        if not xml_path.exists():
            damage[6] = 1.0  # NoDefect
            return damage

        root = ET.parse(xml_path).getroot()
        objects = root.findall("object")

        if not objects:
            damage[6] = 1.0  # NoDefect
            return damage

        for obj in objects:
            name_el = obj.find("name")
            if name_el is not None:
                code = name_el.text.strip()
                if code in _RDD_TO_UNIFIED:
                    damage[_RDD_TO_UNIFIED[code]] = 1.0

        return damage

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img_path = self._samples[idx]
        xml_path = self._annots_dir / (img_path.stem + ".xml")

        img_np = np.array(Image.open(img_path).convert("RGB"))  # HxWxC uint8
        img_tensor: torch.Tensor = self.transform(image=img_np)["image"]  # [3, H, W]

        damage = torch.from_numpy(self._parse_annotation(xml_path))

        return {
            "image":        img_tensor,
            "asset_type":   torch.tensor(0, dtype=torch.long),
            "damage_types": damage,
        }


class CombinedDamageDataset:
    """
    Concatenates BridgeDataset and RoadDataset into a single dataset.

    Uses WeightedRandomSampler to balance the 32:1 road/bridge imbalance so
    that each class contributes ~50 % of samples per epoch.

    Usage:
        dataset = CombinedDamageDataset(split="train")
        loader  = DataLoader(dataset, batch_size=32, sampler=dataset.get_sampler())

    Args:
        split:      'train' or 'val'
        data_root:  root directory for raw data
        transform:  shared albumentations pipeline; defaults to get_transform(split)
    """

    def __init__(
        self,
        split: str,
        data_root: str = "data/raw",
        transform: Optional[A.Compose] = None,
    ) -> None:
        tfm = transform if transform is not None else get_transform(split)

        self.bridge_ds = BridgeDataset(split=split, transform=tfm, data_root=data_root)
        self.road_ds   = RoadDataset(split=split,   transform=tfm, data_root=data_root)

        # road samples first, bridge samples second (matches _weights order)
        self._dataset = ConcatDataset([self.road_ds, self.bridge_ds])

        n_road   = len(self.road_ds)
        n_bridge = len(self.bridge_ds)

        # upweight bridge so total bridge weight ≈ total road weight
        bridge_w = n_road / n_bridge
        self._weights = torch.tensor(
            [1.0] * n_road + [bridge_w] * n_bridge,
            dtype=torch.double,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        return self._dataset[idx]

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler giving ~50 % road / 50 % bridge per batch."""
        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )
