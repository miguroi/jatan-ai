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

_BRIDGE_CLASSES = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]
_N_BRIDGE = len(_BRIDGE_CLASSES)

_ROAD_CLASSES = ["D00", "D10", "D20", "D40", "NoDefect"]
_N_ROAD = len(_ROAD_CLASSES)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(split: str) -> A.Compose:
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
        img_np, label = self._inner[idx]
        img_tensor = self.transform(image=img_np)["image"]

        damage = torch.zeros(_N_BRIDGE, dtype=torch.float32)
        for i, cls_name in enumerate(self._class_names):
            if cls_name in _BRIDGE_CLASSES:
                damage[_BRIDGE_CLASSES.index(cls_name)] = float(label[i])

        return {
            "image": img_tensor,
            "domain": "bridge",
            "damage": damage,
        }


class RoadDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform: Optional[A.Compose] = None,
        data_root: str = "data/raw",
    ) -> None:
        self.transform = transform if transform is not None else get_transform(split)

        if split not in ("train", "val"):
            raise ValueError(f"RoadDataset split must be 'train' or 'val', got '{split}'")

        self._images_dir = Path(data_root) / "rdd2022" / "RDD_SPLIT" / split / "images"
        self._annots_dir = Path(data_root) / "rdd2022" / "RDD_SPLIT" / split / "annotations"

        self._samples: List[Path] = sorted(self._images_dir.glob("*.jpg"))
        if not self._samples:
            raise FileNotFoundError(
                f"No .jpg images found in {self._images_dir}.\n"
                "Run `uv run main.py download` to download RDD2022 first."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _parse_annotation(self, xml_path: Path) -> np.ndarray:
        damage = np.zeros(_N_ROAD, dtype=np.float32)

        if not xml_path.exists():
            damage[_ROAD_CLASSES.index("NoDefect")] = 1.0
            return damage

        root = ET.parse(xml_path).getroot()
        objects = root.findall("object")

        if not objects:
            damage[_ROAD_CLASSES.index("NoDefect")] = 1.0
            return damage

        for obj in objects:
            name_el = obj.find("name")
            if name_el is not None:
                code = name_el.text.strip()
                if code in _ROAD_CLASSES:
                    damage[_ROAD_CLASSES.index(code)] = 1.0

        return damage

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img_path = self._samples[idx]
        xml_path = self._annots_dir / (img_path.stem + ".xml")

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_tensor = self.transform(image=img_np)["image"]
        damage = torch.from_numpy(self._parse_annotation(xml_path))

        return {
            "image": img_tensor,
            "domain": "road",
            "damage": damage,
        }


class CombinedDamageDataset:
    def __init__(
        self,
        split: str,
        data_root: str = "data/raw",
        transform: Optional[A.Compose] = None,
    ) -> None:
        tfm = transform if transform is not None else get_transform(split)

        self.road_ds = RoadDataset(split=split, transform=tfm, data_root=data_root)
        self.bridge_ds = BridgeDataset(split=split, transform=tfm, data_root=data_root)

        self._dataset = ConcatDataset([self.road_ds, self.bridge_ds])

        n_road = len(self.road_ds)
        n_bridge = len(self.bridge_ds)

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
        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )
