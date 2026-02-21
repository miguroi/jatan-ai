from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

_BRIDGE_CLASSES = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket",
    "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering",
    "Restformwork", "ExposedRebars",
    "Bearing", "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
]
_N_BRIDGE = len(_BRIDGE_CLASSES)

_ROAD_CLASSES = ["D00", "D10", "D20", "D40", "NoDefect"]
_N_ROAD = len(_ROAD_CLASSES)

EIDSEG_CLASSES = [
    "Undamaged Building",
    "Damaged Building",
    "Destroyed Building",
    "Undamaged Road",
    "Damaged Road",
    "Background",
]

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
    """dacl10k bridge segmentation dataset (19 classes, pixel-level masks)."""

    def __init__(self, split: str, data_root: str = "data/dacl10k") -> None:
        import pickle
        from dacl10k.dacl10kdataset import Dacl10kDataset

        dacl_split = {"val": "validation"}.get(split, split)
        self._inner = Dacl10kDataset(
            split=dacl_split,
            data_path=data_root,
            resize_img=(512, 512),
            resize_mask=(512, 512),
            normalize_img=True,
        )

        cache_file = os.path.join(data_root, f"prefetched_{dacl_split}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading prefetched cache from {cache_file}")
            with open(cache_file, "rb") as f:
                self._inner.prefetched_data = pickle.load(f)
            self._inner.use_prefetched_data = True
        else:
            print(f"Prefetching {dacl_split} split (one-time, ~minutes)...")
            self._inner.run_prefetching(n_jobs=8)
            with open(cache_file, "wb") as f:
                pickle.dump(self._inner.prefetched_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved to {cache_file}")

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        img, mask = self._inner[idx]   # img: (3,512,512), mask: (19,512,512) float
        return {
            "image": img,
            "domain": torch.tensor(1),
            "mask": mask.float(),
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
        self._annots_dir = Path(data_root) / "rdd2022" / "RDD_SPLIT" / split / "labels"

        self._samples: List[Path] = sorted(self._images_dir.glob("*.jpg"))
        if not self._samples:
            raise FileNotFoundError(
                f"No .jpg images found in {self._images_dir}.\n"
                "Run `uv run main.py download` to download RDD2022 first."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _parse_annotation(self, txt_path: Path) -> np.ndarray:
        damage = np.zeros(_N_ROAD, dtype=np.float32)

        if not txt_path.exists():
            damage[_ROAD_CLASSES.index("NoDefect")] = 1.0
            return damage

        with open(txt_path) as f:
            lines = f.read().strip().splitlines()

        if not lines:
            damage[_ROAD_CLASSES.index("NoDefect")] = 1.0
            return damage

        yolo_to_road = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in yolo_to_road:
                road_class = yolo_to_road[class_id]
                damage[_ROAD_CLASSES.index(road_class)] = 1.0

        if damage.sum() == 0:
            damage[_ROAD_CLASSES.index("NoDefect")] = 1.0

        return damage

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img_path = self._samples[idx]
        txt_path = self._annots_dir / (img_path.stem + ".txt")

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_tensor = self.transform(image=img_np)["image"]
        damage = torch.from_numpy(self._parse_annotation(txt_path))

        return {
            "image": img_tensor,
            "domain": torch.tensor(0, dtype=torch.long),
            "damage": damage,
        }


class CombinedDamageDataset:
    """Road-only dataset for the MTL road classifier."""

    def __init__(
        self,
        split: str,
        data_root: str = "data/raw",
        transform: Optional[A.Compose] = None,
    ) -> None:
        tfm = transform if transform is not None else get_transform(split)
        self.road_ds = RoadDataset(split=split, transform=tfm, data_root=data_root)
        self._dataset = self.road_ds
        self._weights = torch.ones(len(self.road_ds), dtype=torch.double)

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
