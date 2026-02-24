from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

# EIDSeg 3-class bridge damage (replaces dacl10k 19-class)
_BRIDGE_CLASSES = ["Undamaged", "Damaged", "Destroyed"]
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


class EIDSegDataset(Dataset):
    """EIDSeg earthquake infrastructure damage segmentation dataset.

    Remaps 6 EIDSeg polygon labels → 3 bridge-relevant classes:
      0: Undamaged  (UD_Building, UD_Road)
      1: Damaged    (D_Building, D_Road)
      2: Destroyed  (Debris)
    255: Ignore     (Undesignated + unannotated background)

    Annotations are CVAT XML polygons rasterised on-the-fly.
    """

    _LABEL_MAP: dict[str, int] = {
        "UD_Building":  0,
        "UD_Road":      0,
        "D_Building":   1,
        "D_Road":       1,
        "Debris":       2,
        "Undesignated": 255,
    }

    def __init__(
        self,
        split: str,
        data_root: str = "data/raw/eidseg",
        transform: Optional[A.Compose] = None,
    ) -> None:
        import xml.etree.ElementTree as ET

        self.transform = transform if transform is not None else self._default_transform(split)

        split_dir  = Path(data_root) / "data" / split
        xml_file   = split_dir / f"{split}.xml"
        images_dir = split_dir / "images"
        if not any(images_dir.glob("*.*")):
            images_dir = images_dir / "default"
        self._images_dir = images_dir

        tree = ET.parse(xml_file)
        root = tree.getroot()

        self._samples: list[dict] = []
        for img_elem in root.findall("image"):
            name   = img_elem.get("name", "")
            width  = int(img_elem.get("width",  512))
            height = int(img_elem.get("height", 512))

            img_path = images_dir / name
            if not img_path.exists():
                continue

            polygons: list[dict] = []
            for poly in img_elem.findall("polygon"):
                label = poly.get("label", "")
                if label not in self._LABEL_MAP:
                    continue
                pts_str = poly.get("points", "")
                z_order = int(poly.get("z_order", 0))
                pts = [
                    tuple(float(v) for v in pt.split(","))
                    for pt in pts_str.split(";") if pt.strip()
                ]
                if len(pts) >= 3:
                    polygons.append({"label": label, "points": pts, "z_order": z_order})

            polygons.sort(key=lambda p: p["z_order"])
            self._samples.append({
                "image_path": img_path,
                "width": width,
                "height": height,
                "polygons": polygons,
            })

        if not self._samples:
            raise FileNotFoundError(
                f"No valid samples found in {xml_file}. "
                f"Check that images exist in {images_dir}."
            )

    @staticmethod
    def _default_transform(split: str) -> A.Compose:
        if split == "train":
            return A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ])

    def _render_mask(self, sample: dict) -> np.ndarray:
        from PIL import Image as PILImage, ImageDraw

        w, h = sample["width"], sample["height"]
        mask = PILImage.new("L", (w, h), 255)  # start with ignore
        draw = ImageDraw.Draw(mask)
        for poly in sample["polygons"]:
            cls_idx = self._LABEL_MAP[poly["label"]]
            pts = [(int(x), int(y)) for x, y in poly["points"]]
            draw.polygon(pts, fill=cls_idx)
        return np.array(mask, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image as PILImage

        sample  = self._samples[idx]
        img_np  = np.array(PILImage.open(sample["image_path"]).convert("RGB"))
        mask_np = self._render_mask(sample)

        aug         = self.transform(image=img_np, mask=mask_np)
        img_tensor  = aug["image"]
        mask_tensor = torch.from_numpy(aug["mask"].copy()).long()

        return {
            "image":  img_tensor,
            "domain": torch.tensor(1),
            "mask":   mask_tensor,   # [H, W], values: 0,1,2,255
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
