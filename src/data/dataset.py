from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BiomassDataset(Dataset):
    def __init__(self, df, images_root: Path, targets: Sequence[str], transform):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.targets = list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(row[self.targets].values.astype(np.float32))
        return image, target


class BiomassTestDataset(Dataset):
    def __init__(self, df, images_root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), row["image_path"]
