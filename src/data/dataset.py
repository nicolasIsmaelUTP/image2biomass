"""
Datasets de PyTorch para imágenes de biomasa.

BiomassDataset: para entrenamiento/validación (imagen + targets).
BiomassTestDataset: para test (solo imagen + ruta).
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BiomassDataset(Dataset):
    """
    Dataset para entrenamiento y validación.

    Cada muestra retorna:
        - image: tensor de imagen ya transformada
        - target: tensor con los valores de los targets (ya preprocesados)
    """

    def __init__(self, df, images_root, targets: List[str], transform):
        """
        Args:
            df: DataFrame con columnas 'image_path' y los targets.
            images_root: Ruta base donde están las imágenes.
            targets: Lista de nombres de columnas de targets.
            transform: Transformaciones de torchvision a aplicar.
        """
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.targets = list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Cargar imagen RGB
        img_path = self.images_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Targets como tensor float32
        target = torch.tensor(
            row[self.targets].values.astype(np.float32)
        )

        return image, target


class BiomassTestDataset(Dataset):
    """
    Dataset para test (sin targets).

    Cada muestra retorna:
        - image: tensor de imagen ya transformada
        - image_path: ruta de la imagen (para armar submission)
    """

    def __init__(self, df, images_root, transform):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, row["image_path"]
