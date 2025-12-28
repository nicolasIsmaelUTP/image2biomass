from pathlib import Path
from typing import Sequence, Tuple

from torch.utils.data import DataLoader

from .dataset import BiomassDataset
from .transforms import get_transforms


def make_datasets(train_df, val_df, targets: Sequence[str], images_root: Path, img_size: int):
    train_tfms, val_tfms = get_transforms(img_size)
    train_ds = BiomassDataset(train_df, images_root, targets, train_tfms)
    val_ds = BiomassDataset(val_df, images_root, targets, val_tfms)
    return train_ds, val_ds, train_tfms, val_tfms


def make_dataloaders(train_df, val_df, targets: Sequence[str], images_root: Path, img_size: int,
                     batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, object]:
    train_ds, val_ds, train_tfms, val_tfms = make_datasets(train_df, val_df, targets, images_root, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, val_tfms
