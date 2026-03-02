from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BiomassTestDataset


class Predictor:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict(self, test_df, targets: Sequence[str], images_root: Path, transform, batch_size: int = 16,
                num_workers: int = 0) -> pd.DataFrame:
        test_meta = (
            test_df[test_df["target_name"].isin(targets)]
            .drop_duplicates("image_path")
            .reset_index(drop=True)
        )
        test_loader = DataLoader(
            BiomassTestDataset(test_meta, images_root, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.model.eval()
        rows = []
        with torch.no_grad():
            for images, paths in test_loader:
                images = images.to(self.device)
                preds = self.model(images).cpu().numpy()
                for path, pred in zip(paths, preds):
                    rows.append({"image_path": path, **{t: p for t, p in zip(targets, pred)}})

        wide_df = pd.DataFrame(rows).sort_values("image_path").reset_index(drop=True)
        long_df = wide_df.melt(id_vars="image_path", value_vars=targets, var_name="target_name", value_name="target")
        long_df["sample_id"] = long_df.apply(lambda r: f"{Path(r['image_path']).stem}__{r['target_name']}", axis=1)
        return long_df[["sample_id", "target"]]
