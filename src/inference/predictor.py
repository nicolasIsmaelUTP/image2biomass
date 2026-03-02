"""
Predicción sobre test set y generación de submission.

Pipeline de inferencia:
    1. Modelo predice en espacio estandarizado-log.
    2. Inverse transform con el scaler del fold → escala original.
    3. Clip a 0 (biomasa no puede ser negativa).
    4. Derivar GDM y Total.
    5. Formatear como submission (sample_id, target).
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BiomassTestDataset


def predict_test(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    targets: List[str],
    scaler,
    images_root: Path,
    transform,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
) -> pd.DataFrame:
    """
    Genera predicciones sobre el test set en formato de submission.

    Args:
        model: Modelo entrenado (en modo eval).
        test_df: DataFrame del test CSV.
        targets: Lista de targets primarios.
        scaler: TargetScaler del fold (para invertir transformación).
        images_root: Ruta base de las imágenes.
        transform: Transforms de validación.
        device: Dispositivo (CPU/GPU).
        batch_size: Tamaño de batch para predicción.
        num_workers: Número de workers del DataLoader.

    Returns:
        DataFrame con columnas [sample_id, target] listo para submission.
    """
    # Obtener imágenes únicas del test
    test_images = (
        test_df.drop_duplicates("image_path")
        .reset_index(drop=True)
    )

    # Crear dataloader
    test_dataset = BiomassTestDataset(test_images, images_root, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Predecir
    model.eval()
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            all_preds.append(preds)
            all_paths.extend(paths)

    preds_scaled = np.concatenate(all_preds, axis=0)

    # Invertir transformación: des-estandarizar + expm1 → escala original
    preds_original = scaler.inverse_transform(preds_scaled)

    # Clip a 0 (no existe biomasa negativa)
    preds_original = np.clip(preds_original, 0, None)

    # Crear DataFrame ancho con los 3 primarios
    wide_df = pd.DataFrame(preds_original, columns=targets)
    wide_df["image_path"] = all_paths

    # Derivar GDM y Total
    wide_df["GDM_g"] = wide_df["Dry_Green_g"] + wide_df["Dry_Clover_g"]
    wide_df["Dry_Total_g"] = (
        wide_df["Dry_Green_g"]
        + wide_df["Dry_Clover_g"]
        + wide_df["Dry_Dead_g"]
    )

    # Convertir a formato largo (submission)
    all_targets = targets + ["GDM_g", "Dry_Total_g"]
    long_df = wide_df.melt(
        id_vars="image_path",
        value_vars=all_targets,
        var_name="target_name",
        value_name="target",
    )

    # Crear sample_id con el formato: ID_imagen__nombre_target
    long_df["sample_id"] = long_df.apply(
        lambda row: f"{Path(row['image_path']).stem}__{row['target_name']}",
        axis=1,
    )

    # Retornar solo las columnas del submission
    submission = (
        long_df[["sample_id", "target"]]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    return submission
