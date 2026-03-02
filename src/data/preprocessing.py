"""
Preprocesamiento de targets y creación de folds.

Pipeline de targets (ver docs/metodologia_final_justificada.md):
1. log(1 + y)          → reduce right-skewness y estabiliza varianza
2. z-score por fold    → iguala escalas entre targets (Green ~27g vs Clover ~7g)

TargetScaler: aplica y revierte estas transformaciones.
prepare_pivot_table: convierte CSV largo a formato ancho (1 fila por imagen).
create_folds: 5-Fold GroupKFold por Sampling_Date (sin leakage temporal).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class TargetScaler:
    """
    Aplica log(1+y) y estandarización z-score.

    Se ajusta (fit) SOLO con datos de entrenamiento de cada fold.
    Luego transforma tanto train como val con esos mismos parámetros.

    Uso:
        scaler = TargetScaler()
        scaler.fit(y_train)           # Calcular media y std en train
        y_train_scaled = scaler.transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Para evaluar: volver a escala original
        y_pred_original = scaler.inverse_transform(y_pred_scaled)
    """

    def __init__(self):
        self.mean_ = None   # Media de log(1+y) en train
        self.std_ = None    # Std de log(1+y) en train

    def fit(self, y_train: np.ndarray) -> "TargetScaler":
        """
        Ajustar con targets de entrenamiento (escala original).

        Args:
            y_train: array de shape (n_samples, n_targets) en escala original.
        """
        y_log = np.log1p(y_train)
        self.mean_ = y_log.mean(axis=0)
        self.std_ = y_log.std(axis=0)

        # Protección contra división por cero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transformar: log(1+y) → estandarizar con media/std de train."""
        y_log = np.log1p(y)
        return (y_log - self.mean_) / self.std_

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invertir: des-estandarizar → expm1 → escala original."""
        y_log = y_scaled * self.std_ + self.mean_
        return np.expm1(y_log)


def prepare_pivot_table(
    train_df: pd.DataFrame, targets: List[str]
) -> pd.DataFrame:
    """
    Convierte el CSV largo (1 fila por target) a formato ancho (1 fila por imagen).

    También incluye metadata necesaria para GroupKFold (Sampling_Date, State).

    Args:
        train_df: DataFrame del CSV de entrenamiento.
        targets: Lista de targets primarios a incluir.

    Returns:
        DataFrame con columnas: image_path, Sampling_Date, State, + targets.
    """
    # Pivot: formato largo → ancho
    pivot = (
        train_df
        .pivot_table(index="image_path", columns="target_name", values="target")
        .reset_index()
    )

    # Añadir metadata (primera fila de cada imagen)
    metadata = (
        train_df
        .drop_duplicates("image_path")
        [["image_path", "Sampling_Date", "State"]]
    )
    pivot = pivot.merge(metadata, on="image_path", how="left")

    # Seleccionar columnas necesarias
    columns = ["image_path", "Sampling_Date", "State"] + targets
    pivot = pivot[columns].dropna().reset_index(drop=True)

    return pivot


def create_folds(
    pivot: pd.DataFrame,
    n_folds: int = 5,
    group_col: str = "Sampling_Date",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crea GroupKFold splits agrupando por Sampling_Date.

    Esto garantiza que no hay leakage temporal: ninguna fecha
    aparece tanto en train como en validación del mismo fold.

    Args:
        pivot: DataFrame en formato ancho (de prepare_pivot_table).
        n_folds: Número de folds (default: 5).
        group_col: Columna para agrupar (default: Sampling_Date).

    Returns:
        Lista de tuplas (train_indices, val_indices) para cada fold.
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = pivot[group_col].values

    folds = []
    for train_idx, val_idx in gkf.split(pivot, groups=groups):
        folds.append((train_idx, val_idx))

    return folds
