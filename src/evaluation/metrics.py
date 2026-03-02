"""
Métricas de evaluación para el experimento de biomasa.

Métrica primaria: Weighted R² (métrica oficial de la competición).
Métricas secundarias: MAE, RMSE, R² por componente.

Protocolo de evaluación (ver docs/metodologia_final_justificada.md, sección 10):
    1. Predecir 3 targets primarios en escala original.
    2. Derivar GDM = Green + Clover, Total = Green + Clover + Dead.
    3. Aplicar log(1+y) a predicciones y verdad.
    4. Calcular R² por target, ponderar con pesos oficiales.

Pesos oficiales:
    Dry_Green_g:  0.1   (primario)
    Dry_Dead_g:   0.1   (primario)
    Dry_Clover_g: 0.1   (primario)
    GDM_g:        0.2   (derivado = Green + Clover)
    Dry_Total_g:  0.5   (derivado = Green + Clover + Dead)

Los derivados acumulan 70% del score → mejorar los 3 primarios
tiene impacto multiplicado en el score final.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Pesos oficiales de la competición
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

# Targets primarios (los que el modelo predice directamente)
PRIMARY_TARGETS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]


def compute_derived_targets(
    values: np.ndarray, targets: List[str]
) -> Dict[str, np.ndarray]:
    """
    Calcula targets derivados a partir de los 3 primarios.

    GDM_g = Dry_Green_g + Dry_Clover_g
    Dry_Total_g = Dry_Green_g + Dry_Clover_g + Dry_Dead_g

    Args:
        values: Array de shape (n_samples, 3) con los 3 targets primarios.
        targets: Lista con nombres de los 3 targets (en el mismo orden que values).

    Returns:
        Diccionario con los 5 targets (3 primarios + 2 derivados).
    """
    # Crear diccionario nombre → vector de valores
    target_dict = {}
    for i, name in enumerate(targets):
        target_dict[name] = values[:, i]

    # Calcular derivados (coherencia física garantizada)
    target_dict["GDM_g"] = (
        target_dict["Dry_Green_g"] + target_dict["Dry_Clover_g"]
    )
    target_dict["Dry_Total_g"] = (
        target_dict["Dry_Green_g"]
        + target_dict["Dry_Clover_g"]
        + target_dict["Dry_Dead_g"]
    )

    return target_dict


def weighted_r2_score(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> float:
    """
    Calcula el Weighted R² oficial de la competición.

    Aplica log(1+y) antes de calcular R² por target,
    luego pondera según los pesos oficiales.

    Final Score = Σ w_i × R²_i

    Args:
        y_true_dict: Dict con ground truth por target (escala original).
        y_pred_dict: Dict con predicciones por target (escala original).

    Returns:
        Weighted R² (float). Rango teórico: (-inf, 1.0].
    """
    score = 0.0

    for target_name, weight in TARGET_WEIGHTS.items():
        # Aplicar log(1+y) como en la competición
        y_true_log = np.log1p(y_true_dict[target_name])
        y_pred_log = np.log1p(y_pred_dict[target_name])

        # Calcular R² para este target
        if np.std(y_true_log) == 0:
            r2 = 0.0  # Evitar R² indefinido
        else:
            r2 = r2_score(y_true_log, y_pred_log)

        score += weight * r2

    return score


def compute_per_target_metrics(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Calcula MAE, RMSE y R² por cada uno de los 5 targets.

    Las métricas se calculan en escala original (gramos).

    Args:
        y_true_dict: Dict con ground truth por target.
        y_pred_dict: Dict con predicciones por target.

    Returns:
        Dict anidado: {target_name: {"mae": ..., "rmse": ..., "r2": ...}}
    """
    metrics = {}

    for target_name in TARGET_WEIGHTS.keys():
        y_true = y_true_dict[target_name]
        y_pred = y_pred_dict[target_name]

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        if np.std(y_true) == 0:
            r2 = 0.0
        else:
            r2 = float(r2_score(y_true, y_pred))

        metrics[target_name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    return metrics


def evaluate_fold(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    targets: List[str],
) -> Dict:
    """
    Evaluación completa de un fold.

    Recibe predicciones y verdad en escala ORIGINAL (ya invertidas por el scaler).
    Calcula targets derivados, luego weighted R² y métricas por target.

    Args:
        y_true_raw: Ground truth, shape (n_samples, 3), escala original.
        y_pred_raw: Predicciones, shape (n_samples, 3), escala original.
        targets: Lista de nombres de targets primarios.

    Returns:
        Dict con "weighted_r2" y "per_target" métricas.
    """
    # Derivar GDM y Total desde los 3 primarios
    y_true_dict = compute_derived_targets(y_true_raw, targets)
    y_pred_dict = compute_derived_targets(y_pred_raw, targets)

    # Clip predicciones negativas a 0 (biomasa no puede ser negativa)
    for key in y_pred_dict:
        y_pred_dict[key] = np.clip(y_pred_dict[key], 0, None)

    # Métrica primaria: Weighted R²
    w_r2 = weighted_r2_score(y_true_dict, y_pred_dict)

    # Métricas secundarias por target
    per_target = compute_per_target_metrics(y_true_dict, y_pred_dict)

    return {
        "weighted_r2": w_r2,
        "per_target": per_target,
    }
