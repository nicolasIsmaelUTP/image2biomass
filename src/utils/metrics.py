from typing import Mapping, Sequence

import numpy as np

from .logging_config import get_logger

DEFAULT_TARGET_WEIGHTS: Mapping[str, float] = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

logger = get_logger(__name__)


def _validate_targets(target_names_arr: np.ndarray, target_weights: Mapping[str, float]) -> None:
    """Ensure all expected targets are present and report anomalies."""

    unique_targets, counts = np.unique(target_names_arr, return_counts=True)
    unique_set = set(unique_targets.tolist())
    expected_set = set(target_weights.keys())

    missing_targets = expected_set - unique_set
    unexpected_targets = unique_set - expected_set

    if missing_targets:
        logger.error("Missing targets for weighted R2: %s", sorted(missing_targets))
        raise ValueError(f"Missing targets: {sorted(missing_targets)}")

    if unexpected_targets:
        logger.warning("Unexpected targets present for weighted R2: %s", sorted(unexpected_targets))

    logger.info(
        "Computing weighted R2 | samples=%d | unique_targets=%s | counts=%s",
        target_names_arr.shape[0],
        sorted(unique_set),
        {t: int(c) for t, c in zip(unique_targets, counts)},
    )


def weighted_r2_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    target_names: Sequence[str],
    target_weights: Mapping[str, float] = DEFAULT_TARGET_WEIGHTS,
) -> float:
    """
    Compute globally weighted R^2 pooling all targets with per-target weights.
    
    Automatically computes derived targets:
    - Dry_Total_g = Dry_Clover_g + Dry_Green_g + Dry_Dead_g
    - GDM_g = Dry_Green_g + Dry_Clover_g

    Args:
        y_true: Ground truth values (1D sequence).
        y_pred: Predicted values (same shape as ``y_true``).
        target_names: Target name for each element in ``y_true``/``y_pred``.
        target_weights: Mapping from target name to weight.

    Returns:
        Weighted R^2 in the range (-inf, 1]. Returns ``nan`` if variance is zero.

    Raises:
        ValueError: If shapes mismatch, targets are missing, or an unknown target name is encountered.
    """

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    target_names_arr = np.asarray(target_names)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.shape[0] != target_names_arr.shape[0]:
        raise ValueError("target_names length must match y_true/y_pred")

    # Compute derived targets if base targets are present
    y_true_expanded = []
    y_pred_expanded = []
    target_names_expanded = []
    
    # Group values by sample (assuming targets are in repeating blocks)
    unique_targets = np.unique(target_names_arr)
    n_targets = len(unique_targets)
    n_samples = len(y_true_arr) // n_targets
    
    for i in range(n_samples):
        start_idx = i * n_targets
        end_idx = start_idx + n_targets
        
        sample_true = y_true_arr[start_idx:end_idx]
        sample_pred = y_pred_arr[start_idx:end_idx]
        sample_names = target_names_arr[start_idx:end_idx]
        
        # Add original values
        y_true_expanded.extend(sample_true)
        y_pred_expanded.extend(sample_pred)
        target_names_expanded.extend(sample_names)
        
        # Create lookup for this sample
        name_to_true = {name: val for name, val in zip(sample_names, sample_true)}
        name_to_pred = {name: val for name, val in zip(sample_names, sample_pred)}
        
        # Compute Dry_Total_g if base components exist
        if all(k in name_to_true for k in ['Dry_Clover_g', 'Dry_Green_g', 'Dry_Dead_g']):
            total_true = name_to_true['Dry_Clover_g'] + name_to_true['Dry_Green_g'] + name_to_true['Dry_Dead_g']
            total_pred = name_to_pred['Dry_Clover_g'] + name_to_pred['Dry_Green_g'] + name_to_pred['Dry_Dead_g']
            y_true_expanded.append(total_true)
            y_pred_expanded.append(total_pred)
            target_names_expanded.append('Dry_Total_g')
        
        # Compute GDM_g if components exist
        if all(k in name_to_true for k in ['Dry_Green_g', 'Dry_Clover_g']):
            gdm_true = name_to_true['Dry_Green_g'] + name_to_true['Dry_Clover_g']
            gdm_pred = name_to_pred['Dry_Green_g'] + name_to_pred['Dry_Clover_g']
            y_true_expanded.append(gdm_true)
            y_pred_expanded.append(gdm_pred)
            target_names_expanded.append('GDM_g')
    
    y_true_arr = np.array(y_true_expanded, dtype=np.float64)
    y_pred_arr = np.array(y_pred_expanded, dtype=np.float64)
    target_names_arr = np.array(target_names_expanded)

    _validate_targets(target_names_arr, target_weights)

    try:
        weights = np.array([target_weights[name] for name in target_names_arr], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing weight for target '{exc.args[0]}'") from exc

    weighted_mean = np.average(y_true_arr, weights=weights)
    ss_res = np.sum(weights * np.square(y_true_arr - y_pred_arr))
    ss_tot = np.sum(weights * np.square(y_true_arr - weighted_mean))

    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def weighted_r2_from_frame(
    df,
    target_col: str,
    pred_col: str,
    target_name_col: str = "target_name",
    target_weights: Mapping[str, float] = DEFAULT_TARGET_WEIGHTS,
) -> float:
    """Convenience wrapper to compute weighted R^2 from a long-format DataFrame."""

    return weighted_r2_score(
        y_true=df[target_col].to_numpy(),
        y_pred=df[pred_col].to_numpy(),
        target_names=df[target_name_col].to_numpy(),
        target_weights=target_weights,
    )
