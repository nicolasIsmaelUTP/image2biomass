"""
Generación de gráficas de evaluación por fold.

Produce dos figuras:
    1. Curva de pérdida (train vs val) a lo largo de las épocas.
    2. Dispersión predicción vs verdad para cada target primario.

Las figuras se guardan como PNG para subir como artifacts a MLflow.
"""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def plot_loss_curve(
    history: List[Dict],
    save_path: Path,
    title: str = "Evolución de la pérdida",
) -> Path:
    """
    Genera gráfica de train loss vs val loss por época.

    Args:
        history: Lista de dicts con "epoch", "train_loss", "val_loss".
        save_path: Ruta completa del archivo PNG.
        title: Título de la gráfica.

    Returns:
        Ruta del archivo guardado.
    """
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, "o-", label="Train loss", markersize=3)
    ax.plot(epochs, val_losses, "o-", label="Val loss", markersize=3)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_scatter_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: List[str],
    save_path: Path,
    title: str = "Dispersión predicciones vs verdad en validación",
) -> Path:
    """
    Genera scatter plots de predicción vs ground truth para cada target.

    Muestra la línea ideal (y=x) y el R² en el subtítulo.

    Args:
        y_true: Ground truth, shape (n_samples, n_targets), escala original.
        y_pred: Predicciones, shape (n_samples, n_targets), escala original.
        targets: Lista de nombres de targets (mismo orden que columnas).
        save_path: Ruta completa del archivo PNG.
        title: Título general de la figura.

    Returns:
        Ruta del archivo guardado.
    """
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, target_name) in enumerate(zip(axes, targets)):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # Calcular R²
        if np.std(yt) == 0:
            r2 = 0.0
        else:
            r2 = r2_score(yt, yp)

        # Scatter
        ax.scatter(yt, yp, alpha=0.6, s=30, label="Pred vs GT")

        # Línea ideal
        all_vals = np.concatenate([yt, yp])
        lim_min = min(0, all_vals.min())
        lim_max = all_vals.max() * 1.05
        ax.plot(
            [lim_min, lim_max], [lim_min, lim_max],
            "r-", linewidth=2, label="Ideal",
        )

        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Predicción")
        ax.set_title(f"{target_name}\nR²={r2:.3f}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path
