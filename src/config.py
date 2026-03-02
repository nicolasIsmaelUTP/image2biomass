"""
Configuración central del experimento.

Contiene todos los hiperparámetros y opciones del experimento.
Si debug=True, se reducen epochs y paciencia para verificar rápidamente
que el pipeline y MLflow funcionan correctamente.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    """
    Configuración del experimento de comparación de arquitecturas.

    Atributos principales:
        debug: Si True, usa pocos epochs para probar el pipeline.
        seed: Semilla para reproducibilidad.
        model_names: Lista de modelos a evaluar.
        epochs: Número máximo de épocas de entrenamiento.
    """

    # --- Modo debug (para probar que MLflow funciona) ---
    debug: bool = False

    # --- Semilla para reproducibilidad ---
    seed: int = 42

    # --- Targets primarios (los que predice el modelo) ---
    targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"
    ])

    # --- Validación cruzada ---
    n_folds: int = 5
    group_col: str = "Sampling_Date"

    # --- Imágenes ---
    img_size: int = 224

    # --- Entrenamiento ---
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    dropout: float = 0.3

    # --- Modelos a evaluar ---
    model_names: List[str] = field(default_factory=lambda: [
        "resnet50",
        "efficientnet_b2",
        "convnext_tiny",
        "maxvit_tiny",
        "vit_small",
        "swin_tiny",
    ])

    # --- Rutas ---
    model_dir: Path = Path("models")

    def __post_init__(self):
        """Si debug=True, reducir epochs y paciencia para test rápido."""
        if self.debug:
            self.epochs = 5
            self.early_stopping_patience = 3
            self.scheduler_patience = 2
