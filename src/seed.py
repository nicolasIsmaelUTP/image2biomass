"""
Control de reproducibilidad.

Fija semillas en todas las librerías relevantes y activa modo
determinístico de CUDA para garantizar resultados reproducibles.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fija todas las semillas para reproducibilidad total.

    Aplica a: random, numpy, torch (CPU y CUDA).
    Activa modo determinístico de cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modo determinístico: resultados reproducibles pero más lento
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
