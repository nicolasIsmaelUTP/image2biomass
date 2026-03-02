"""
Factoría de modelos con backbone congelado y cabeza MLP idéntica.

Arquitectura común (control experimental):
    Imagen RGB
        → Backbone preentrenado en ImageNet (CONGELADO)
        → Global Average Pooling / CLS Token
        → Linear(num_features, 128) → ReLU → Dropout(0.3)
        → Linear(128, 3)  [3 targets primarios]

Modelos disponibles (ver docs/metodologia_experimental.md, sección 3):
    - resnet50:         CNN clásica (baseline)
    - efficientnet_b2:  CNN eficiente moderna
    - convnext_tiny:    CNN híbrida (puente CNN-ViT)
    - maxvit_tiny:      Híbrido (CNN + atención)
    - vit_small:        Transformer puro (atención global)
    - swin_tiny:        Transformer jerárquico (multi-escala)

Se usan variantes Small/Tiny para mantener coherencia con dataset pequeño (N=357).
"""

import torch.nn as nn
import timm


# Mapeo de nombre corto → nombre exacto en timm
MODEL_REGISTRY = {
    "resnet50": "resnet50",
    "efficientnet_b2": "efficientnet_b2",
    "convnext_tiny": "convnext_tiny",
    "maxvit_tiny": "maxvit_tiny_rw_224",
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


class BiomassModel(nn.Module):
    """
    Modelo completo: backbone (congelado) + cabeza MLP.

    El backbone extrae features de la imagen.
    La cabeza MLP mapea esas features a los 3 targets primarios.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def create_model(
    model_name: str,
    num_outputs: int = 3,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Crea un modelo con backbone preentrenado + cabeza MLP idéntica.

    Args:
        model_name: Nombre corto del modelo (ej: "resnet50").
        num_outputs: Número de salidas (3 targets primarios).
        dropout: Probabilidad de dropout en la cabeza.
        freeze_backbone: Si True, congela todos los pesos del backbone.

    Returns:
        BiomassModel listo para entrenar.
    """
    # Verificar que el nombre es válido
    if model_name not in MODEL_REGISTRY:
        nombres_validos = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Modelo '{model_name}' no encontrado. "
            f"Opciones: {nombres_validos}"
        )

    timm_name = MODEL_REGISTRY[model_name]

    # Crear backbone SIN cabeza de clasificación (num_classes=0)
    # Esto hace que el modelo devuelva solo las features (vector 1D)
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

    # Congelar todos los parámetros del backbone
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Obtener dimensión de features del backbone
    num_features = backbone.num_features

    # Cabeza MLP idéntica para TODOS los modelos (control experimental)
    head = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(128, num_outputs),
    )

    # Ensamblar modelo completo
    model = BiomassModel(backbone, head)

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Cuenta parámetros entrenables y totales del modelo.

    Útil para verificar que solo la cabeza es entrenable.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    return {
        "trainable": trainable,
        "total": total,
        "frozen": frozen,
        "trainable_ratio": f"{trainable / total:.4f}" if total > 0 else "0",
    }
