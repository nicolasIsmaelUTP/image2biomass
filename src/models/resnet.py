from typing import Optional

import torch.nn as nn
import torchvision.models as models


def create_resnet(num_outputs: int, weights: Optional[str] = "IMAGENET1K_V1", freeze_backbone: bool = True) -> nn.Module:
    try:
        resnet_weights = getattr(models.ResNet18_Weights, weights)
        model = models.resnet18(weights=resnet_weights)
    except Exception:
        model = models.resnet18(weights=None)

    # Congelar todas las capas del backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    # Reemplazar la capa FC y dejar sus parámetros entrenables
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_outputs)
    )

    return model