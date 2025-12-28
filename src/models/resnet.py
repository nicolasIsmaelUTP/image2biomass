from typing import Optional

import torch
from torch import nn
from torchvision import models


def create_resnet(num_outputs: int, weights: Optional[str] = "IMAGENET1K_V1") -> nn.Module:
    try:
        resnet_weights = getattr(models.ResNet18_Weights, weights)
        model = models.resnet18(weights=resnet_weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model
