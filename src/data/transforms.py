"""
Augmentations para imágenes de biomasa.

Train: Resize + HFlip + Rotación + Brillo/Contraste + Normalización ImageNet.
Val/Test: Resize + Normalización ImageNet.

Justificación (ver docs/metodologia_final_justificada.md, sección 9):
- HFlip: la orientación del pasto no afecta biomasa.
- Rotación ±15°: simula variación del ángulo de captura.
- Brillo/Contraste: simula diferencias de iluminación entre sitios.
"""

from torchvision import transforms


# Normalización estándar de ImageNet (para transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """
    Augmentations de entrenamiento (ligeras).

    Resolución original 2000×1000 → resize a img_size × img_size.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224):
    """Transforms de validación/test (sin augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
