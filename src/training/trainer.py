"""
Trainer para entrenamiento de modelos de biomasa.

Incluye:
    - ReduceLROnPlateau: reduce learning rate cuando val_loss se estanca.
    - Early stopping: detiene el entrenamiento si no hay mejora.
    - Checkpoint: guarda el mejor modelo automáticamente.

Configuración (ver docs/metodologia_final_justificada.md, sección 14):
    - Optimizer: Adam, lr=1e-3
    - Scheduler: ReduceLROnPlateau (factor=0.5, paciencia=5)
    - Early stopping: paciencia=15
    - Loss: MSE sin ponderación
"""

import os
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class Trainer:
    """
    Entrena un modelo con scheduler y early stopping.

    Uso:
        trainer = Trainer(model, criterion, optimizer, device)
        history = trainer.fit(train_loader, val_loader, epochs=100,
                              checkpoint_path="best.pt",
                              early_stopping_patience=15)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # ReduceLROnPlateau: baja el lr cuando val_loss se estanca
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """
        Ejecuta una época completa (train o eval).

        Args:
            loader: DataLoader con los datos.
            train: Si True, actualiza pesos. Si False, solo evalúa.

        Returns:
            Loss promedio de la época.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0

        with torch.set_grad_enabled(train):
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)

                # Backward pass (solo en entrenamiento)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(images)

        avg_loss = total_loss / len(loader.dataset)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: Path,
        early_stopping_patience: int = 15,
    ) -> List[Dict]:
        """
        Entrena el modelo completo con early stopping.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación.
            epochs: Número máximo de épocas.
            checkpoint_path: Ruta donde guardar el mejor modelo.
            early_stopping_patience: Épocas sin mejora antes de parar.

        Returns:
            Lista de diccionarios con métricas por época.
        """
        checkpoint_path = Path(checkpoint_path)
        os.makedirs(checkpoint_path.parent, exist_ok=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        history = []

        for epoch in range(1, epochs + 1):
            # Entrenar y evaluar
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)

            # Actualizar scheduler con la val_loss
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1

            # Registrar historial
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "lr": current_lr,
                "epochs_no_improve": epochs_no_improve,
            })

            # Mostrar progreso
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"best={best_val_loss:.4f} | "
                f"lr={current_lr:.1e} | "
                f"no_improve={epochs_no_improve}"
            )

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"  >> Early stopping en epoch {epoch} "
                    f"(paciencia={early_stopping_patience})"
                )
                break

        return history
