import os
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * len(images)
        return total_loss / len(loader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, checkpoint_path: Path, early_stopping_patience: int = None) -> List[Dict[str, float]]:
        history = []
        best_val = float("inf")
        epochs_without_improvement = 0
        checkpoint_path = Path(checkpoint_path)
        os.makedirs(checkpoint_path.parent, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.run_epoch(train_loader, train=True)
            val_loss = self.run_epoch(val_loader, train=False)
            
            if val_loss < best_val:
                best_val = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                
            history.append({
                "epoch": epoch, 
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "best_val": best_val,
                "epochs_without_improvement": epochs_without_improvement
            })
            print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f} | no improv {epochs_without_improvement}")
            
            # Early stopping
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={early_stopping_patience})")
                break
                
        return history
