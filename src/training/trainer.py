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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, checkpoint_path: Path) -> List[Dict[str, float]]:
        history = []
        best_val = float("inf")
        checkpoint_path = Path(checkpoint_path)
        os.makedirs(checkpoint_path.parent, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.run_epoch(train_loader, train=True)
            val_loss = self.run_epoch(val_loader, train=False)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val})
            print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")
        return history
