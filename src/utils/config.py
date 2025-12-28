from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4
    epochs: int = 5
    model_dir: Path = Path("models")
    checkpoint_name: str = "baseline_resnet18.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / self.checkpoint_name
