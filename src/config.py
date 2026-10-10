"""
config.py - Centralized configuration for the Casting Defect Classification project.

All hyperparameters, paths, and training settings are defined here as dataclasses
so they can be easily serialized, logged, and modified from the command line.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Path to the RAW (non-augmented) dataset root
    data_dir: str = "casting_512x512/casting_512x512"

    # Image resizing target (both models expect 224x224)
    img_size: int = 224

    # Train / Validation / Test split ratios
    test_ratio: float = 0.15
    val_ratio: float = 0.15  # fraction of the *remaining* data after test split

    # Reproducibility
    random_seed: int = 42

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # Class label mapping  (folder name -> integer label)
    # ok_front = 0  (Normal / Negative class)
    # def_front = 1 (Defective / Positive class)
    class_map: dict = field(
        default_factory=lambda: {"ok_front": 0, "def_front": 1}
    )
    class_names: List[str] = field(
        default_factory=lambda: ["ok", "defective"]
    )


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Supported: "mobilenetv3_small", "efficientnet_b0"
    model_name: str = "mobilenetv3_small"

    # Number of output classes (binary: ok vs defective)
    num_classes: int = 2

    # Use ImageNet-pretrained weights for transfer learning
    pretrained: bool = True

    # Backbone freezing strategy
    freeze_backbone: bool = True
    unfreeze_after_epoch: int = 5  # unfreeze backbone after N epochs of head-only training


@dataclass
class TrainConfig:
    """Configuration for the training loop."""

    num_epochs: int = 25

    # Optimizer (AdamW)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Lower learning rate for backbone layers after unfreezing
    backbone_lr: float = 1e-5

    # Learning rate scheduler (ReduceLROnPlateau)
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 3

    # Early stopping
    early_stopping_patience: int = 7

    # Use class weights in the loss function to handle imbalance
    use_class_weights: bool = True

    # Output directory for checkpoints, logs, and plots
    output_dir: str = "outputs"

    # Device
    device: str = "cuda"

    # Logging
    log_file: str = "training.log"


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.train.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.train.output_dir, self.model.model_name), exist_ok=True
        )

    @property
    def model_output_dir(self) -> str:
        """Per-model output directory (e.g., outputs/mobilenetv3_small/)."""
        return os.path.join(self.train.output_dir, self.model.model_name)

    def summary(self) -> str:
        """Return a human-readable summary of all settings."""
        lines = ["=" * 60, "  Configuration Summary", "=" * 60]
        for section_name in ["data", "model", "train"]:
            section = getattr(self, section_name)
            lines.append(f"\n[{section_name.upper()}]")
            for k, v in vars(section).items():
                lines.append(f"  {k:.<30} {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
