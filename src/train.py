"""
train.py - Training loop with validation, early stopping, and checkpointing.

Handles:
  - Weighted CrossEntropyLoss for class imbalance
  - Two-phase training: frozen backbone → unfrozen fine-tuning
  - Discriminative learning rates (lower LR for backbone, higher for head)
  - ReduceLROnPlateau scheduler
  - Early stopping based on validation loss
  - Per-epoch metric logging
  - Best model checkpoint saving

Usage (from project root):
    python -m src.train --model mobilenetv3_small --epochs 25 --batch_size 32
    python -m src.train --model efficientnet_b0   --epochs 25 --batch_size 32
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.data import build_dataloaders
from src.models import (
    build_model,
    unfreeze_backbone,
    get_optimizer_param_groups,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Training & Validation Steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        (average_loss, accuracy) over the entire epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on the validation set.

    Returns:
        (average_loss, accuracy) over the entire validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
#  Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when the monitored metric stops improving.

    Tracks validation loss and saves the best model checkpoint.
    """

    def __init__(self, patience: int, checkpoint_path: str):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.

        Returns True if patience is exhausted (should stop).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
            return False
        else:
            self.counter += 1
            logger.info(
                f"  EarlyStopping: {self.counter}/{self.patience} "
                f"(best={self.best_loss:.4f} at epoch {self.best_epoch})"
            )
            return self.counter >= self.patience


# ---------------------------------------------------------------------------
#  Main Training Function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Dict:
    """
    Full training pipeline.

    1. Build data loaders and class weights.
    2. Build model with frozen backbone.
    3. Phase 1: Train the classifier head only.
    4. Phase 2: Unfreeze backbone and fine-tune with lower LR.
    5. Save the best checkpoint and training history.

    Returns:
        Training history dictionary.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----- Data -----
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    class_weights = class_weights.to(device)

    # ----- Model -----
    logger.info("\n" + "=" * 50)
    logger.info("Building model")
    logger.info("=" * 50)
    model = build_model(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    )
    model = model.to(device)

    # ----- Loss -----
    if cfg.train.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"  Using weighted CrossEntropyLoss: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ----- Optimizer (head only initially) -----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # ----- LR Scheduler -----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.lr_scheduler_factor,
        patience=cfg.train.lr_scheduler_patience,
    )

    # ----- Early Stopping -----
    checkpoint_path = os.path.join(cfg.model_output_dir, "best_model.pth")
    early_stopping = EarlyStopping(cfg.train.early_stopping_patience, checkpoint_path)

    # ----- Training History -----
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_time": [],
    }

    # ----- Training Loop -----
    logger.info("\n" + "=" * 50)
    logger.info(f"Starting training: {cfg.model.model_name}")
    logger.info("=" * 50)

    backbone_unfrozen = not cfg.model.freeze_backbone

    for epoch in range(1, cfg.train.num_epochs + 1):
        epoch_start = time.time()

        # --- Phase transition: unfreeze backbone ---
        if (
            cfg.model.freeze_backbone
            and not backbone_unfrozen
            and epoch > cfg.model.unfreeze_after_epoch
        ):
            logger.info(f"\n>>> Epoch {epoch}: Unfreezing backbone for fine-tuning <<<")
            unfreeze_backbone(model, cfg.model.model_name)
            backbone_unfrozen = True

            # Rebuild optimizer with discriminative learning rates
            param_groups = get_optimizer_param_groups(
                model, cfg.model.model_name,
                head_lr=cfg.train.learning_rate,
                backbone_lr=cfg.train.backbone_lr,
            )
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=cfg.train.weight_decay
            )
            # Reset scheduler for the new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=cfg.train.lr_scheduler_factor,
                patience=cfg.train.lr_scheduler_patience,
            )

        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # --- Validate ---
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # --- Scheduler step ---
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR

        epoch_time = time.time() - epoch_start

        # --- Log ---
        logger.info(
            f"Epoch [{epoch:>2}/{cfg.train.num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}  |  Time: {epoch_time:.1f}s"
        )

        # --- Record history ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        # --- Early stopping check ---
        if early_stopping.step(val_loss, model, epoch):
            logger.info(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    # ----- Save training history -----
    history_path = os.path.join(cfg.model_output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nTraining history saved to: {history_path}")
    logger.info(f"Best model saved to: {checkpoint_path}")
    logger.info(f"Best val_loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch}")

    return history


# ---------------------------------------------------------------------------
#  CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a casting defect classification model."
    )
    parser.add_argument(
        "--model", type=str, default="mobilenetv3_small",
        choices=["mobilenetv3_small", "efficientnet_b0"],
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the head.")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="LR for backbone after unfreezing.")
    parser.add_argument("--unfreeze_after", type=int, default=5, help="Unfreeze backbone after N epochs.")
    parser.add_argument("--data_dir", type=str, default="casting_512x512/casting_512x512", help="Dataset root.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from CLI args
    cfg = Config()
    cfg.model.model_name = args.model
    cfg.train.num_epochs = args.epochs
    cfg.data.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.backbone_lr = args.backbone_lr
    cfg.model.unfreeze_after_epoch = args.unfreeze_after
    cfg.data.data_dir = args.data_dir
    cfg.train.output_dir = args.output_dir
    cfg.train.device = args.device
    cfg.train.use_class_weights = not args.no_class_weights

    # Ensure output dirs exist
    cfg.__post_init__()

    # Setup logging
    log_path = os.path.join(cfg.model_output_dir, cfg.train.log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(cfg.summary())

    # Train
    train(cfg)


if __name__ == "__main__":
    main()
