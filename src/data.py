"""
data.py - Data loading, splitting, augmentation, and class-weight computation.

Responsibilities:
  1. Discover all images from the raw casting_512x512 dataset.
  2. Perform a stratified Train / Validation / Test split.
  3. Apply on-the-fly data augmentation (training only).
  4. Compute class weights to address the ~60/40 class imbalance.
  5. Return ready-to-use DataLoaders.

Dataset structure expected:
    casting_512x512/casting_512x512/
        def_front/   (781 JPEG images - defective castings)
        ok_front/    (519 JPEG images - normal castings)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int) -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Augmentations chosen for manufacturing defect images:
      - Horizontal/Vertical flips (defects can appear in any orientation)
      - Small rotations (±15°) to simulate slight misalignment
      - Color jitter for lighting variation on the factory floor
      - Random affine for minor scale/translation shifts
      - Normalize with ImageNet stats (required for pretrained models)
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(img_size: int) -> transforms.Compose:
    """Validation / Test transforms — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
#  Custom Dataset
# ---------------------------------------------------------------------------

class CastingDataset(Dataset):
    """
    Custom Dataset for casting defect images.

    Attributes:
        image_paths: List of absolute paths to JPEG images.
        labels:      Corresponding integer labels (0=ok, 1=defective).
        transform:   torchvision transform to apply to each image.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: transforms.Compose = None,
    ):
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to RGB (some images may be grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
#  Data Discovery & Splitting
# ---------------------------------------------------------------------------

def discover_images(data_dir: str, class_map: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """
    Walk the dataset directory and collect all image paths with their labels.

    Args:
        data_dir:  Root of the dataset (e.g., casting_512x512/casting_512x512).
        class_map: Mapping from folder name to integer label.

    Returns:
        image_paths: List of file paths.
        labels:      Corresponding integer labels.
    """
    image_paths = []
    labels = []
    valid_extensions = {".jpeg", ".jpg", ".png", ".bmp"}

    for folder_name, label in class_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")

        count = 0
        for fname in sorted(os.listdir(folder_path)):
            if Path(fname).suffix.lower() in valid_extensions:
                image_paths.append(os.path.join(folder_path, fname))
                labels.append(label)
                count += 1

        logger.info(f"  Found {count:>5} images in '{folder_name}' (label={label})")

    return image_paths, labels


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Perform stratified train/val/test split.

    The split is done in two stages:
      1. Split off the test set first.
      2. Split the remainder into train and validation.

    Stratification ensures class proportions are preserved in every split.

    Returns:
        Dictionary with keys "train", "val", "test", each mapping to
        (image_paths, labels) tuples.
    """
    # Stage 1: separate test set
    paths_remaining, paths_test, labels_remaining, labels_test = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Stage 2: from the remainder, separate validation set
    # val_ratio is fraction of *original* data, so adjust for remaining size
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)

    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_remaining, labels_remaining,
        test_size=adjusted_val_ratio,
        stratify=labels_remaining,
        random_state=seed,
    )

    splits = {
        "train": (paths_train, labels_train),
        "val": (paths_val, labels_val),
        "test": (paths_test, labels_test),
    }

    for name, (paths, lbls) in splits.items():
        unique, counts = np.unique(lbls, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"  {name:>5} split: {len(paths):>4} images | class distribution: {dist}")

    return splits


# ---------------------------------------------------------------------------
#  Class Weight Computation
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training labels.

    Formula:  weight_c = N_total / (num_classes * N_c)

    This gives higher weight to under-represented classes so the loss
    function penalizes mistakes on minority samples more heavily.

    Args:
        labels:      Training set labels.
        num_classes:  Number of classes.

    Returns:
        Tensor of shape (num_classes,) with the computed weights.
    """
    labels_np = np.array(labels)
    total = len(labels_np)
    weights = []

    for c in range(num_classes):
        n_c = (labels_np == c).sum()
        w = total / (num_classes * n_c) if n_c > 0 else 1.0
        weights.append(w)
        logger.info(f"  Class {c}: {n_c} samples -> weight = {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
#  Public API: build everything
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    End-to-end data pipeline: discover → split → augment → load.

    Args:
        cfg: Master configuration object.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    logger.info("=" * 50)
    logger.info("Building data pipeline")
    logger.info("=" * 50)

    # 1. Discover images
    logger.info(f"\n[1/4] Discovering images from: {cfg.data.data_dir}")
    image_paths, labels = discover_images(cfg.data.data_dir, cfg.data.class_map)
    logger.info(f"  Total images found: {len(image_paths)}")

    # 2. Stratified split
    logger.info(f"\n[2/4] Splitting dataset (test={cfg.data.test_ratio}, val={cfg.data.val_ratio})")
    splits = split_dataset(
        image_paths, labels,
        test_ratio=cfg.data.test_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.random_seed,
    )

    # 3. Compute class weights from training set
    logger.info("\n[3/4] Computing class weights for imbalance handling")
    class_weights = compute_class_weights(splits["train"][1], cfg.model.num_classes)

    # 4. Create Datasets & DataLoaders
    logger.info(f"\n[4/4] Creating DataLoaders (batch_size={cfg.data.batch_size})")

    train_dataset = CastingDataset(
        *splits["train"], transform=get_train_transforms(cfg.data.img_size)
    )
    val_dataset = CastingDataset(
        *splits["val"], transform=get_eval_transforms(cfg.data.img_size)
    )
    test_dataset = CastingDataset(
        *splits["test"], transform=get_eval_transforms(cfg.data.img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    logger.info(f"\n  Train batches: {len(train_loader)}")
    logger.info(f"  Val   batches: {len(val_loader)}")
    logger.info(f"  Test  batches: {len(test_loader)}")
    logger.info("Data pipeline ready.\n")

    return train_loader, val_loader, test_loader, class_weights
