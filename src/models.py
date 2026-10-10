"""
models.py - Model definitions with Transfer Learning support.

Implements two pretrained architectures for binary casting defect classification:
  1. MobileNetV3-Small  — lightweight, mobile-optimized (~2.5M params)
  2. EfficientNet-B0    — compound-scaled, higher accuracy (~5.3M params)

Both models are loaded with ImageNet-pretrained weights. The classifier head is
replaced with a custom head for our binary task (ok vs. defective). The backbone
can be frozen initially and unfrozen later for fine-tuning with a lower learning
rate (discriminative learning rates).
"""

import logging
from typing import List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Model Factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build and return a classification model with transfer learning.

    Args:
        model_name:      One of "mobilenetv3_small" or "efficientnet_b0".
        num_classes:     Number of output classes.
        pretrained:      Whether to load ImageNet-pretrained weights.
        freeze_backbone: Whether to freeze all backbone layers initially.

    Returns:
        A PyTorch nn.Module ready for training.
    """
    if model_name == "mobilenetv3_small":
        model = _build_mobilenetv3_small(num_classes, pretrained)
    elif model_name == "efficientnet_b0":
        model = _build_efficientnet_b0(num_classes, pretrained)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Supported: 'mobilenetv3_small', 'efficientnet_b0'"
        )

    if freeze_backbone:
        frozen_count = freeze_backbone_layers(model, model_name)
        logger.info(f"  Frozen {frozen_count} backbone parameters (head remains trainable)")

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters:     {total_params:>10,}")
    logger.info(f"  Trainable parameters: {trainable_params:>10,}")

    return model


# ---------------------------------------------------------------------------
#  MobileNetV3-Small
# ---------------------------------------------------------------------------

def _build_mobilenetv3_small(num_classes: int, pretrained: bool) -> nn.Module:
    """
    MobileNetV3-Small with a custom classification head.

    Original head: Linear(576, 1024) -> Hardswish -> Dropout -> Linear(1024, 1000)
    Our head:      Linear(576, 1024) -> Hardswish -> Dropout(0.3) -> Linear(1024, num_classes)
    """
    logger.info("Building MobileNetV3-Small")

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1024, num_classes),
    )

    logger.info(f"  Replaced classifier head: {in_features} -> 1024 -> {num_classes}")
    return model


# ---------------------------------------------------------------------------
#  EfficientNet-B0
# ---------------------------------------------------------------------------

def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    """
    EfficientNet-B0 with a custom classification head.

    Original head: Dropout(0.2) -> Linear(1280, 1000)
    Our head:      Dropout(0.3) -> Linear(1280, num_classes)
    """
    logger.info("Building EfficientNet-B0")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    logger.info(f"  Replaced classifier head: {in_features} -> {num_classes}")
    return model


# ---------------------------------------------------------------------------
#  Backbone Freezing / Unfreezing
# ---------------------------------------------------------------------------

def freeze_backbone_layers(model: nn.Module, model_name: str) -> int:
    """
    Freeze all layers except the classifier head.

    Returns the number of frozen parameters.
    """
    frozen_count = 0

    if model_name == "mobilenetv3_small":
        # Freeze everything in model.features (the convolutional backbone)
        for param in model.features.parameters():
            param.requires_grad = False
            frozen_count += 1

    elif model_name == "efficientnet_b0":
        # Freeze everything in model.features
        for param in model.features.parameters():
            param.requires_grad = False
            frozen_count += 1

    return frozen_count


def unfreeze_backbone(model: nn.Module, model_name: str) -> int:
    """
    Unfreeze all backbone layers for full fine-tuning.

    Call this after the classifier head has been trained for a few epochs.

    Returns the number of newly unfrozen parameters.
    """
    unfrozen_count = 0

    if model_name == "mobilenetv3_small":
        for param in model.features.parameters():
            param.requires_grad = True
            unfrozen_count += 1

    elif model_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = True
            unfrozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Unfroze {unfrozen_count} backbone parameters")
    logger.info(f"  Total trainable parameters now: {trainable:,}")

    return unfrozen_count


def get_optimizer_param_groups(
    model: nn.Module,
    model_name: str,
    head_lr: float,
    backbone_lr: float,
) -> List[dict]:
    """
    Create parameter groups with discriminative learning rates.

    The backbone uses a much lower learning rate than the freshly initialized
    classifier head. This prevents catastrophic forgetting of the pretrained
    features while still allowing them to adapt to our domain.

    Args:
        model:       The full model.
        model_name:  Model identifier string.
        head_lr:     Learning rate for the classifier head.
        backbone_lr: Learning rate for the backbone layers.

    Returns:
        List of param group dicts for torch.optim.
    """
    if model_name in ("mobilenetv3_small", "efficientnet_b0"):
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]

    logger.info(f"  Optimizer groups: backbone_lr={backbone_lr}, head_lr={head_lr}")
    return param_groups
