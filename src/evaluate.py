"""
evaluate.py - Model evaluation, metrics computation, and error analysis.

Responsibilities:
  1. Load the best checkpoint and run inference on the test set.
  2. Compute Accuracy, Precision, Recall, F1-Score (per-class and macro).
  3. Generate and save a Confusion Matrix heatmap.
  4. Plot training history curves (loss & accuracy).
  5. Visualize misclassified images (False Positives & False Negatives)
     with explanations for *why* the model might have failed.

Usage:
    python -m src.evaluate --model mobilenetv3_small
    python -m src.evaluate --model efficientnet_b0
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/Kaggle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import Config
from src.data import build_dataloaders, get_eval_transforms, CastingDataset
from src.models import build_model
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Inference on Test Set
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader and collect predictions.

    Returns:
        all_labels:  Ground-truth labels (N,)
        all_preds:   Predicted class indices (N,)
        all_probs:   Softmax probabilities (N, num_classes)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = probs.max(1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute and log classification metrics.

    Returns a dictionary of all metrics for later serialization.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    logger.info("\n" + "=" * 50)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 50)
    logger.info(f"\nAccuracy: {acc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    metrics = {
        "accuracy": float(acc),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
            }
            for i, name in enumerate(class_names)
        },
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
        },
    }

    return metrics


# ---------------------------------------------------------------------------
#  Confusion Matrix Plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    model_name: str,
):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"Confusion Matrix — {model_name}",
    )

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved to: {save_path}")


# ---------------------------------------------------------------------------
#  Training History Plots
# ---------------------------------------------------------------------------

def plot_training_history(history: Dict, save_dir: str, model_name: str):
    """Plot loss and accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Curves — {model_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy Curves — {model_name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Training curves saved to: {save_path}")


# ---------------------------------------------------------------------------
#  Error Analysis: Visualize Misclassified Samples
# ---------------------------------------------------------------------------

def visualize_misclassifications(
    test_loader: DataLoader,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str,
    max_samples: int = 8,
):
    """
    Visualize False Positives and False Negatives with confidence scores.

    False Positive (FP): Model predicts "defective" but the part is actually "ok".
        → This is costly: good parts are rejected, reducing yield.

    False Negative (FN): Model predicts "ok" but the part is actually "defective".
        → This is dangerous: defective parts slip through quality control.
    """
    # Reconstruct image paths from the dataset
    dataset = test_loader.dataset
    image_paths = dataset.image_paths

    # ImageNet denormalization for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # --- Identify FPs and FNs ---
    # FP: true=0 (ok), pred=1 (defective)
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    # FN: true=1 (defective), pred=0 (ok)
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    logger.info(f"\n  False Positives: {len(fp_indices)}")
    logger.info(f"  False Negatives: {len(fn_indices)}")

    for error_type, indices, label in [
        ("False_Positives", fp_indices, "Predicted DEFECTIVE, Actually OK"),
        ("False_Negatives", fn_indices, "Predicted OK, Actually DEFECTIVE"),
    ]:
        if len(indices) == 0:
            logger.info(f"  No {error_type} found — perfect on this category!")
            continue

        # Pick the most confident mistakes (most interesting for analysis)
        # For FP: highest confidence in class 1 (defective)
        # For FN: highest confidence in class 0 (ok)
        mistake_class = 1 if error_type == "False_Positives" else 0
        confidence_in_wrong = y_probs[indices, mistake_class]
        sorted_idx = np.argsort(-confidence_in_wrong)  # descending
        selected = indices[sorted_idx[:max_samples]]

        n_show = len(selected)
        cols = min(4, n_show)
        rows = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n_show == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle(f"{error_type.replace('_', ' ')} — {model_name}\n({label})", fontsize=13)

        for ax_idx, sample_idx in enumerate(selected):
            img = Image.open(image_paths[sample_idx]).convert("RGB")
            img = img.resize((224, 224))
            img_np = np.array(img) / 255.0

            true_label = class_names[y_true[sample_idx]]
            pred_label = class_names[y_pred[sample_idx]]
            confidence = y_probs[sample_idx, y_pred[sample_idx]] * 100

            axes[ax_idx].imshow(img_np)
            axes[ax_idx].set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
                fontsize=9, color="red",
            )
            axes[ax_idx].axis("off")

        # Hide unused subplots
        for ax_idx in range(n_show, len(axes)):
            axes[ax_idx].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{error_type.lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  {error_type} visualization saved to: {save_path}")


# ---------------------------------------------------------------------------
#  Main Evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Config):
    """
    Full evaluation pipeline:
      1. Rebuild the data pipeline (same split via fixed seed).
      2. Load the best checkpoint.
      3. Run inference on the test set.
      4. Compute and save metrics.
      5. Generate visualizations.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----- Data -----
    _, _, test_loader, _ = build_dataloaders(cfg)

    # ----- Model -----
    checkpoint_path = os.path.join(cfg.model_output_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    model = build_model(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=False,  # we load our own weights
        freeze_backbone=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    logger.info(f"  Loaded checkpoint: {checkpoint_path}")

    # ----- Predict -----
    y_true, y_pred, y_probs = predict_on_loader(model, test_loader, device)

    # ----- Metrics -----
    metrics = compute_metrics(y_true, y_pred, cfg.data.class_names)

    metrics_path = os.path.join(cfg.model_output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to: {metrics_path}")

    # ----- Confusion Matrix -----
    plot_confusion_matrix(
        y_true, y_pred, cfg.data.class_names,
        save_path=os.path.join(cfg.model_output_dir, "confusion_matrix.png"),
        model_name=cfg.model.model_name,
    )

    # ----- Training History Plots -----
    history_path = os.path.join(cfg.model_output_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, cfg.model_output_dir, cfg.model.model_name)

    # ----- Error Analysis -----
    logger.info("\nError Analysis: Visualizing misclassified samples")
    visualize_misclassifications(
        test_loader, y_true, y_pred, y_probs,
        cfg.data.class_names, cfg.model_output_dir,
        cfg.model.model_name, max_samples=8,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Evaluation complete!")
    logger.info("=" * 50)

    return metrics


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument(
        "--model", type=str, default="mobilenetv3_small",
        choices=["mobilenetv3_small", "efficientnet_b0"],
    )
    parser.add_argument("--data_dir", type=str, default="casting_512x512/casting_512x512")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.model.model_name = args.model
    cfg.data.data_dir = args.data_dir
    cfg.train.output_dir = args.output_dir
    cfg.train.device = args.device
    cfg.__post_init__()

    log_path = os.path.join(cfg.model_output_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    evaluate(cfg)


if __name__ == "__main__":
    main()
