"""
inference.py - Single-image inference with a trained model.

Provides both a Python API and a CLI for predicting whether a casting
image is "ok" or "defective", along with a confidence score.

Usage:
    python -m src.inference --model mobilenetv3_small --image path/to/image.jpeg
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import Config
from src.data import get_eval_transforms
from src.models import build_model

logger = logging.getLogger(__name__)


def load_trained_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        cfg:    Configuration object (used to resolve model name and paths).
        device: Target device.

    Returns:
        The model in eval mode with loaded weights.
    """
    checkpoint_path = os.path.join(cfg.model_output_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    model = build_model(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    transform,
    class_names: list,
    device: torch.device,
) -> dict:
    """
    Predict the class of a single casting image.

    Args:
        model:       Trained model in eval mode.
        image_path:  Path to the input JPEG image.
        transform:   Preprocessing transform (eval transforms).
        class_names: List of class names (index-aligned with model output).
        device:      Target device.

    Returns:
        Dictionary with keys:
          - "predicted_class": str ("ok" or "defective")
          - "confidence": float (0.0 to 1.0)
          - "probabilities": dict mapping class names to their probabilities
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Forward pass
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    # Extract prediction
    confidence, predicted_idx = probabilities.max(0)
    predicted_class = class_names[predicted_idx.item()]

    result = {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            name: round(probabilities[i].item(), 4)
            for i, name in enumerate(class_names)
        },
    }

    return result


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single casting image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--model", type=str, default="mobilenetv3_small",
        choices=["mobilenetv3_small", "efficientnet_b0"],
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Config()
    cfg.model.model_name = args.model
    cfg.train.output_dir = args.output_dir
    cfg.train.device = args.device

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(cfg, device)

    # Prepare transform
    transform = get_eval_transforms(cfg.data.img_size)

    # Predict
    result = predict_single_image(model, args.image, transform, cfg.data.class_names, device)

    # Display
    print("\n" + "=" * 40)
    print("  Inference Result")
    print("=" * 40)
    print(f"  Image:       {args.image}")
    print(f"  Model:       {args.model}")
    print(f"  Prediction:  {result['predicted_class']}")
    print(f"  Confidence:  {result['confidence'] * 100:.1f}%")
    print(f"  Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"    {cls}: {prob * 100:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
