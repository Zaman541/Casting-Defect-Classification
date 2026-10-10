"""
app.py - REST API for casting defect classification.

Provides a FastAPI endpoint that accepts an image upload and returns
the predicted class ("ok" or "defective") with a confidence score.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import io
import logging
import os
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image

# Add project root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import get_eval_transforms
from src.models import build_model
from src.inference import predict_single_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Casting Defect Classifier API",
    description=(
        "Upload a casting image to classify it as **ok** or **defective**. "
        "Supports MobileNetV3-Small and EfficientNet-B0 models."
    ),
    version="1.0.0",
)

# Configuration — select model via environment variable (default: EfficientNet-B0)
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Build config
cfg = Config()
cfg.model.model_name = MODEL_NAME
cfg.train.output_dir = OUTPUT_DIR

# Load model at startup
device = torch.device(DEVICE)
checkpoint_path = os.path.join(cfg.model_output_dir, "best_model.pth")

model = build_model(
    model_name=cfg.model.model_name,
    num_classes=cfg.model.num_classes,
    pretrained=False,
    freeze_backbone=False,
)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

transform = get_eval_transforms(cfg.data.img_size)

logger.info(f"Model '{MODEL_NAME}' loaded from {checkpoint_path}")
logger.info(f"Running on device: {DEVICE}")


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a casting image as ok or defective.

    Accepts:
        - JPEG or PNG image file upload.

    Returns:
        - predicted_class: "ok" or "defective"
        - confidence: float between 0 and 1
        - probabilities: per-class probability breakdown
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG accepted.",
        )

    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, pred_idx = probs.max(0)

        predicted_class = cfg.data.class_names[pred_idx.item()]

        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(cfg.data.class_names)
            },
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
