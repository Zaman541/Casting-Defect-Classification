# Casting Defect Classification

Binary image classifier that labels industrial casting surfaces as **`ok`** or **`defective`**. The repo contains a reproducible training and evaluation pipeline, a comparison of two pretrained CNNs, an error analysis, and a live REST API.

Built as a machine learning engineering assessment (PyTorch, FastAPI, Docker, Render).

**Live API:** https://classification-of-defective-mechanical.onrender.com/docs

\---

## 1\. Summary

|Model|Role|Parameters|Test accuracy|Defective precision|Defective recall|Defective F1|Missed defects (FN)|False alarms (FP)|
|-|-|-|-|-|-|-|-|-|
|MobileNetV3-Small|Baseline|1,519,906|75.38%|0.99|0.60|0.74|47|1|
|**EfficientNet-B0**|**Selected / deployed**|4,010,110|**97.95%**|**1.00**|**0.97**|**0.98**|**4**|**0**|

* **Problem framing:** pass/fail inspection of casting images. A missed defect (false negative) is the costly error, so recall on the `defective` class is reported alongside accuracy.
* **Selected model:** EfficientNet-B0, served by the API.
* **Main finding from error analysis:** the remaining errors are low-margin predictions on parts whose defects are tiny at the 224x224 input size (see [Section 5](#5-evaluation-and-error-analysis)).
* **Caveat:** one split, one seed, 195 test images. See [Limitations](#8-limitations).

\---

## 2\. Live Demo



The service runs on Render's **free plan, CPU only**, and spins down when idle, so the first request after a quiet period may be slow.



**Link:** [https://classification-of-defective-mechanical.onrender.com/docs](https://classification-of-defective-mechanical.onrender.com/docs)





Response format (values illustrative):



```json
{
  "predicted\_class": "defective",
  "confidence": 0.9731,
  "probabilities": { "ok": 0.0269, "defective": 0.9731 }
}
```

\---

## 3\. Problem and Dataset

### Why this problem and dataset

Casting inspection is a realistic computer-vision task with a clear deployment story (accept/reject a part). The dataset is a good fit for an engineering exercise for three reasons:

* **Non-trivial:** defects are small, localized surface anomalies on a visually uniform part, which is harder than generic object classification.
* **Moderately imbalanced:** about 60/40, so imbalance handling and error-type reasoning are both relevant.
* **Compute-friendly:** 1,300 images are enough for transfer learning but small enough to train each model in about a minute on a free Kaggle GPU.

Dataset: the raw (non-augmented) 512x512 version of the Kaggle *Real-life industrial dataset of casting product*. It is **not included** in this repository.

### Dataset inspection

|Property|Value|
|-|-|
|Images|1,300 JPEG, 512x512|
|Label format|Folder name: `ok\_front` = 0, `def\_front` = 1 (one label per image, no bounding boxes or masks)|
|Class distribution|781 defective (60.1%), 519 ok (39.9%)|
|Appearance|Grayscale-looking photos of the same part type from a top-down view; converted to 3-channel RGB for the pretrained networks|

Expected layout:

```
casting\_512x512/casting\_512x512/
├── def\_front/   # 781 images (label 1)
└── ok\_front/    # 519 images (label 0)
```

Visible variation between images (seen in the saved error plots): part position, tilt, scale, background brightness and cast shadows. Defects themselves are small relative to the part.

### Split

Stratified 70 / 15 / 15, `random\_state=42`. The test set is not used for training, early stopping or checkpoint selection.

|Split|Images|ok|defective|
|-|-|-|-|
|Train|910|363|547|
|Validation|195|78|117|
|Test|195|78|117|

\---

## 4\. Data Preparation, Models and Training

### Preprocessing and augmentation

All images are resized to **224x224** and normalized with ImageNet statistics. Training-only augmentation:

* Random horizontal and vertical flips (p = 0.5)
* Random rotation (±15°) and small affine shifts (translation up to 5%, scale 0.95–1.05) to mimic part misalignment
* Color jitter (brightness 0.2, contrast 0.2, saturation 0.1) for lighting variation

### Class imbalance

Inverse-frequency class weights, computed from the training labels, are passed to `CrossEntropyLoss`:

```
weight\_c = N\_total / (num\_classes \* N\_c)
weight\_ok = 1.2534,  weight\_defective = 0.8318
```

This balances the loss across the two classes. It does not by itself favor catching defects; that trade-off is handled by the decision threshold (see [Section 5](#improvement-hypotheses)).

### Label quality

No systematic label audit was performed. Labels come from the dataset's folder structure and were used as given. The few remaining misclassified images were inspected visually (Section 5); two of the four EfficientNet-B0 misses show no defect that is visible at 224x224, so a closer look at full resolution is listed as a follow-up.

### Models

|Model|Role|New head|Parameters|Checkpoint|
|-|-|-|-|-|
|MobileNetV3-Small|Baseline|Linear(576→1024) → Hardswish → Dropout(0.3) → Linear(1024→2)|1,519,906|\~6 MB|
|EfficientNet-B0|Selected|Dropout(0.3) → Linear(1280→2)|4,010,110|\~16 MB|

**Why transfer learning:** with 910 training images, training a CNN from scratch would be data-starved; ImageNet features give a strong starting point for surface texture and edges. Both models use torchvision ImageNet-1k weights. Note that the baseline is therefore a small pretrained network, not a from-scratch CNN.

**Why these two:** both are small enough to train on free resources and to serve on CPU. MobileNetV3-Small represents the lowest-latency end; EfficientNet-B0 is a stronger but still light model. Larger backbones were not tried, in line with the compute constraint.

### Training strategy (identical for both models)

1. **Phase 1, head only (epochs 1–4):** backbone frozen, AdamW, lr = 1e-3.
2. **Phase 2, fine-tuning (epoch 5 onward):** backbone unfrozen with discriminative learning rates (backbone 1e-5, head 1e-3); optimizer and scheduler rebuilt at the transition.

|Setting|Value|
|-|-|
|Epochs / batch size|12 / 64|
|Optimizer|AdamW, weight decay 1e-4|
|Scheduler|ReduceLROnPlateau (factor 0.1, patience 3) on val loss|
|Early stopping|patience 7 on val loss|
|Checkpoint|best validation loss|
|Loss|class-weighted cross-entropy|

`src/config.py` defaults to 25 epochs and batch size 32; the reported runs override these via CLI. The LR scheduler never triggered, so the head LR stayed at 1e-3.

### Hyperparameter tuning

No systematic hyperparameter search was run. Values were chosen in advance using standard fine-tuning practice (small backbone LR, larger head LR, unfreeze after a few head-only epochs) and kept identical across both models so the comparison isolates the architecture. A small sweep over learning rates, unfreeze epoch and epoch count is the first planned follow-up; each run takes about a minute.

### Compute awareness

* Trained on a free Kaggle notebook (NVIDIA Tesla T4): about 5–6 seconds per epoch, roughly one minute per model.
* 224x224 input and sub-5M-parameter models keep training and CPU inference cheap.
* The API runs on Render's free CPU tier. Inference latency has not been benchmarked.

\---

## 5\. Evaluation and Error Analysis

### Test-set results (195 images)

||MobileNetV3-Small|EfficientNet-B0|
|-|-|-|
|Accuracy|75.38%|97.95%|
|Macro F1|0.75|0.98|
|`ok` precision / recall / F1|0.62 / 0.99 / 0.76|0.95 / 1.00 / 0.97|
|`defective` precision / recall / F1|0.99 / 0.60 / 0.74|1.00 / 0.97 / 0.98|
|Confusion matrix (TN / FP / FN / TP)|77 / 1 / 47 / 70|78 / 0 / 4 / 113|
|Best epoch (val loss)|10|12|
|Best val loss / val accuracy|0.5127 / 72.82%|0.1458 / 94.87%|





### Error analysis

`src/evaluate.py` saves the misclassified test images with the model's confidence. Reviewed: all 4 EfficientNet-B0 errors, the 8 most confident of MobileNetV3-Small's 47 false negatives, and its single false positive. Images are shown at the 224x224 size the models receive.

**EfficientNet-B0: false negatives (4 of 117 defective parts)**



* The model's confidence in "ok" on the four missed parts is 78.4%, 69.7%, 54.1% and 53.1%, so P(defective) is roughly 0.22–0.47. All four are close to the decision boundary rather than confidently wrong.
* Two of the four show only a tiny dark mark on the rim (a small pit-like spot and a small notch) that covers just a few pixels at 224x224. The other two show no defect that is visible at this resolution.

**MobileNetV3-Small: false negatives (47 of 117) and false positive (1 of 78)**



* The eight most confident misses are all predicted "ok" with 98.5–99.7% confidence on parts that look normal at this scale. Errors this confident and uniform, together with `ok` recall of 0.99 and `defective` recall of 0.60, point to a systematic bias toward "ok" rather than to individually hard cases. This is consistent with the unstable validation curve noted below.
* The single false positive is borderline (50.8% defective) and shows a tilted view with a cast shadow.



**Reading the MobileNetV3-Small result.** Its training accuracy reached about 92% while validation accuracy stayed erratic (41–73%, below the 60% majority-class rate for the first five epochs) and validation loss did not decrease smoothly. A model that fits the training defects but behaves this way on validation suggests a training or normalization issue rather than a pure capacity limit. One untested hypothesis is a train/eval BatchNorm mismatch: the frozen backbone still runs BatchNorm in training mode, and MobileNetV3's low BN momentum means its running statistics barely adapt. Until this is tested, the baseline numbers describe this recipe as run, not the architecture's ceiling.

### Failure modes observed

1. **Tiny, localized defects.** Pit-like marks and rim notches occupy only a few pixels after downsampling 512 → 224.
2. **Low-margin decisions.** Every EfficientNet-B0 error and the MobileNetV3-Small false positive sit near P = 0.5.
3. **Nuisance variation.** Tilt, scale, background brightness and shadows vary across images, which makes subtle defects harder to separate from normal variation.

### Improvement hypotheses

|#|Hypothesis|How to test|
|-|-|-|
|1|A lower decision threshold recovers some missed defects at a modest false-alarm cost|Choose the threshold from the validation precision/recall curve (not the test set), then report test results once|
|2|Higher input resolution or a crop around the part helps with tiny defects|Retrain at 320x320 or 384x384 and compare defect recall|
|3|The MobileNetV3-Small gap is mostly a BatchNorm issue|Keep the frozen backbone in `eval()` mode during phase 1 and rerun|
|4|Some misses are label noise or very subtle defects|Review the remaining errors at full 512x512 resolution|

\---

## 6\. Inference and Deployment

The API (`api/app.py`) loads the selected model once at startup and reuses `get\_eval\_transforms` from `src/data.py`, so serving preprocessing is identical to evaluation preprocessing.

|Endpoint|Method|Description|
|-|-|-|
|`/`|GET|Health check: status, model name, device|
|`/predict`|POST|Multipart upload (field `file`, JPEG or PNG); returns `predicted\_class`, `confidence`, `probabilities`. Other content types return HTTP 400|

The model is chosen by the `MODEL\_NAME` environment variable (default `efficientnet\_b0`; `OUTPUT\_DIR` defaults to `outputs`).

**Docker** (`python:3.10-slim`, bundles `src/`, `api/` and the trained weights):

```bash
docker build -t casting-defect-detector .
docker run -p 8000:8000 -e MODEL\_NAME=efficientnet\_b0 casting-defect-detector
```

**Render:** `render.yaml` deploys the Docker image on the free plan.

\---

## 7\. Project Structure and How to Run

```
.
├── api/app.py                  # FastAPI service
├── src/
│   ├── config.py               # Dataclass configs: data, model, training
│   ├── data.py                 # Discovery, stratified split, augmentation, class weights, DataLoaders
│   ├── models.py               # Model factory, freeze/unfreeze, discriminative LR groups
│   ├── train.py                # Two-phase training, early stopping, checkpointing
│   ├── evaluate.py             # Test metrics, confusion matrix, curves, misclassification plots
│   └── inference.py            # Single-image inference (Python API + CLI)
├── outputs/
│   ├── efficientnet\_b0/        # weights, metrics JSON, plots, training history
│   └── mobilenetv3\_small/
├── classification-using-mobilenet-and-efficientnet.ipynb   # Kaggle notebook used to run training/evaluation
├── Dockerfile
├── render.yaml
└── requirements.txt            # minimum versions (>=), not exact pins
```

Data processing, training, evaluation and inference live in separate modules. Paths, model choice and hyperparameters are configurable through the dataclasses in `src/config.py` and CLI flags, and the pipeline uses Python `logging` with file and console handlers.

```bash
python -m venv venv \&\& source venv/bin/activate    # Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt

# Train (use --model mobilenetv3\_small for the baseline)
python -m src.train --model efficientnet\_b0 \\
    --data\_dir casting\_512x512/casting\_512x512 \\
    --epochs 12 --batch\_size 64 --lr 0.001 --backbone\_lr 0.00001 --unfreeze\_after 4

# Evaluate (metrics JSON, confusion matrix, curves, misclassification plots)
python -m src.evaluate --model efficientnet\_b0 --data\_dir casting\_512x512/casting\_512x512

# Single-image inference (the CLI default model is mobilenetv3\_small, so pass --model)
python -m src.inference --model efficientnet\_b0 --image path/to/sample.jpeg

# Run the API locally
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

\---

## 8\. Limitations

* **Small evaluation set.** One split, one seed, 195 test images. Approximate 95% Wilson intervals for EfficientNet-B0: accuracy about 95–99%, defective recall about 91–99% (113/117). The gap to MobileNetV3-Small is clearly larger than this noise, but small differences would not be.
* **No hyperparameter search**, and EfficientNet-B0's validation loss was still falling at epoch 12 (best checkpoint = final epoch), so training was capped rather than converged.
* **Baseline not tuned** and not a from-scratch CNN (see Section 5).
* **Resolution.** Downsampling 512 → 224 likely hides the smallest defects.
* **Narrow domain.** One part type and imaging setup; other parts, cameras or lighting are untested.
* **Default 0.5 threshold** and **no label audit.**

\---

## 9\. With More Compute

* **Training:** multiple seeds and k-fold cross-validation for mean ± std, a proper hyperparameter search, longer schedules, and higher input resolution (320–512) or tiled inference to keep small defects visible.
* **Models:** larger backbones (EfficientNet-B3/B4, ConvNeXt) and self-supervised or in-domain pretraining; ensembling and test-time augmentation.
* **Data:** more labeled defects, hard-negative mining from the near-boundary cases, and synthetic defect augmentation.
* **Task:** move from image-level labels to detection or segmentation (e.g. YOLOv8-seg, Mask R-CNN) to localize defects and measure severity.
* **Deployment:** export to ONNX Runtime or TensorRT with INT8 quantization for edge inference, and add latency benchmarking and monitoring.

