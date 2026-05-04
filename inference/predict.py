# ============================================================
# inference/predict.py
# ============================================================

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import load_model, get_device

MODEL_PATH    = Path("saved_models/best_model.pth")
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Temperature — 1.0 means no scaling (raw softmax)
# Lower than 1.0 sharpens predictions
# Higher than 1.0 softens predictions
TEMPERATURE = 0.7

CLASS_NAMES = [
    "Broken stitch",
   
    "defect free",
    "hole",
    "stain"
]

# Confidence thresholds for factory decision logic
HIGH_CONFIDENCE   = 70.0
MEDIUM_CONFIDENCE = 45.0


# ────────────────────────────────────────────────
# TRANSFORM
# ────────────────────────────────────────────────

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ────────────────────────────────────────────────
# IMAGE LOADING
# ────────────────────────────────────────────────

def load_image(image_path: str):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found at: {image_path.resolve()}"
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported format: {image_path.suffix}\n"
            f"Supported: {valid_extensions}"
        )

    return Image.open(image_path).convert("RGB")


def preprocess_image(image: Image.Image):
    transform = get_inference_transform()
    return transform(image).unsqueeze(0)


# ────────────────────────────────────────────────
# PREDICT
# ────────────────────────────────────────────────

def predict(image_path: str, model=None, device=None):
    """
    Predicts defect class for a single image.
    Returns predicted class, confidence (0-100), all scores.
    """
    if device is None:
        device = get_device()

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model at: {MODEL_PATH}\n"
                "Run training/train.py first."
            )
        model = load_model(
            num_classes=len(CLASS_NAMES),
            path=str(MODEL_PATH),
            device=device
        )

    image  = load_image(image_path)
    tensor = preprocess_image(image).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

    # Temperature scaling then softmax
    # outputs shape: [1, 5]
    scaled        = outputs / TEMPERATURE
    probabilities = torch.softmax(scaled, dim=1)
    probabilities = probabilities.squeeze(0).cpu().numpy()  # shape: [5]

    predicted_idx   = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]

    # Multiply by 100 ONCE here — this was the bug causing 9954%
    confidence = float(probabilities[predicted_idx]) * 100

    all_scores = {
        CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, all_scores


# ────────────────────────────────────────────────
# DECISION LOGIC
# ────────────────────────────────────────────────

def get_decision(predicted_class: str, confidence: float) -> str:
    """
    Converts prediction into factory floor decision.
    """
    if confidence >= HIGH_CONFIDENCE:
        if predicted_class == "defect free":
            return "PASS — Send to production"
        else:
            return f"REJECT — Defect: {predicted_class}"
    elif confidence >= MEDIUM_CONFIDENCE:
        return "LOW CONFIDENCE — Send for human review"
    else:
        return "UNCERTAIN — Manual inspection required"


# ────────────────────────────────────────────────
# DISPLAY
# ────────────────────────────────────────────────

def display_result(image_path: str, predicted_class: str,
                   confidence: float, all_scores: dict):

    decision = get_decision(predicted_class, confidence)

    print("\n" + "=" * 50)
    print("       FABRIC DEFECT PREDICTION")
    print("=" * 50)
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {predicted_class.upper()}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  Decision   : {decision}")
    print("-" * 50)
    print("  All class scores:")
    for cls, score in sorted(
        all_scores.items(), key=lambda x: x[1], reverse=True
    ):
        bar = "|" * int(score // 5)
        print(f"  {cls:<20} : {score:>6.2f}%  {bar}")
    print("=" * 50)


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference/predict.py <path_to_image>")
        print("Example: python inference/predict.py data/test_images/stain/img1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predicted_class, confidence, all_scores = predict(image_path)
    display_result(image_path, predicted_class, confidence, all_scores)