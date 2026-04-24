# ============================================================
# inference/predict.py
# ============================================================

TEMPERATURE = 0.6  # calibration

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import load_model, get_device
from utils.model_loader import ensure_model   # ✅ added


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

MODEL_PATH  = Path("saved_models/best_model.pth")

IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Broken stitch",
    "Needle mark",
    "Pinched fabric",
    "Vertical",
    "defect free",
    "hole",
    "horizontal",
    "lines",
    "stain"
]


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
# LOAD IMAGE
# ────────────────────────────────────────────────

def load_image(image_path: str):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path.resolve()}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    return image


# ────────────────────────────────────────────────
# PREPROCESS
# ────────────────────────────────────────────────

def preprocess_image(image: Image.Image):
    transform = get_inference_transform()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


# ────────────────────────────────────────────────
# PREDICT
# ────────────────────────────────────────────────

def predict(image_path: str, model=None, device=None):

    # 🔥 IMPORTANT — ensures model exists
    ensure_model()

    if device is None:
        device = get_device()

    if model is None:
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

    # Temperature scaling
    scaled_outputs = outputs / TEMPERATURE
    probabilities = torch.softmax(scaled_outputs, dim=1)
    probabilities = probabilities.squeeze(0).cpu().numpy()

    predicted_idx   = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probabilities[predicted_idx]) * 100

    all_scores = {
        CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, all_scores


# ────────────────────────────────────────────────
# DECISION LOGIC
# ────────────────────────────────────────────────

HIGH_CONFIDENCE    = 70.0
MEDIUM_CONFIDENCE  = 45.0


def get_decision(predicted_class: str, confidence: float):

    if confidence >= HIGH_CONFIDENCE:
        if predicted_class == "defect free":
            return "✅ PASS — Send to production", "GREEN"
        else:
            return f"❌ REJECT — Defect: {predicted_class}", "RED"

    elif confidence >= MEDIUM_CONFIDENCE:
        return "⚠️ LOW CONFIDENCE — Send for human review", "YELLOW"

    else:
        return "❓ UNCERTAIN — Manual inspection required", "YELLOW"


# ────────────────────────────────────────────────
# DISPLAY
# ────────────────────────────────────────────────

def display_result(image_path: str, predicted_class: str,
                   confidence: float, all_scores: dict):

    decision, status = get_decision(predicted_class, confidence)

    print("\n" + "=" * 50)
    print("       FABRIC DEFECT PREDICTION")
    print("=" * 50)
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {predicted_class.upper()}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  Decision   : {decision}")
    print("-" * 50)

    for cls, score in sorted(all_scores.items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(score // 5)
        print(f"  {cls:<20} : {score:>6.2f}%  {bar}")

    print("=" * 50)


# ────────────────────────────────────────────────
# ENTRY
# ────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python inference/predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    pred, conf, scores = predict(image_path)
    display_result(image_path, pred, conf, scores)