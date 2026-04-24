# ============================================================
# inference/predict.py
#
# Loads the trained model and predicts the defect class
# for a single input fabric image.
# This file is also used by the Streamlit frontend (Phase 7).
# ============================================================
# Add this constant at the top of predict.py
# Temperature > 1 makes predictions softer/more uncertain
# Temperature < 1 makes predictions sharper
# We use it to fix overconfidence without touching the model
TEMPERATURE = 0.6  # tune this between 1.2 and 2.0
import sys
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import transforms

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import load_model, get_device


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

MODEL_PATH  = Path("saved_models/best_model.pth")

# Must match exactly what was used in preprocessing
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Must match the exact folder names from your dataset
# Order matters — must match how ImageFolder sorted them
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
# IMAGE TRANSFORM
# ────────────────────────────────────────────────

def get_inference_transform():
    """
    Same as validation transform — no augmentation.
    Just resize, convert to tensor, and normalize.
    """
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
    """
    Loads and validates a single image from disk.
    Converts to RGB to handle any image format safely.

    Args:
        image_path : path to the image file (str or Path)

    Returns:
        PIL Image object
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found at: {image_path.resolve()}"
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported file format: {image_path.suffix}\n"
            f"Supported formats: {valid_extensions}"
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    return image


# ────────────────────────────────────────────────
# PREPROCESS IMAGE
# ────────────────────────────────────────────────

def preprocess_image(image: Image.Image):
    """
    Applies transforms and adds batch dimension.

    A model always expects input as a batch even if
    we're only predicting one image. So we add a
    dimension at position 0: [C, H, W] → [1, C, H, W]

    Args:
        image : PIL Image

    Returns:
        tensor of shape [1, 3, 224, 224]
    """
    transform = get_inference_transform()
    tensor    = transform(image)        # shape: [3, 224, 224]
    tensor    = tensor.unsqueeze(0)     # shape: [1, 3, 224, 224]
    return tensor


# ────────────────────────────────────────────────
# PREDICT
# ────────────────────────────────────────────────

def predict(image_path: str, model=None, device=None):
    if device is None:
        device = get_device()

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model found at: {MODEL_PATH}\n"
                "Please run training/train.py first."
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
        outputs = model(tensor)  # raw logits shape: [1, 9]

    # ── Temperature scaling ──────────────────────────────
    # Divide logits by temperature BEFORE softmax
    # Higher temp = softer probabilities = less overconfident
    # This is mathematically correct calibration
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
# DISPLAY RESULT
# ────────────────────────────────────────────────
# Confidence thresholds — product-grade decision logic
HIGH_CONFIDENCE    = 70.0   # trust the prediction
MEDIUM_CONFIDENCE  = 45.0   # show but flag for review
# below 45% = uncertain, flag for human inspection

def get_decision(predicted_class: str, confidence: float) -> tuple:
    """
    Converts raw model output into a factory-floor decision.
    
    Returns:
        decision : what action to take
        status   : traffic light color for UI
    """
    if confidence >= HIGH_CONFIDENCE:
        if predicted_class == "defect free":
            return "✅ PASS — Send to production", "GREEN"
        else:
            return f"❌ REJECT — Defect: {predicted_class}", "RED"

    elif confidence >= MEDIUM_CONFIDENCE:
        return "⚠️  LOW CONFIDENCE — Send for human review", "YELLOW"

    else:
        return "❓ UNCERTAIN — Manual inspection required", "YELLOW"


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
    print("  All class scores:")
    for cls, score in sorted(all_scores.items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(score // 5)
        print(f"  {cls:<20} : {score:>6.2f}%  {bar}")
    print("=" * 50)

# ────────────────────────────────────────────────
# ENTRY POINT — test with a real image
# ────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    # Pass image path as command line argument
    # Example: python inference/predict.py data/raw/Dataset/stain/image1.jpg
    if len(sys.argv) < 2:
        print("Usage: python inference/predict.py <path_to_image>")
        print("Example: python inference/predict.py data/raw/Dataset/stain/stain_1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    predicted_class, confidence, all_scores = predict(image_path)
    display_result(image_path, predicted_class, confidence, all_scores)