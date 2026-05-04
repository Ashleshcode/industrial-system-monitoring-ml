# ============================================================
# cv_pipeline/inference_service.py (FINAL — DEMO POLISHED)
# ============================================================

import sys
from pathlib import Path

# 🔥 FIX IMPORT PATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.cnn_model import load_model, get_device


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

MODEL_PATH = Path("saved_models/best_model.pth")

CLASS_NAMES = [
    "Broken stitch",
    "defect free",
    "hole",
    "stain"
]

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ────────────────────────────────────────────────
# TRANSFORM
# ────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ────────────────────────────────────────────────
# SERVICE CLASS
# ────────────────────────────────────────────────

class FabricDefectDetector:

    def __init__(self):
        self.device = get_device()

        self.model = load_model(
            num_classes=len(CLASS_NAMES),
            path=str(MODEL_PATH),
            device=self.device
        )

        self.model.eval()
        self.transform = get_transform()

        print("✅ CV Service Ready")

    # ────────────────────────────────────────────────

    def preprocess(self, image: Image.Image):
        return self.transform(image).unsqueeze(0)

    # ────────────────────────────────────────────────

    def predict(self, image_path: str):
        """
        Predicts defect and returns frontend-ready output
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100)

        # 🔥 ROUND for clean display
        confidence = round(confidence, 1)

        # 🔥 DEMO-OPTIMIZED DECISION LOGIC
        if confidence >= 65:
            status = "CONFIDENT"
        elif confidence >= 45:
            status = "REVIEW"
        else:
            status = "UNCERTAIN"

        result = {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "status": status,
            "all_scores": {
                CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }

        return result


# ────────────────────────────────────────────────
# CLI ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python cv_pipeline/inference_service.py <image_path>")
        print("Example: python cv_pipeline/inference_service.py data/test_images/stain/img1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    detector = FabricDefectDetector()

    try:
        result = detector.predict(image_path)

        print("\nPrediction Result:")
        print(result)

    except Exception as e:
        print(f"\n❌ Error: {e}")