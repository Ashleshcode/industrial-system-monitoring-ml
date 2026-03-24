# ============================================================
# cv_pipeline/inference_service.py
# FINAL — DEMO POLISHED + GRAD-CAM
# ============================================================

import sys
from pathlib import Path

# 🔥 FIX IMPORT PATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from cv_pipeline.gradcam import GradCAM, overlay_heatmap
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

        # 🔥 Grad-CAM setup
        self.gradcam = GradCAM(
            model=self.model,
            target_layer=self.model.features[-1]
        )

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

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        tensor = self.preprocess(image).to(self.device)

        # Forward pass (NO torch.no_grad for Grad-CAM)
        outputs = self.model(tensor)

        # Probabilities
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))

        confidence = float(probs[pred_idx] * 100)

        # Clean display
        confidence = round(confidence, 1)

        # ────────────────────────────────────────────────
        # 🔥 GRAD-CAM
        # ────────────────────────────────────────────────

        cam = self.gradcam.generate(tensor, pred_idx)

        overlay = overlay_heatmap(image, cam)

        # Save Grad-CAM output
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        heatmap_path = output_dir / f"{image_path.stem}_gradcam.jpg"

        cv2.imwrite(
            str(heatmap_path),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        # ────────────────────────────────────────────────
        # DECISION LOGIC
        # ────────────────────────────────────────────────

        if confidence >= 65:
            status = "CONFIDENT"

        elif confidence >= 45:
            status = "REVIEW"

        else:
            status = "UNCERTAIN"

        # ────────────────────────────────────────────────
        # FINAL RESULT
        # ────────────────────────────────────────────────

        result = {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "status": status,
            "heatmap_path": str(heatmap_path),

            # Optional debug info
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

        print("Usage:")
        print("python cv_pipeline/inference_service.py <image_path>")

        print("\nExample:")
        print(
            "python cv_pipeline/inference_service.py "
            "data/test_images/stain/img1.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    detector = FabricDefectDetector()

    try:

        result = detector.predict(image_path)

        print("\nPrediction Result:")
        print(result)

    except Exception as e:

        print(f"\n❌ Error: {e}")