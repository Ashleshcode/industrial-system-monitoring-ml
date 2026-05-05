import os
import gdown

MODEL_URL = "https://drive.google.com/file/d/1Q68-NJlNvlQmp_UIaBUhkeLT_h52a3K2/view?usp=sharing"
MODEL_PATH = "saved_models/best_model.pth"


def download_model():
    os.makedirs("saved_models", exist_ok=True)

    print("⬇️ Downloading model (via gdown)...")

    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    print("✅ Model downloaded successfully")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        download_model()