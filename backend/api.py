# ============================================================
# backend/api.py
# ============================================================

import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import shutil
import uuid

from cv_pipeline.inference_service import FabricDefectDetector


# ────────────────────────────────────────────────
# INIT
# ────────────────────────────────────────────────

app = FastAPI()
detector = FabricDefectDetector()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    result = detector.predict(str(file_path))

    return result