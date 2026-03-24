# ============================================================
# backend/api.py
# FINAL — SINGLE + BATCH INSPECTION API
# ============================================================

import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shutil
import uuid
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cv_pipeline.inference_service import FabricDefectDetector


# ────────────────────────────────────────────────
# INIT
# ────────────────────────────────────────────────

app = FastAPI()

detector = FabricDefectDetector()


# ────────────────────────────────────────────────
# CORS
# ────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ────────────────────────────────────────────────
# DIRECTORIES
# ────────────────────────────────────────────────

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────
# STATIC FILES
# ────────────────────────────────────────────────

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# ────────────────────────────────────────────────
# ROOT
# ────────────────────────────────────────────────

@app.get("/")
async def root():

    return {
        "message": "Fabric Defect Detection API Running"
    }


# ────────────────────────────────────────────────
# SINGLE IMAGE PREDICTION
# ────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.time()

    # Unique filename
    file_id = str(uuid.uuid4())

    file_path = UPLOAD_DIR / f"{file_id}.jpg"

    # Save upload
    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    result = detector.predict(str(file_path))

    # Frontend-accessible heatmap URL
    heatmap_path = result.get("heatmap_path")

    if heatmap_path:

        heatmap_filename = Path(heatmap_path).name

        result["heatmap_url"] = (
            f"http://127.0.0.1:8000/outputs/{heatmap_filename}"
        )

    # Extra metadata
    result["inspection_id"] = file_id[:8]

    result["processing_time_sec"] = round(
        time.time() - start_time,
        2
    )

    return result


# ────────────────────────────────────────────────
# BATCH INSPECTION
# ────────────────────────────────────────────────

@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...)
):

    batch_start = time.time()

    results = []

    successful = 0
    failed = 0

    for file in files:

        try:

            # Unique ID
            file_id = str(uuid.uuid4())

            # Preserve extension if possible
            extension = Path(file.filename).suffix

            if extension == "":
                extension = ".jpg"

            file_path = (
                UPLOAD_DIR /
                f"{file_id}{extension}"
            )

            # Save upload
            with open(file_path, "wb") as buffer:

                shutil.copyfileobj(file.file, buffer)

            # Run inference
            result = detector.predict(str(file_path))

            # Heatmap URL
            heatmap_path = result.get("heatmap_path")

            if heatmap_path:

                heatmap_filename = Path(
                    heatmap_path
                ).name

                result["heatmap_url"] = (
                    f"http://127.0.0.1:8000/outputs/"
                    f"{heatmap_filename}"
                )

            # Metadata
            result["filename"] = file.filename

            result["inspection_id"] = file_id[:8]

            result["success"] = True

            successful += 1

            results.append(result)

        except Exception as e:

            failed += 1

            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    total_time = round(
        time.time() - batch_start,
        2
    )

    return {

        "batch_summary": {

            "total_images": len(files),

            "successful": successful,

            "failed": failed,

            "processing_time_sec": total_time
        },

        "results": results
    }