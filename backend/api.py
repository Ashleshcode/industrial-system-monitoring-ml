# ============================================================
# backend/api.py
# FINAL — SINGLE + BATCH + HISTORY API
# ============================================================

import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shutil
import uuid
import time
import json
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cv_pipeline.inference_service import FabricDefectDetector


# ────────────────────────────────────────────────
# INIT
# ────────────────────────────────────────────────

app = FastAPI()

detector = FabricDefectDetector()

# IMPORTANT:
# Replace this with your CURRENT ngrok URL
BASE_URL = "https://mama-rasping-autopilot.ngrok-free.dev"


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

# History directory
HISTORY_DIR = Path("inspection_history")
HISTORY_DIR.mkdir(exist_ok=True)

HISTORY_FILE = HISTORY_DIR / "history.json"

# Create history file if missing
if not HISTORY_FILE.exists():

    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)


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
# HISTORY HELPERS
# ────────────────────────────────────────────────

def save_inspection(record):

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    # newest first
    history.insert(0, record)

    # keep only latest 100
    history = history[:100]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ────────────────────────────────────────────────
# GET HISTORY
# ────────────────────────────────────────────────

@app.get("/history")
async def get_history():

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    return {
        "history": history
    }


# ────────────────────────────────────────────────
# CLEAR HISTORY
# ────────────────────────────────────────────────

@app.delete("/history")
async def clear_history():

    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

    return {
        "message": "Inspection history cleared"
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

    # Frontend heatmap URL
    heatmap_path = result.get("heatmap_path")

    if heatmap_path:

        heatmap_filename = Path(heatmap_path).name

        result["heatmap_url"] = (
            f"{BASE_URL}/outputs/{heatmap_filename}"
        )

    # Extra metadata
    result["inspection_id"] = file_id[:8]

    result["processing_time_sec"] = round(
        time.time() - start_time,
        2
    )

    # Save history
    save_inspection({

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "type":
            "single",

        "filename":
            file.filename,

        "prediction":
            result["predicted_class"],

        "confidence":
            result["confidence"],

        "status":
            result["status"],

        "heatmap_url":
            result.get("heatmap_url"),

        "inspection_id":
            result["inspection_id"]
    })

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

            # Preserve extension
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
                    f"{BASE_URL}/outputs/"
                    f"{heatmap_filename}"
                )

            # Metadata
            result["filename"] = file.filename

            result["inspection_id"] = file_id[:8]

            result["success"] = True

            # Save history
            save_inspection({

                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "type":
                    "batch",

                "filename":
                    file.filename,

                "prediction":
                    result["predicted_class"],

                "confidence":
                    result["confidence"],

                "status":
                    result["status"],

                "heatmap_url":
                    result.get("heatmap_url"),

                "inspection_id":
                    result["inspection_id"]
            })

            successful += 1

            results.append(result)

        except Exception as e:

            failed += 1

            results.append({

                "filename":
                    file.filename,

                "success":
                    False,

                "error":
                    str(e)
            })

    total_time = round(
        time.time() - batch_start,
        2
    )

    return {

        "batch_summary": {

            "total_images":
                len(files),

            "successful":
                successful,

            "failed":
                failed,

            "processing_time_sec":
                total_time
        },

        "results":
            results
    }