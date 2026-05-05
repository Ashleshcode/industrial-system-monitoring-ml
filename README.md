
# 🧵 Fabric Defect Detection System

AI-powered industrial system for detecting fabric defects using Computer Vision and Deep Learning.

---

# 🚨 IMPORTANT (READ FIRST)

## ❌ DO NOT PUSH

```
data/
saved_models/
venv/
```

## ❌ DO NOT MODIFY

* Folder structure
* Class names / order
* Core inference pipeline

## ✅ ALWAYS

* Pull latest changes before working
* Test before pushing

---

# ⚙️ SETUP (ONE-TIME)

## 1️⃣ Clone Repository

```
git clone https://github.com/Ashleshcode/industrial-system-monitoring-ml.git
cd industrial-system-monitoring-ml
```

## 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 📥 MODEL DOWNLOAD (AUTOMATIC)

The trained model is **NOT stored in this repository** due to size constraints.

### 🔗 Google Drive Model Link

```
"https://drive.google.com/file/d/1Q68-NJlNvlQmp_UIaBUhkeLT_h52a3K2/view?usp=sharing" (just in case it does not auto download )
```

---

## ⚙️ How it Works

* On first run:

  * If model file is missing:

```
saved_models/best_model.pth
```

* It is automatically downloaded from Google Drive

* After download:

  * Stored locally inside `saved_models/`
  * Reused for all future runs

---

## ✅ You DO NOT need to:

* manually download model
* place `.pth` file manually

---

## ⚠️ Notes

* Internet required only for first run
* Do NOT delete `saved_models/` after download
* Do NOT push `.pth` file to GitHub

---

## 🔁 If Model is Missing

Just run:

```
uvicorn backend.api:app --reload
```

Model will download automatically.

---

# 🧠 SYSTEM OVERVIEW

```
Image → Frontend → Backend (FastAPI) → CV Pipeline → Model → Decision → UI
```

---

# 🚀 HOW TO RUN

## 🔹 Start Backend

```
uvicorn backend.api:app --reload
```

---

## 🔹 Open Frontend

```
start frontend\index.html
```

---

## 🔹 Use Application

1. Upload image
2. Click Predict
3. View result

---

# 📂 PROJECT STRUCTURE

```
backend/                → FastAPI API server
cv_pipeline/            → Core inference logic
frontend/               → HTML/CSS/JS UI
models/                 → Model architecture
training/               → Training pipeline
data_preprocessing/     → Data transforms
saved_models/           → Model storage (auto-downloaded)
```

---

# 🎯 CURRENT SYSTEM STATUS

```
✔ Model trained and stabilized
✔ Dataset cleaned and optimized
✔ CV pipeline implemented
✔ Backend API working
✔ Frontend UI complete
✔ End-to-end system working

```

---

# ⚠️ TEAM INSTRUCTIONS (CRITICAL)

## 🔴 CORE SYSTEM — DO NOT TOUCH

```
cv_pipeline/
models/
training/
data_preprocessing/
```

🚫 Any changes here can break:

* model loading
* preprocessing
* predictions

---

## 🟡 FRONTEND TEAM

You may modify ONLY:

```
frontend/index.html
frontend/style.css
```

### ❌ DO NOT:

* Change element IDs
* Modify `script.js` logic
* Change API URL

### ✅ YOU CAN:

* Improve UI/UX
* Add animations
* Improve layout

---

## 🟡 BACKEND TEAM

You may modify ONLY:

```
backend/api.py
```

### ❌ DO NOT:

* Change `/predict` endpoint
* Change response format
* Modify model loading

### ✅ YOU CAN:

* Add logging
* Improve performance
* Add optional endpoints (without breaking `/predict`)

---

# 📡 API CONTRACT (DO NOT CHANGE)

### POST `/predict`

**Request:**

* form-data → file

**Response:**

```json
{
  "predicted_class": "stain",
  "confidence": 92.3,
  "status": "CONFIDENT"
}
```

---

# 🧪 TESTING CHECKLIST

Before demo, verify:

* Backend running
* Frontend loads
* Image preview works
* Prediction works
* Status colors correct
* No console errors

---

# 🚧 CURRENT LIMITATIONS

* Similar defects may overlap
* Dataset size limited
* Confidence may vary on real-world samples

---

# 🚀 FUTURE IMPROVEMENTS

* Improved dataset quality
* Real-time webcam detection
* Edge deployment (Jetson / Raspberry Pi)
* Better confidence calibration

---

# 🏁 FINAL NOTE

This system prioritizes:

* Stability
* Simplicity
* Real-world usability

NOT overengineering.

---

## 👨‍💻 Author

Core ML, CV, and System Integration:

* Aryan Mangalore

---

## 🚀 STATUS

```
✔ Fully working end-to-end system
✔ Ready for demo
✔ Ready for further improvements
```
