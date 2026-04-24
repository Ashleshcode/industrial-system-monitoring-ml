🏭 Fabric Defect Detection System

AI-powered industrial system for detecting fabric defects using Computer Vision and Deep Learning.

🚨 IMPORTANT (READ FIRST)
❌ Do NOT push:
data/
saved_models/
venv/
❌ Do NOT change folder structure
❌ Do NOT rename classes
✅ Always pull latest changes before working
⚙️ SETUP (ONE-TIME)
1. Clone repo
git clone https://github.com/Ashleshcode/industrial-system-monitoring-ml.git
cd industrial-system-monitoring-ml
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
🤖 MODEL SYSTEM (AUTO)

✅ No manual model download required

When you run prediction for the first time:

model is automatically downloaded from Google Drive
saved in saved_models/best_model.pth
🚀 HOW TO RUN
🔹 Single Image Prediction
python inference/predict.py "path_to_image.jpg"
🔹 Batch Prediction
python inference/batch_predict.py "folder_path"
🔹 Run UI
streamlit run frontend/app.py
🧠 SYSTEM OVERVIEW
Image → Preprocessing → CNN → Prediction → Decision Logic → UI
📁 PROJECT STRUCTURE
data_preprocessing/   → transforms
models/               → CNN model
training/             → training pipeline
inference/            → prediction scripts
frontend/             → Streamlit UI
utils/                → helpers (model loader, etc.)
👥 TEAM EXECUTION PLAN
🧑‍💻 MODEL ENGINEER
🎯 Goal:

Improve accuracy + robustness

🔧 Work in:
models/
training/
data_preprocessing/
✅ Tasks:
try EfficientNet / ConvNeXt
tune augmentation + LR
reduce class confusion
⚠️ Rules:
❌ do NOT change class order
❌ do NOT push .pth
❌ do NOT break inference
🎨 FRONTEND ENGINEER
🎯 Goal:

Make system usable for factory workers

🔧 Work in:
frontend/app.py
✅ Tasks:
improve layout (cards, spacing)
multi-image upload
clear status display
📷 CV ENGINEER
🎯 Goal:

Real-time detection

🔧 Work:
create: cv_integration.py
Tasks:
webcam capture
real-time prediction
display output
📊 DATA ENGINEER
🎯 Goal:

Improve dataset quality

Tasks:
find misclassified images
balance dataset
improve quality
🧩 SYSTEM INTEGRATION
🎯 Goal:

Connect everything

Tasks:
UI + model integration
optimize speed
logging
📊 TESTING GUIDELINES
fabric fills frame
proper lighting
no heavy blur
🚧 CURRENT LIMITATIONS
confidence may be low (intentional calibration)
similar defects may overlap
dataset limited
🚀 ROADMAP
Phase 1 ✅
training
evaluation
inference
Phase 2 🚧
UI
batch processing
Phase 3
real-time CV
robustness
Phase 4
deployment
# 🤖 LLM PROMPTS (USE THIS FOR YOUR TASKS)

Each role should use these prompts with ChatGPT / Claude / any LLM.

---

# 🧑‍💻 MODEL ENGINEER — PROMPT

Use this when improving model:

```
I am working on a fabric defect detection system using PyTorch.

Current setup:
- Model: ResNet (transfer learning)
- Input: 224x224 grayscale converted to 3-channel
- Classes: ['Broken stitch', 'Needle mark', 'Pinched fabric', 'Vertical', 'defect free', 'hole', 'horizontal', 'lines', 'stain']
- Dataset is imbalanced
- We already use augmentation and class weights
- Inference pipeline is already built and MUST NOT break

Your task:
- Suggest improvements to increase real-world robustness (not just validation accuracy)
- Focus on reducing confusion between visually similar defects
- Suggest improvements ONLY in:
  - model architecture
  - augmentation
  - training strategy

STRICT RULES:
- Do NOT change class order
- Do NOT modify inference function signature
- Do NOT suggest breaking folder structure
- Code must be directly usable in PyTorch

Output:
- Explain change briefly
- Then give FULL updated code block
```

---

# 🎨 FRONTEND ENGINEER — PROMPT

```
I am building a Streamlit UI for a fabric defect detection system.

Backend:
- predict(image_path) → returns (class, confidence, scores)

Your task:
- Improve UI/UX for non-technical factory workers
- Add:
  - multi-image upload
  - better layout (cards, sections)
  - visual status (OK / DEFECT / REVIEW)
  - loading indicators

STRICT RULES:
- Do NOT modify backend logic
- Do NOT change predict() function
- Keep everything inside Streamlit

Goal:
Make UI so simple that a factory worker can use it without training

Output:
- Give FULL updated app.py code
```

---

# 📷 CV ENGINEER — PROMPT

```
I want to integrate real-time image capture into a fabric defect detection system.

Current system:
- predict(image_path) works
- Model already trained
- Need real-time input

Your task:
- Implement webcam-based detection using OpenCV
- Capture frame → send to model → display result

STRICT RULES:
- Do NOT modify existing model code
- Keep inference logic same
- Write separate file (cv_integration.py)

Output:
- Complete working code
- Minimal dependencies
```

---

# 📊 DATA ENGINEER — PROMPT

```
I am working on a fabric defect dataset for classification.

Problems:
- Class imbalance
- Some classes visually similar
- Model struggles on real-world images

Your task:
- Suggest data improvements:
  - augmentation strategies
  - balancing techniques
  - dataset cleaning methods

STRICT RULES:
- Do NOT suggest changing labels
- Do NOT suggest unrealistic data collection

Output:
- Actionable steps (not theory)
```

---

# 🧩 SYSTEM INTEGRATION — PROMPT

```
I am integrating multiple components of a fabric defect detection system:

- PyTorch model
- Streamlit UI
- Batch processing
- (Future) webcam input

Your task:
- Suggest how to structure system cleanly
- Improve performance and response time
- Add logging system

STRICT RULES:
- Do NOT break existing modules
- Keep modular structure

Output:
- Practical architecture suggestions
```

---

# ⚠️ FINAL RULE

Always include this line in your prompt:

```
"Do NOT break existing pipeline or file structure"
```

This prevents bad AI suggestions.
