# Fabric Defect Detection in Textile Manufacturing using Computer Vision and Machine Learning
# 🏭 Fabric Defect Detection System — Team Playbook

This is an **AI-powered industrial system** to detect fabric defects using deep learning and computer vision.

This README is your **execution guide**. Follow it exactly.

---

# 🚨 IMPORTANT (READ FIRST)

Before doing anything:

* ❌ Do NOT push:

  * `data/`
  * `saved_models/`
  * `venv/`
* ❌ Do NOT change folder structure
* ❌ Do NOT rename classes
* ✅ Always pull latest changes first

---

# ⚙️ SETUP (EVERY TEAM MEMBER MUST DO THIS)

## 1. Clone repo

```bash id="n3t0qg"
git clone https://github.com/Ashleshcode/industrial-system-monitoring-ml.git
cd industrial-system-monitoring-ml
```

## 2. Create virtual environment

```bash id="d9c9cq"
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash id="sc9q0f"
pip install -r requirements.txt
```

---

# 📦 MODEL SETUP (MANDATORY)

Model is NOT in GitHub.

## Download from:

👉 [PASTE DRIVE LINK HERE]

## Place here:

```text id="wr07pq"
saved_models/best_model.pth
```

---

# 🚀 HOW TO RUN

## 🔹 Test single image

```bash id="hyx2cs"
python inference/predict.py "path_to_image.jpg"
```

## 🔹 Run batch testing

```bash id="dzb6rq"
python inference/batch_predict.py "folder_path"
```

## 🔹 Run UI

```bash id="rj0k6r"
streamlit run frontend/app.py
```

---

# 🧠 SYSTEM OVERVIEW

Pipeline:

```text id="7p38zx"
Image → Preprocessing → CNN Model → Prediction → UI Display
```

---

# 📁 PROJECT STRUCTURE

```text id="r8i9du"
data_preprocessing/   → transforms & loaders
models/               → CNN architecture
training/             → training & evaluation
inference/            → prediction scripts
frontend/             → UI (Streamlit)
utils/                → helpers
```

---

# 👥 TEAM EXECUTION PLAN

Each role has **clear instructions + prompts**

---

# 🧑‍💻 ROLE 1 — MODEL ENGINEER

## 🎯 Goal:

Improve model performance and robustness

## 🔧 Where to work:

```text id="f3mz4l"
models/
training/
data_preprocessing/
```

---

## ✅ Tasks:

1. Improve model:

   * Try EfficientNet / ConvNeXt
   * Tune dropout, LR, epochs

2. Improve preprocessing:

   * better augmentation
   * lighting robustness

3. Improve imbalance:

   * class weights
   * sampling

---

## 🧪 How to test:

```bash id="slk2bo"
python training/train.py
python training/evaluate.py
```

---

## ⚠️ Rules:

* ❌ Do NOT change class order
* ❌ Do NOT push `.pth`
* ❌ Do NOT break inference pipeline

---

## 💡 Prompt for you:

```text id="q4k47d"
Goal: Improve real-world accuracy, not just validation accuracy.

Focus on:
- confusion between similar classes
- robustness to lighting changes
- reducing false positives
```

---

# 🎨 ROLE 2 — FRONTEND ENGINEER

## 🎯 Goal:

Make UI simple enough for factory workers

---

## 🔧 Where to work:

```text id="hkt3og"
frontend/app.py
```

---

## ✅ Tasks:

1. Improve layout:

   * card-style UI
   * color coding (green/red/yellow)

2. Add features:

   * multi-image upload
   * better loading animations
   * clear error messages

3. Improve usability:

   * large text
   * simple outputs

---

## ⚠️ Rules:

* ❌ Do NOT change backend logic
* ❌ Do NOT break predict() function

---

## 💡 Prompt for you:

```text id="3q9f0y"
Imagine a factory worker using this.

Make it:
- fast
- obvious
- no confusion

If they need to think → UI is wrong
```

---

# 📷 ROLE 3 — COMPUTER VISION ENGINEER

## 🎯 Goal:

Make system work with real-time input

---

## 🔧 Where to work:

```text id="cldrzj"
new file: cv_integration.py
```

---

## ✅ Tasks:

1. Webcam integration:

   * capture frame
   * send to model
   * show result

2. OR folder monitoring:

   * auto-detect new images
   * process instantly

---

## 💡 Prompt:

```text id="qlx9y5"
Think like a factory line:

fabric moves → camera captures → system detects → result shown instantly
```

---

# 📊 ROLE 4 — DATA ENGINEER

## 🎯 Goal:

Improve dataset quality

---

## Tasks:

* find misclassified images
* balance dataset
* collect new samples
* clean noisy data

---

## 💡 Prompt:

```text id="8gfjlwm"
Better data > better model

Find:
- where model fails
- why it fails
- fix data accordingly
```

---

# 🧩 ROLE 5 — SYSTEM INTEGRATION

## 🎯 Goal:

Make everything work together

---

## Tasks:

* connect UI + model + CV
* optimize speed
* logging system

---

# 📊 TESTING PROTOCOL

When testing:

* fabric fills frame
* lighting is normal
* no extreme blur

---

# 🚧 CURRENT LIMITATIONS

* confidence may be low → expected
* similar defects may confuse
* dataset limited

---

# 🚀 ROADMAP

## Phase 1 (DONE)

* training
* evaluation
* inference

## Phase 2 (NOW)

* UI
* batch processing

## Phase 3

* real-time CV
* robustness

## Phase 4

* deployment

---

# 🧠 FINAL NOTE

This is NOT a college demo.

Build it like:

```text id="zz5c7m"
a system someone would actually pay for
```

Focus on:

* reliability
* clarity
* usability
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
