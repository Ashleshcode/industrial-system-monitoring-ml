const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const predictButton = document.getElementById("predictButton");
const resultSection = document.getElementById("resultSection");
const predictedClass = document.getElementById("predictedClass");
const confidenceValue = document.getElementById("confidenceValue");
const statusBadge = document.getElementById("statusBadge");
const message = document.getElementById("message");
const scoresWrapper = document.getElementById("scoresWrapper");
const scoresList = document.getElementById("scoresList");

let selectedFile = null;
let currentObjectURL = null;


// ────────────────────────────────────────────────
// IMAGE SELECTION + PREVIEW
// ────────────────────────────────────────────────

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];

  if (!file) {
    clearPreview();
    return;
  }

  // 🔥 Validate file type
  if (!file.type.startsWith("image/")) {
    setMessage("Please upload a valid image file.");
    clearPreview();
    return;
  }

  // 🔥 Cleanup old preview URL
  if (currentObjectURL) {
    URL.revokeObjectURL(currentObjectURL);
  }

  selectedFile = file;
  currentObjectURL = URL.createObjectURL(file);

  imagePreview.src = currentObjectURL;
  imagePreview.classList.remove("hidden");
  previewPlaceholder.classList.add("hidden");

  setMessage("");
});


// ────────────────────────────────────────────────
// PREDICT BUTTON
// ────────────────────────────────────────────────

predictButton.addEventListener("click", async () => {
  if (!selectedFile) {
    setMessage("Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  setLoadingState(true);
  setMessage("Predicting...");

  try {
    // 🔥 Timeout-safe request
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();

    renderResult(data);
    setMessage(`Prediction: ${data.predicted_class} (${data.status})`);

  } catch (error) {
    if (error.name === "AbortError") {
      setMessage("Request timed out. Try again.");
    } else {
      setMessage("Prediction failed. Check backend.");
    }
    console.error(error);
  } finally {
    setLoadingState(false);
  }
});


// ────────────────────────────────────────────────
// UI HELPERS
// ────────────────────────────────────────────────

function setLoadingState(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.textContent = isLoading ? "Predicting..." : "Predict";
}

function setMessage(text) {
  message.textContent = text;
}

function clearPreview() {
  selectedFile = null;
  imageInput.value = "";

  if (currentObjectURL) {
    URL.revokeObjectURL(currentObjectURL);
    currentObjectURL = null;
  }

  imagePreview.src = "";
  imagePreview.classList.add("hidden");
  previewPlaceholder.classList.remove("hidden");
}


// ────────────────────────────────────────────────
// RESULT RENDERING
// ────────────────────────────────────────────────

function renderResult(data) {

  predictedClass.textContent = data.predicted_class ?? "-";

  confidenceValue.textContent =
    typeof data.confidence === "number"
      ? `${data.confidence.toFixed(1)}%`
      : "-";

  const status = data.status ?? "UNKNOWN";

  statusBadge.textContent = status;
  statusBadge.className = "status-badge";
  statusBadge.classList.add(getStatusClass(status));

 // renderScores(data.all_scores);

  // 🔥 Reset animation before re-trigger
  resultSection.classList.remove("show");
  resultSection.classList.remove("hidden");

  requestAnimationFrame(() => {
    resultSection.classList.add("show");
  });
}


// ────────────────────────────────────────────────
// STATUS STYLING
// ────────────────────────────────────────────────

function getStatusClass(status) {
  const normalizedStatus = String(status).toUpperCase();

  if (normalizedStatus === "CONFIDENT") return "status-confident";
  if (normalizedStatus === "REVIEW") return "status-review";
  return "status-uncertain";
}


// ────────────────────────────────────────────────
// SCORE DISPLAY
// ────────────────────────────────────────────────

function renderScores(allScores) {

  scoresList.innerHTML = "";

  if (!allScores || typeof allScores !== "object") {
    scoresWrapper.classList.add("hidden");
    return;
  }

  const entries = Object.entries(allScores)
    .sort((a, b) => b[1] - a[1]); // 🔥 sorted descending

  if (entries.length === 0) {
    scoresWrapper.classList.add("hidden");
    return;
  }

  for (const [label, score] of entries) {
    const row = document.createElement("div");
    row.className = "score-row";

    const labelSpan = document.createElement("span");
    labelSpan.textContent = label;

    const scoreSpan = document.createElement("span");
    scoreSpan.textContent =
      typeof score === "number"
        ? `${score.toFixed(1)}%`
        : String(score);

    row.appendChild(labelSpan);
    row.appendChild(scoreSpan);
    scoresList.appendChild(row);
  }

  scoresWrapper.classList.remove("hidden");
}