const API_URL =
  "https://mama-rasping-autopilot.ngrok-free.dev/predict";

const BATCH_API_URL =
  "https://mama-rasping-autopilot.ngrok-free.dev/predict-batch";

const HISTORY_API_URL =
  "https://mama-rasping-autopilot.ngrok-free.dev/history";
  const NGROK_HEADERS = {
  "ngrok-skip-browser-warning": "true"
};

// ────────────────────────────────────────────────
// ELEMENTS
// ────────────────────────────────────────────────

const imageInput =
  document.getElementById("imageInput");

const imagePreview =
  document.getElementById("imagePreview");

const previewPlaceholder =
  document.getElementById("previewPlaceholder");

const predictButton =
  document.getElementById("predictButton");

const resultSection =
  document.getElementById("resultSection");

const predictedClass =
  document.getElementById("predictedClass");

const confidenceValue =
  document.getElementById("confidenceValue");

const statusBadge =
  document.getElementById("statusBadge");

const message =
  document.getElementById("message");

const cameraFeed =
  document.getElementById("cameraFeed");

const startCameraButton =
  document.getElementById("startCameraButton");

const captureButton =
  document.getElementById("captureButton");


// ────────────────────────────────────────────────
// HEATMAP ELEMENTS
// ────────────────────────────────────────────────

const heatmapSection =
  document.getElementById("heatmapSection");

const originalResultImage =
  document.getElementById("originalResultImage");

const heatmapImage =
  document.getElementById("heatmapImage");


// ────────────────────────────────────────────────
// BATCH ELEMENTS
// ────────────────────────────────────────────────

const batchImageInput =
  document.getElementById("batchImageInput");

const batchPredictButton =
  document.getElementById("batchPredictButton");

const batchSummary =
  document.getElementById("batchSummary");

const batchResults =
  document.getElementById("batchResults");


// ────────────────────────────────────────────────
// HISTORY ELEMENTS
// ────────────────────────────────────────────────

const historyContainer =
  document.getElementById("historyContainer");

const refreshHistoryButton =
  document.getElementById("refreshHistoryButton");

const clearHistoryButton =
  document.getElementById("clearHistoryButton");


// ────────────────────────────────────────────────
// GLOBAL STATE
// ────────────────────────────────────────────────

let selectedFile = null;

let currentObjectURL = null;

let cameraStream = null;


// ────────────────────────────────────────────────
// IMAGE UPLOAD
// ────────────────────────────────────────────────

imageInput.addEventListener(
  "change",
  (event) => {

    const file = event.target.files[0];

    if (!file) {

      clearPreview();

      return;
    }

    if (!file.type.startsWith("image/")) {

      setMessage(
        "Please upload a valid image."
      );

      clearPreview();

      return;
    }

    if (currentObjectURL) {

      URL.revokeObjectURL(
        currentObjectURL
      );
    }

    selectedFile = file;

    currentObjectURL =
      URL.createObjectURL(file);

    imagePreview.src =
      currentObjectURL;

    imagePreview.classList.remove(
      "hidden"
    );

    previewPlaceholder.classList.add(
      "hidden"
    );

    cameraFeed.classList.add(
      "hidden"
    );

    stopCamera();

    setMessage("");

    resultSection.classList.add(
      "hidden"
    );

    heatmapSection.classList.add(
      "hidden"
    );
  }
);


// ────────────────────────────────────────────────
// START CAMERA
// ────────────────────────────────────────────────

startCameraButton.addEventListener(
  "click",
  async () => {

    try {

      stopCamera();

      cameraStream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment"
          }
        });

      cameraFeed.srcObject =
        cameraStream;

      cameraFeed.classList.remove(
        "hidden"
      );

      imagePreview.classList.add(
        "hidden"
      );

      previewPlaceholder.classList.add(
        "hidden"
      );

      captureButton.classList.remove(
        "hidden"
      );

      setMessage("Camera ready.");

    } catch (error) {

      console.error(error);

      setMessage(
        "Unable to access camera."
      );
    }
  }
);


// ────────────────────────────────────────────────
// CAPTURE PHOTO
// ────────────────────────────────────────────────

captureButton.addEventListener(
  "click",
  () => {

    if (
      !cameraFeed.videoWidth ||
      !cameraFeed.videoHeight
    ) {

      setMessage(
        "Camera not ready yet."
      );

      return;
    }

    const canvas =
      document.createElement("canvas");

    canvas.width =
      cameraFeed.videoWidth;

    canvas.height =
      cameraFeed.videoHeight;

    const ctx =
      canvas.getContext("2d");

    ctx.drawImage(
      cameraFeed,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      (blob) => {

        if (!blob) {

          setMessage(
            "Failed to capture image."
          );

          return;
        }

        selectedFile = new File(
          [blob],
          "captured_image.jpg",
          {
            type: "image/jpeg"
          }
        );

        if (currentObjectURL) {

          URL.revokeObjectURL(
            currentObjectURL
          );
        }

        currentObjectURL =
          URL.createObjectURL(blob);

        imagePreview.src =
          currentObjectURL;

        imagePreview.classList.remove(
          "hidden"
        );

        cameraFeed.classList.add(
          "hidden"
        );

        stopCamera();

        setMessage(
          "Photo captured."
        );

        resultSection.classList.add(
          "hidden"
        );

        heatmapSection.classList.add(
          "hidden"
        );

      },
      "image/jpeg"
    );
  }
);


// ────────────────────────────────────────────────
// SINGLE IMAGE PREDICTION
// ────────────────────────────────────────────────

predictButton.addEventListener(
  "click",
  async () => {

    if (!selectedFile) {

      setMessage(
        "Please upload or capture an image."
      );

      return;
    }

    const formData =
      new FormData();

    formData.append(
      "file",
      selectedFile
    );

    setLoadingState(true);

    setMessage("Predicting...");

    try {

      const controller =
        new AbortController();

      const timeout =
        setTimeout(() => {

          controller.abort();

        }, 15000);

      const response =
  await fetch(API_URL, {

    method: "POST",

    headers: NGROK_HEADERS,

    body: formData,

    signal: controller.signal
  });

      clearTimeout(timeout);

      if (!response.ok) {

        throw new Error(
          `Request failed with status ${response.status}`
        );
      }

      const data =
        await response.json();

      renderResult(data);

      loadHistory();

      setMessage(
        `Prediction: ${data.predicted_class} (${data.status})`
      );

    } catch (error) {

      console.error(error);

      if (error.name === "AbortError") {

        setMessage(
          "Request timed out."
        );

      } else {

        setMessage(
          "Prediction failed. Check backend."
        );
      }

    } finally {

      setLoadingState(false);
    }
  }
);


// ────────────────────────────────────────────────
// BATCH INSPECTION
// ────────────────────────────────────────────────

batchPredictButton.addEventListener(
  "click",
  async () => {

    const files =
      batchImageInput.files;

    if (
      !files ||
      files.length === 0
    ) {

      setMessage(
        "Please select batch images."
      );

      return;
    }

    const formData =
      new FormData();

    for (const file of files) {

      formData.append(
        "files",
        file
      );
    }

    batchPredictButton.disabled = true;

    batchPredictButton.textContent =
      "Running Batch Inspection...";

    setMessage(
      `Processing ${files.length} images...`
    );

    try {

     const response =
  await fetch(
    BATCH_API_URL,
    {
      method: "POST",
      headers: NGROK_HEADERS,
      body: formData
    }
  );

      if (!response.ok) {

        throw new Error(
          `Batch request failed: ${response.status}`
        );
      }

      const data =
        await response.json();

      renderBatchResults(data);

      loadHistory();

      setMessage(
        "Batch inspection completed."
      );

    } catch (error) {

      console.error(error);

      setMessage(
        "Batch inspection failed."
      );

    } finally {

      batchPredictButton.disabled = false;

      batchPredictButton.textContent =
        "Run Batch Inspection";
    }
  }
);


// ────────────────────────────────────────────────
// SINGLE RESULT RENDERING
// ────────────────────────────────────────────────

function renderResult(data) {

  predictedClass.textContent =
    data.predicted_class || "-";

  confidenceValue.textContent =
    typeof data.confidence === "number"
      ? `${data.confidence.toFixed(1)}%`
      : "-";

  const status =
    data.status || "UNKNOWN";

  statusBadge.textContent =
    status;

  statusBadge.className =
    "status-badge";

  statusBadge.classList.add(
    getStatusClass(status)
  );

  if (currentObjectURL) {

    originalResultImage.src =
      currentObjectURL;
  }

  if (data.heatmap_url) {

    heatmapImage.src =
      `${data.heatmap_url}?t=${Date.now()}`;

    heatmapSection.classList.remove(
      "hidden"
    );
  }

  resultSection.classList.remove(
    "show"
  );

  resultSection.classList.remove(
    "hidden"
  );

  requestAnimationFrame(() => {

    resultSection.classList.add(
      "show"
    );
  });
}


// ────────────────────────────────────────────────
// BATCH RESULT RENDERING
// ────────────────────────────────────────────────

function renderBatchResults(data) {

  batchResults.innerHTML = "";

  const summary =
    data.batch_summary;

  batchSummary.innerHTML = `
    <div class="batch-summary-card">

      <p>
        <strong>Total Images:</strong>
        ${summary.total_images}
      </p>

      <p>
        <strong>Successful:</strong>
        ${summary.successful}
      </p>

      <p>
        <strong>Failed:</strong>
        ${summary.failed}
      </p>

      <p>
        <strong>Processing Time:</strong>
        ${summary.processing_time_sec}s
      </p>

    </div>
  `;

  batchSummary.classList.remove(
    "hidden"
  );

  for (const result of data.results) {

    const card =
      document.createElement("div");

    card.className =
      "batch-result-card";

    if (!result.success) {

      card.innerHTML = `
        <div class="batch-error">

          <h3>
            ${result.filename}
          </h3>

          <p>
            Inspection failed
          </p>

          <p>
            ${result.error}
          </p>

        </div>
      `;

      batchResults.appendChild(card);

      continue;
    }

    card.innerHTML = `

      <div class="batch-card-header">

        <h3 class="batch-filename">
          ${result.filename}
        </h3>

        <span class="
          status-badge
          ${getStatusClass(result.status)}
        ">
          ${result.status}
        </span>

      </div>

      <div class="batch-card-grid">

        <div class="batch-image-wrapper">

          <p class="batch-image-label">
            AI Localization
          </p>

          <img
            class="batch-heatmap-image"
            src="${result.heatmap_url}?t=${Date.now()}"
            alt="Heatmap"
          >

        </div>

      </div>

      <div class="batch-card-info">

        <p>
          <strong>Prediction:</strong>
          ${result.predicted_class}
        </p>

        <p>
          <strong>Confidence:</strong>
          ${result.confidence.toFixed(1)}%
        </p>

        <p>
          <strong>Inspection ID:</strong>
          ${result.inspection_id}
        </p>

      </div>
    `;

    batchResults.appendChild(card);
  }
}


// ────────────────────────────────────────────────
// HISTORY
// ────────────────────────────────────────────────

async function loadHistory() {

  try {

    const response =
  await fetch(HISTORY_API_URL, {
    headers: NGROK_HEADERS
  });

    const data =
      await response.json();

    renderHistory(data.history);

  } catch (error) {

    console.error(error);
  }
}


function renderHistory(history) {

  historyContainer.innerHTML = "";

  if (!history || history.length === 0) {

    historyContainer.innerHTML = `
      <p class="history-empty">
        No inspection history yet.
      </p>
    `;

    return;
  }

  for (const item of history) {

    const wrapper =
      document.createElement("div");

    wrapper.className =
      "history-dropdown";

    wrapper.innerHTML = `

      <button class="history-toggle">

        <span>
          ${item.filename}
        </span>

        <span class="history-arrow">
          ▼
        </span>

      </button>

      <div class="history-content hidden">

        <div class="history-card-top">

          <div>

            <p class="history-time">
              ${item.timestamp}
            </p>

          </div>

          <span class="
            status-badge
            ${getStatusClass(item.status)}
          ">
            ${item.status}
          </span>

        </div>

        <div class="history-details">

          <p>
            <strong>Prediction:</strong>
            ${item.prediction}
          </p>

          <p>
            <strong>Confidence:</strong>
            ${item.confidence}%
          </p>

          <p>
            <strong>Inspection ID:</strong>
            ${item.inspection_id}
          </p>

        </div>

        ${
          item.heatmap_url
            ? `
              <img
                class="history-heatmap"
                src="${item.heatmap_url}?t=${Date.now()}"
              >
            `
            : ""
        }

      </div>
    `;

    const toggle =
      wrapper.querySelector(
        ".history-toggle"
      );

    const content =
      wrapper.querySelector(
        ".history-content"
      );

    const arrow =
      wrapper.querySelector(
        ".history-arrow"
      );

    toggle.addEventListener(
      "click",
      () => {

        content.classList.toggle(
          "hidden"
        );

        arrow.textContent =
          content.classList.contains(
            "hidden"
          )
            ? "▼"
            : "▲";
      }
    );

    historyContainer.appendChild(
      wrapper
    );
  }
}


// ────────────────────────────────────────────────
// HISTORY BUTTONS
// ────────────────────────────────────────────────

refreshHistoryButton.addEventListener(
  "click",
  loadHistory
);

clearHistoryButton.addEventListener(
  "click",
  async () => {

    try {

      await fetch(HISTORY_API_URL, {
  method: "DELETE",
  headers: NGROK_HEADERS
});

      loadHistory();

    } catch (error) {

      console.error(error);
    }
  }
);


// ────────────────────────────────────────────────
// STATUS COLORS
// ────────────────────────────────────────────────

function getStatusClass(status) {

  const normalizedStatus =
    String(status).toUpperCase();

  if (normalizedStatus === "CONFIDENT") {
    return "status-confident";
  }

  if (normalizedStatus === "REVIEW") {
    return "status-review";
  }

  return "status-uncertain";
}


// ────────────────────────────────────────────────
// HELPERS
// ────────────────────────────────────────────────

function setLoadingState(isLoading) {

  predictButton.disabled =
    isLoading;

  predictButton.textContent =
    isLoading
      ? "Predicting..."
      : "Run AI Inspection";
}


function setMessage(text) {

  message.textContent = text;
}


function clearPreview() {

  selectedFile = null;

  imageInput.value = "";

  if (currentObjectURL) {

    URL.revokeObjectURL(
      currentObjectURL
    );

    currentObjectURL = null;
  }

  imagePreview.src = "";

  imagePreview.classList.add(
    "hidden"
  );

  previewPlaceholder.classList.remove(
    "hidden"
  );

  resultSection.classList.add(
    "hidden"
  );

  heatmapSection.classList.add(
    "hidden"
  );
}


function stopCamera() {

  if (cameraStream) {

    cameraStream
      .getTracks()
      .forEach(track => track.stop());

    cameraStream = null;
  }

  captureButton.classList.add(
    "hidden"
  );
}


// ────────────────────────────────────────────────
// AUTO LOAD HISTORY
// ────────────────────────────────────────────────

loadHistory();