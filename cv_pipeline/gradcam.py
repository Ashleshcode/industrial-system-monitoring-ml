# ============================================================
# cv_pipeline/gradcam.py
# INDUSTRIAL STYLE DEFECT LOCALIZATION
# ============================================================

import cv2
import torch
import numpy as np
from PIL import Image


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    # ────────────────────────────────────────────────

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    # ────────────────────────────────────────────────

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ────────────────────────────────────────────────

    def generate(self, input_tensor, class_idx=None):

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not provided
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Clear gradients
        self.model.zero_grad()

        # Backprop for target class
        loss = output[:, class_idx]
        loss.backward()

        # Extract gradients + activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Channel weights
        weights = gradients.mean(dim=(1, 2))

        # Weighted feature combination
        cam = torch.zeros(
            activations.shape[1:],
            device=input_tensor.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Convert to numpy
        cam = cam.cpu().numpy()

        return cam


# ────────────────────────────────────────────────
# INDUSTRIAL VISUALIZATION
# ────────────────────────────────────────────────

def overlay_heatmap(image_pil, cam):

    # PIL → numpy
    image = np.array(image_pil).copy()

    # Resize CAM
    heatmap = cv2.resize(
        cam,
        (image.shape[1], image.shape[0])
    )

    # Smooth slightly
    heatmap = cv2.GaussianBlur(
        heatmap,
        (11, 11),
        0
    )

    # Threshold strong activations only
    binary = np.zeros_like(heatmap)

    binary[heatmap > 0.55] = 1

    # Convert to uint8
    binary_uint8 = np.uint8(binary * 255)

    # Find contours
    contours, _ = cv2.findContours(
        binary_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours + boxes
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noisy regions
        if area < 2500:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Red rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        # Optional contour outline
       

        # Label
        cv2.putText(
            image,
            "Defect Region",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return image