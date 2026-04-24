# ============================================================
# models/cnn_model.py
#
# Defines the transfer learning model using pretrained ResNet18.
# Optimized for better generalization on fabric defect data.
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


# ────────────────────────────────────────────────
# MODEL BUILDER
# ────────────────────────────────────────────────

def build_model(num_classes: int, freeze_backbone: bool = True):
    """
    Uses ResNet18 (better generalization than ResNet50 for small datasets).
    """

    # 🔥 BACK TO RESNET18 (IMPORTANT)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze backbone if needed
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ResNet18 final layer input = 512
    in_features = model.fc.in_features

    # 🔥 Clean, stable classifier head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


# ────────────────────────────────────────────────
# DEVICE HELPER
# ────────────────────────────────────────────────

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ────────────────────────────────────────────────
# MODEL SUMMARY
# ────────────────────────────────────────────────

def print_model_summary(model, num_classes: int):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print("=" * 45)
    print("         MODEL SUMMARY — ResNet18")
    print("=" * 45)
    print(f"  Output classes    : {num_classes}")
    print(f"  Total params      : {total_params:,}")
    print(f"  Trainable params  : {trainable_params:,}")
    print(f"  Frozen params     : {frozen_params:,}")
    print("=" * 45)


# ────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ────────────────────────────────────────────────

def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to: {path}")


def load_model(num_classes: int, path: str, device):
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {path}")
    return model


# ────────────────────────────────────────────────
# SANITY CHECK
# ────────────────────────────────────────────────

if __name__ == "__main__":

    NUM_CLASSES = 9

    device = get_device()
    model  = build_model(num_classes=NUM_CLASSES, freeze_backbone=True)
    model  = model.to(device)

    print_model_summary(model, NUM_CLASSES)

    dummy_input  = torch.randn(2, 3, 224, 224).to(device)
    dummy_output = model(dummy_input)

    print(f"\n  Dummy input shape  : {dummy_input.shape}")
    print(f"  Dummy output shape : {dummy_output.shape}")
    print(f"  ← Should be [2, {NUM_CLASSES}]")
    print("\n✅ Model ready!")