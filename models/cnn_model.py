# ============================================================
# models/cnn_model.py
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = False):
    """
    EfficientNetV2-S with custom classifier head.
    freeze_backbone=False means full fine-tuning (recommended).
    freeze_backbone=True only trains the classifier head.
    """
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        # Unfreeze everything — full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def print_model_summary(model, num_classes: int):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print("=" * 45)
    print("      MODEL SUMMARY — EfficientNetV2-S")
    print("=" * 45)
    print(f"  Output classes    : {num_classes}")
    print(f"  Total params      : {total_params:,}")
    print(f"  Trainable params  : {trainable_params:,}")
    print(f"  Frozen params     : {frozen_params:,}")
    print("=" * 45)


def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(num_classes: int, path: str, device):
    """
    Loads saved model weights for inference.
    Always loads with full fine-tuning architecture.
    """
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Model loaded from: {path}")
    return model