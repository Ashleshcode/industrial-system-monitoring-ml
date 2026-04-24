# training/train.py — FINAL STABLE VERSION

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_preprocessing.preprocess import get_dataloaders
from models.cnn_model import build_model, get_device, save_model, print_model_summary

NUM_EPOCHS    = 25
LEARNING_RATE = 1e-4
SAVE_DIR      = Path("saved_models")
SAVE_PATH     = SAVE_DIR / "best_model.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        correct       += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, correct / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            correct       += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, correct / total_samples


def train():
    device = get_device()
    SAVE_DIR.mkdir(exist_ok=True)

    print("\nLoading dataset...")
    train_loader, val_loader, class_names, num_classes = get_dataloaders()
    print(f"Classes: {class_names}\n")

    # ── Model ───────────────────────────────────
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model = model.to(device)
    print_model_summary(model, num_classes)

    # ── Class weights — sqrt balancing ──────────
    # Load full dataset just to get targets for weight calculation
    from torchvision import datasets
    from data_preprocessing.preprocess import DATASET_PATH
    full_dataset  = datasets.ImageFolder(root=str(DATASET_PATH))
    all_targets   = np.array(full_dataset.targets)
    class_counts  = np.bincount(all_targets)
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)

    # Boost defect free (index 4) — it has most images but
    # is most likely to be confused with stain in real world
    class_weights[4] *= 1.5

    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights applied:")
    for i, (cls, w) in enumerate(zip(class_names, class_weights)):
        print(f"  {cls:<20} : {w:.4f}")

    # ── Loss ────────────────────────────────────
    # label_smoothing=0.05 — light smoothing, don't over-soften
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.05
    )

    # ── Optimizer ───────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=4,
        factor=0.5
    )

    # ── Training loop ───────────────────────────
    best_val_accuracy = 0.0

    print("\n" + "=" * 65)
    print(f"  Starting training — {NUM_EPOCHS} epochs — GPU enabled")
    print("=" * 65)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc'}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model, str(SAVE_PATH))
            saved_tag = " ← best saved"
        else:
            saved_tag = ""

        print(
            f"  {epoch:<8} "
            f"{train_loss:<14.4f} "
            f"{train_acc * 100:<14.2f} "
            f"{val_loss:<14.4f} "
            f"{val_acc * 100:<12.2f}"
            f"{saved_tag}"
        )

    print("-" * 65)
    print(f"\n✅ Training complete!")
    print(f"   Best Val Accuracy : {best_val_accuracy * 100:.2f}%")
    print(f"   Model saved at    : {SAVE_PATH}")


if __name__ == "__main__":
    train()