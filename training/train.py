# ============================================================
# training/train.py (CLEAN — DEMO OPTIMIZED)
# ============================================================

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_preprocessing.preprocess import get_dataloaders, DATASET_PATH
from models.cnn_model import build_model, get_device, save_model, print_model_summary

NUM_EPOCHS    = 60
LEARNING_RATE = 5e-5   # 🔥 slightly higher for faster convergence
SAVE_DIR      = Path("saved_models")
SAVE_PATH     = SAVE_DIR / "best_model.pth"


# ────────────────────────────────────────────────
# EARLY STOPPING
# ────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = 0.0
        self.stop       = False

    def __call__(self, val_accuracy: float):
        if val_accuracy > self.best_score + self.min_delta:
            self.best_score = val_accuracy
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


# ────────────────────────────────────────────────
# FOCAL LOSS
# ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt      = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss


# ────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ────────────────────────────────────────────────

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


# ────────────────────────────────────────────────
# VALIDATE
# ────────────────────────────────────────────────

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


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def train():
    device = get_device()
    SAVE_DIR.mkdir(exist_ok=True)

    print("\nLoading dataset...")
    train_loader, val_loader, class_names, num_classes = get_dataloaders()
    print(f"Classes: {class_names}\n")

    # ── Model ────────────────────────────────────
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model = model.to(device)
    print_model_summary(model, num_classes)

    # ── Class weights ────────────────────────────
    from torchvision import datasets
    train_dataset_raw = datasets.ImageFolder(root=str(DATASET_PATH))
    all_targets       = np.array(train_dataset_raw.targets)
    class_counts      = np.bincount(all_targets)

    # balanced but not too aggressive
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print("\nClass weights:")
    for cls, w, count in zip(class_names, class_weights, class_counts):
        print(f"  {cls:<20} : weight={w:.4f}  (train images={count})")

    # ── Loss ─────────────────────────────────────
    criterion = FocalLoss(
        weight=class_weights_tensor,
        gamma=2.0
    )

    # ── Optimizer ────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ── Scheduler ────────────────────────────────
    def warmup_cosine(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        else:
            import math
            progress = (epoch - 5) / (NUM_EPOCHS - 5)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine
    )

    # ── Early stopping ───────────────────────────
    early_stopping = EarlyStopping(patience=7)

    best_val_accuracy = 0.0

    print("\n" + "=" * 65)
    print(f"  Training — up to {NUM_EPOCHS} epochs")
    print("=" * 65)
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc'}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model, str(SAVE_PATH))
            saved_tag = " <- best saved"
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

        early_stopping(val_acc)
        if early_stopping.stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    print("-" * 65)
    print(f"\nTraining complete!")
    print(f"  Best Val Accuracy : {best_val_accuracy * 100:.2f}%")
    print(f"  Model saved at    : {SAVE_PATH}")


if __name__ == "__main__":
    train()