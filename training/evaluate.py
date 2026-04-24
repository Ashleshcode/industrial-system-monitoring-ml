# ============================================================
# training/evaluate.py
#
# Loads the best saved model and evaluates it on the
# validation set. Produces accuracy scores, per-class
# breakdown, and a confusion matrix saved to results/
# ============================================================

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_preprocessing.preprocess import get_dataloaders
from models.cnn_model import get_device, load_model


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

MODEL_PATH   = Path("saved_models/best_model.pth")
RESULTS_DIR  = Path("results")


# ────────────────────────────────────────────────
# GET ALL PREDICTIONS
# ────────────────────────────────────────────────

def get_predictions(model, loader, device):
    """
    Runs the model on every batch in the loader.
    Collects all true labels and predicted labels.

    Returns:
        all_preds  : list of predicted class indices
        all_labels : list of true class indices
    """
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs    = model(images)
            predicted  = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


# ────────────────────────────────────────────────
# OVERALL ACCURACY
# ────────────────────────────────────────────────

def print_accuracy(preds, labels, class_names):
    """
    Prints overall accuracy and per-class breakdown.
    Per-class accuracy tells you which defect types
    the model is struggling with specifically.
    """
    overall_acc = accuracy_score(labels, preds)

    print("=" * 50)
    print("        EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Overall Accuracy : {overall_acc * 100:.2f}%")
    print("=" * 50)

    # Per-class accuracy
    print("\n  Per-Class Accuracy:")
    print("  " + "-" * 40)

    for i, cls in enumerate(class_names):
        # Find all samples belonging to this class
        cls_mask    = (labels == i)
        cls_correct = (preds[cls_mask] == labels[cls_mask]).sum()
        cls_total   = cls_mask.sum()

        if cls_total > 0:
            cls_acc = cls_correct / cls_total * 100
        else:
            cls_acc = 0.0

        bar = "█" * int(cls_acc // 5)
        print(f"  {cls:<20} : {cls_acc:>6.2f}%  {bar}")

    print()


# ────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ────────────────────────────────────────────────

def save_classification_report(preds, labels, class_names):
    """
    Saves precision, recall, F1-score for each class.

    - Precision : of all images predicted as class X,
                  how many were actually class X?
    - Recall    : of all actual class X images,
                  how many did we correctly find?
    - F1-score  : harmonic mean of precision and recall
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    report = classification_report(labels, preds, target_names=class_names)

    print("  Classification Report:")
    print("  " + "-" * 40)
    print(report)

    # Save to file
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("FABRIC DEFECT DETECTION — CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"  ✅ Report saved to: {report_path}")


# ────────────────────────────────────────────────
# CONFUSION MATRIX
# ────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names):
    """
    Plots and saves a confusion matrix heatmap.

    Rows    = actual class
    Columns = predicted class
    Diagonal = correct predictions (want these high)
    Off-diagonal = mistakes (want these low/zero)

    Example: if row='hole', col='stain' has value 5,
    it means 5 hole images were wrongly predicted as stain.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    cm = confusion_matrix(labels, preds)

    # Normalize so each row sums to 1 (shows % not raw counts)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Raw counts ──
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        linewidths=0.5
    )
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13, pad=12)
    axes[0].set_ylabel("Actual Class", fontsize=11)
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    # ── Normalized (%) ──
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Confusion Matrix — Normalized (0 to 1)", fontsize=13, pad=12)
    axes[1].set_ylabel("Actual Class", fontsize=11)
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle(
        "Fabric Defect Detection — Model Evaluation",
        fontsize=15, y=1.01
    )
    plt.tight_layout()

    save_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  ✅ Confusion matrix saved to: {save_path}")


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def evaluate():
    """
    Full evaluation pipeline:
      1. Load best saved model
      2. Run on validation set
      3. Print accuracy + per-class breakdown
      4. Save classification report
      5. Save confusion matrix
    """
    device = get_device()

    # ── Check model exists ──
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at: {MODEL_PATH}\n"
            "Please run training/train.py first."
        )

    # ── Load data ──
    print("\nLoading validation data...")
    _, val_loader, class_names, num_classes = get_dataloaders()

    # ── Load model ──
    print("Loading best saved model...")
    model = load_model(
        num_classes=num_classes,
        path=str(MODEL_PATH),
        device=device
    )

    # ── Predict ──
    print("Running predictions on validation set...\n")
    preds, labels = get_predictions(model, val_loader, device)

    # ── Results ──
    print_accuracy(preds, labels, class_names)
    save_classification_report(preds, labels, class_names)
    plot_confusion_matrix(preds, labels, class_names)

    print("\n✅ Evaluation complete! Check the results/ folder.")


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()