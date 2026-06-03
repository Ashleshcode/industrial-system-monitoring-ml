# utils/plot_training.py
# Run this after training to generate training curve plots

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ── Paste your training log values here ──────────────
# Copy from your terminal output

EPOCHS = list(range(1, 22))

TRAIN_LOSS = [
    0.7287, 0.3808, 0.1471, 0.0710, 0.0304,
    0.0207, 0.0144, 0.0132, 0.0080, 0.0084,
    0.0071, 0.0057, 0.0047, 0.0053, 0.0048,
    0.0033, 0.0035, 0.0057, 0.0030, 0.0025,
    0.0028
]

VAL_ACC = [
    76.85, 91.64, 91.00, 92.60, 94.86,
    95.82, 97.11, 96.78, 97.75, 98.39,
    98.39, 97.75, 98.39, 99.04, 98.71,
    98.71, 98.39, 97.75, 98.39, 98.71,
    98.71
]

TRAIN_ACC = [
    34.14, 67.11, 84.33, 90.43, 93.60,
    95.00, 95.58, 96.10, 96.62, 96.62,
    96.32, 96.84, 96.91, 96.84, 96.69,
    97.28, 97.57, 96.32, 97.28, 97.65,
    97.65
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1 — Loss curve ──
    axes[0].plot(EPOCHS, TRAIN_LOSS, 'b-o', markersize=4,
                 linewidth=2, label="Train Loss")
    axes[0].axvline(x=14, color='red', linestyle='--',
                    alpha=0.7, label="Best model (epoch 14)")
    axes[0].set_title("Training Loss over Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(alpha=0.3)

    # ── Plot 2 — Accuracy curve ──
    axes[1].plot(EPOCHS, TRAIN_ACC, 'b-o', markersize=4,
                 linewidth=2, label="Train Accuracy")
    axes[1].plot(EPOCHS, VAL_ACC, 'g-s', markersize=4,
                 linewidth=2, label="Val Accuracy")
    axes[1].axvline(x=14, color='red', linestyle='--',
                    alpha=0.7, label="Best model (epoch 14)")
    axes[1].axhline(y=99.04, color='green', linestyle=':',
                    alpha=0.7, label="Best Val Acc: 99.04%")
    axes[1].set_title("Train vs Validation Accuracy", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(30, 102)

    plt.suptitle(
        "EfficientNetV2-S — Fabric Defect Detection Training",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    save_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {save_path}")


def plot_batch_test_results():
    """Bar chart of real world batch test accuracy per class."""

    classes  = ["Broken stitch", "defect free", "hole", "stain"]
    accuracy = [80.0, 100.0, 85.7, 90.0]
    colors   = ["#e74c3c" if a < 85 else "#2ecc71" for a in accuracy]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(classes, accuracy, color=colors,
                  edgecolor="white", linewidth=0.8)

    for bar, acc in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    ax.axhline(y=89.19, color='blue', linestyle='--',
               linewidth=1.5, label=f"Overall: 89.19%")
    ax.set_title(
        "Real-world Batch Test Accuracy per Class",
        fontsize=13, pad=12
    )
    ax.set_xlabel("Defect Class", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_path = RESULTS_DIR / "batch_test_accuracy.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Batch test chart saved to: {save_path}")


if __name__ == "__main__":
    plot_training_curves()
    plot_batch_test_results()
    print("\nAll plots saved to results/ folder.")