# ============================================================
# inference/batch_test.py (FINAL — deterministic + stable)
# ============================================================

import sys
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

# ────────────────────────────────────────────────
# 🔥 FULL DETERMINISM
# ────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.predict import predict
from models.cnn_model import get_device, load_model

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

DATASET_PATH = Path("data/test_images")
MODEL_PATH   = Path("saved_models/best_model.pth")
RESULTS_PATH = Path("results/batch_test_report.txt")

IMAGES_PER_CLASS = 15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "Broken stitch",
    "defect free",
    "hole",
    "stain"
]

# ────────────────────────────────────────────────
# OPTIONAL CONTROL
# ────────────────────────────────────────────────

USE_TTA = True   # 🔥 turn ON later if needed


# ────────────────────────────────────────────────
# SAFE NORMALIZATION
# ────────────────────────────────────────────────

def normalize_probs(probs):
    probs = np.array(probs, dtype=np.float32)
    total = np.sum(probs)
    if total == 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


# ────────────────────────────────────────────────
# TTA (optional)
# ────────────────────────────────────────────────

def predict_with_tta(img_path, model, device):

    preds = []

    for _ in range(3):
        _, _, probs = predict(img_path, model=model, device=device)

        prob_array = np.array([probs[c] for c in CLASS_NAMES])
        prob_array = normalize_probs(prob_array)

        preds.append(prob_array)

    avg_probs = np.mean(preds, axis=0)
    avg_probs = normalize_probs(avg_probs)

    pred_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx] * 100)

    return pred_idx, confidence, avg_probs


# ────────────────────────────────────────────────
# IMAGE LOADING (FIXED)
# ────────────────────────────────────────────────

def get_test_images(dataset_path: Path, n_per_class: int):

    test_images = {}

    for cls in sorted(CLASS_NAMES):

        cls_folder = dataset_path / cls

        if not cls_folder.exists():
            print(f"Warning — Folder not found: {cls_folder}")
            continue

        # 🔥 SORTED — CRITICAL FIX
        all_images = sorted([
            f for f in cls_folder.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ])

        if len(all_images) == 0:
            print(f"Warning — No images found in: {cls_folder}")
            continue

        # 🔥 DETERMINISTIC SELECTION (NO RANDOM)
        sampled = all_images[:min(n_per_class, len(all_images))]

        test_images[cls] = sampled

    return test_images


# ────────────────────────────────────────────────
# SAVE WRONG PREDICTIONS
# ────────────────────────────────────────────────

def save_wrong_predictions(wrong_predictions: list):

    if not wrong_predictions:
        print("\n  All predictions correct — no wrong prediction report needed.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n     = len(wrong_predictions)
    cols  = min(4, n)
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, wrong in enumerate(wrong_predictions):
        ax  = axes[idx]
        img = mpimg.imread(str(wrong["path"]))

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        ax.set_title(
            f"TRUE : {wrong['true_class']}\n"
            f"PRED : {wrong['pred_class']} ({wrong['confidence']:.1f}%)",
            fontsize=9,
            color="red",
            fontweight="bold"
        )
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)
            spine.set_visible(True)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Wrong Predictions — {n} errors",
        fontsize=14,
        fontweight="bold",
        color="red"
    )

    plt.tight_layout()

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    save_path = RESULTS_DIR / "wrong_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Wrong predictions image saved to: {save_path}")


# ────────────────────────────────────────────────
# RUN TEST
# ────────────────────────────────────────────────

def run_batch_test(test_images: dict, model, device) -> dict:

    results = {
        cls: {
            "correct"    : 0,
            "total"      : 0,
            "wrong_as"   : defaultdict(int),
            "confidences": []
        }
        for cls in CLASS_NAMES
    }

    wrong_predictions = []

    total_images = sum(len(imgs) for imgs in test_images.values())
    processed    = 0

    print(f"\nTesting {total_images} images across {len(test_images)} classes...\n")

    for true_class, image_paths in test_images.items():
        for img_path in image_paths:

            try:

                if USE_TTA:
                    pred_idx, confidence, _ = predict_with_tta(
                        str(img_path), model, device
                    )
                    predicted_class = CLASS_NAMES[pred_idx]
                else:
                    predicted_class, confidence, _ = predict(
                        str(img_path), model=model, device=device
                    )

                results[true_class]["total"] += 1
                results[true_class]["confidences"].append(confidence)

                is_correct = (predicted_class == true_class)

                if is_correct:
                    results[true_class]["correct"] += 1
                else:
                    results[true_class]["wrong_as"][predicted_class] += 1
                    wrong_predictions.append({
                        "path"       : img_path,
                        "true_class" : true_class,
                        "pred_class" : predicted_class,
                        "confidence" : confidence
                    })

                processed += 1
                status = "OK   " if is_correct else "WRONG"

                print(
                    f"  [{processed}/{total_images}] "
                    f"{true_class:<20} → "
                    f"[{status}] "
                    f"{predicted_class:<20} "
                    f"({confidence:.1f}%)"
                )

            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")

    save_wrong_predictions(wrong_predictions)

    return results


# ────────────────────────────────────────────────
# REPORT
# ────────────────────────────────────────────────

def print_report(results):

    lines = []
    lines.append("=" * 65)
    lines.append("       FABRIC DEFECT MODEL — BATCH TEST REPORT")
    lines.append("=" * 65)

    total_correct = 0
    total_images = 0

    for cls in CLASS_NAMES:
        r = results[cls]

        if r["total"] == 0:
            continue

        acc = r["correct"] / r["total"] * 100
        avg_conf = np.mean(r["confidences"]) if r["confidences"] else 0
        bar = "|" * int(acc // 5)

        total_correct += r["correct"]
        total_images += r["total"]

        lines.append(f"\nClass: {cls}")
        lines.append(f"Accuracy        : {acc:.1f}%  {bar}")
        lines.append(f"Avg Confidence  : {avg_conf:.1f}%")
        lines.append(f"Tested          : {r['correct']}/{r['total']} correct")

        if r["wrong_as"]:
            lines.append("Confused with   :")
            for wrong_cls, count in sorted(
                r["wrong_as"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"  -> {wrong_cls:<20}: {count}")

    overall_acc = total_correct / total_images * 100

    lines.append("\n" + "=" * 65)
    lines.append(f"OVERALL ACCURACY : {overall_acc:.2f}% ({total_correct}/{total_images})")
    lines.append("=" * 65)

    print("\n".join(lines))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines))


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

if __name__ == "__main__":

    device = get_device()

    model = load_model(
        num_classes=len(CLASS_NAMES),
        path=str(MODEL_PATH),
        device=device
    )

    model.eval()   # 🔥 EXTRA SAFETY

    test_images = get_test_images(DATASET_PATH, IMAGES_PER_CLASS)

    results = run_batch_test(test_images, model, device)

    print_report(results)