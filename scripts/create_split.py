# ============================================================
# scripts/create_split.py
# SAFE STRATIFIED TRAIN/VAL SPLIT (NO LEAKAGE)
# ============================================================

import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

SOURCE_DIR = Path("data/raw/Dataset_5class")
OUTPUT_DIR = Path("data/processed")

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR   = OUTPUT_DIR / "val"

VAL_SPLIT = 0.2
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ────────────────────────────────────────────────
# UTILS
# ────────────────────────────────────────────────

def is_image(file_path):
    return file_path.suffix.lower() in VALID_EXTENSIONS


def collect_dataset(source_dir):
    """
    Collect all images with labels
    """
    image_paths = []
    labels = []

    class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])

    for class_idx, class_name in enumerate(class_names):
        class_folder = source_dir / class_name

        for file in class_folder.iterdir():
            if file.is_file() and is_image(file):
                image_paths.append(file)
                labels.append(class_name)

    return image_paths, labels, class_names


def create_dirs(base_dir, class_names):
    for split in ["train", "val"]:
        for cls in class_names:
            path = base_dir / split / cls
            path.mkdir(parents=True, exist_ok=True)


def copy_files(file_list, dest_root):
    for file_path, class_name in file_list:
        dest_path = dest_root / class_name / file_path.name
        shutil.copy2(file_path, dest_path)


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():

    print("\n🔹 Creating train/val split...\n")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_DIR}")

    # Step 1: Collect dataset
    image_paths, labels, class_names = collect_dataset(SOURCE_DIR)

    print(f"Total images found: {len(image_paths)}")
    print(f"Classes: {class_names}\n")

    # Step 2: Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # Step 3: Prepare folders
    create_dirs(OUTPUT_DIR, class_names)

    # Step 4: Pair paths with labels
    train_data = list(zip(train_paths, train_labels))
    val_data   = list(zip(val_paths, val_labels))

    # Step 5: Copy files
    print("Copying training data...")
    copy_files(train_data, TRAIN_DIR)

    print("Copying validation data...")
    copy_files(val_data, VAL_DIR)

    # Step 6: Summary
    print("\n✅ Split complete!\n")

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}\n")

    # Class distribution
    def count_by_class(data):
        counts = {}
        for _, label in data:
            counts[label] = counts.get(label, 0) + 1
        return counts

    print("Train distribution:")
    print(count_by_class(train_data))

    print("\nVal distribution:")
    print(count_by_class(val_data))

    print("\n📁 Output directory:", OUTPUT_DIR.resolve())


# ────────────────────────────────────────────────

if __name__ == "__main__":
    main()