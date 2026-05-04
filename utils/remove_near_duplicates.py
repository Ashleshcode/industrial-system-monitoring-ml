# utils/remove_near_duplicates.py

from pathlib import Path
from PIL import Image
import imagehash

DATASET_PATH = Path("data/raw/Dataset_5class")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

THRESHOLD = 2       # STRICT → only very similar images
DELETE = True     # ALWAYS run False first

# 🔥 Choose which classes to clean (IMPORTANT)
TARGET_CLASSES = [
    #"defect free",   # start with this only
    # "hole",
     "Broken stitch",
    # "Needle mark",
    # "stain"
]

print("Checking for NEAR duplicates...\n")

total_removed = 0
total_checked = 0

for cls_folder in DATASET_PATH.iterdir():
    if not cls_folder.is_dir():
        continue

    if cls_folder.name not in TARGET_CLASSES:
        continue

    print(f"Processing: {cls_folder.name}")

    hashes = []
    removed_in_class = 0
    checked_in_class = 0

    for img_path in cls_folder.iterdir():
        if img_path.suffix.lower() not in VALID_EXT:
            continue

        checked_in_class += 1
        total_checked += 1

        try:
            img = Image.open(img_path).convert("RGB")
            h = imagehash.phash(img)

            duplicate_found = False

            for existing_hash, existing_path in hashes:
                if h - existing_hash <= THRESHOLD:
                    print(f"  DUPLICATE: {img_path.name} ≈ {existing_path.name}")

                    if DELETE:
                        img_path.unlink()
                        removed_in_class += 1
                        total_removed += 1

                    duplicate_found = True
                    break

            if not duplicate_found:
                hashes.append((h, img_path))

        except Exception as e:
            print(f"  Error reading {img_path.name}: {e}")

    print(f"  Checked: {checked_in_class}")
    print(f"  Removed: {removed_in_class}")
    print("-" * 40)

print("\n" + "=" * 50)
print(f"Total checked : {total_checked}")
print(f"Total removed : {total_removed}")
print("=" * 50)