# utils/check_duplicates.py

import hashlib
from pathlib import Path
from collections import defaultdict

DATASET_PATH = Path("data/raw/Dataset_5class")
VALID_EXT    = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "Broken stitch",
    "Needle mark",
    "defect free",
    "hole",
    "stain"
]

def get_file_hash(filepath: Path) -> str:
    """Returns MD5 hash of file — identical files have identical hashes."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("Checking for duplicate images...\n")

total_files      = 0
total_duplicates = 0

for cls in CLASS_NAMES:
    cls_folder = DATASET_PATH / cls
    images     = [
        f for f in cls_folder.iterdir()
        if f.suffix.lower() in VALID_EXT
    ]

    # Map hash → list of files with that hash
    hash_map = defaultdict(list)
    for img in images:
        h = get_file_hash(img)
        hash_map[h].append(img.name)

    duplicates = {
        h: names for h, names in hash_map.items()
        if len(names) > 1
    }

    unique_count    = len(hash_map)
    duplicate_count = sum(len(v) - 1 for v in duplicates.values())

    total_files      += len(images)
    total_duplicates += duplicate_count

    status = "PROBLEM" if duplicate_count > 0 else "OK"
    print(f"  [{status}] {cls:<20}")
    print(f"    Total images    : {len(images)}")
    print(f"    Unique images   : {unique_count}")
    print(f"    Duplicates      : {duplicate_count}")

    if duplicates:
        print(f"    Example duplicates:")
        for i, (h, names) in enumerate(list(duplicates.items())[:3]):
            print(f"      {names}")
    print()

print("=" * 45)
print(f"  Total files      : {total_files}")
print(f"  Total duplicates : {total_duplicates}")
print(f"  Duplicate rate   : {total_duplicates/total_files*100:.1f}%")
print("=" * 45)