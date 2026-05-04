from pathlib import Path

DATASET_PATH = Path("data/raw/Dataset_5class")
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

print("\nCounting images per class...\n")

total = 0

for cls_folder in sorted(DATASET_PATH.iterdir()):
    if not cls_folder.is_dir():
        continue

    count = sum(
        1 for f in cls_folder.iterdir()
        if f.suffix.lower() in VALID_EXT
    )

    print(f"{cls_folder.name:<20} : {count}")
    total += count

print("\n" + "=" * 40)
print(f"Total images          : {total}")
print("=" * 40)