# utils/setup_test_images.py
# Automatically copies random images from dataset into test folder

import random
import shutil
from pathlib import Path

DATASET_PATH = Path("data/raw/Dataset_5class")
TEST_PATH    = Path("data/test_images")
N_PER_CLASS  = 5
VALID_EXT    = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "Broken stitch",
  
    "defect free",
    "hole",
    "stain"
]

random.seed(99)  # different seed from training (was 42) so different images

for cls in CLASS_NAMES:
    src_folder = DATASET_PATH / cls
    dst_folder = TEST_PATH / cls
    dst_folder.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in src_folder.iterdir()
        if f.suffix.lower() in VALID_EXT
    ]

    sampled = random.sample(images, min(N_PER_CLASS, len(images)))

    for img in sampled:
        shutil.copy(img, dst_folder / img.name)

    print(f"✅ {cls:<20} : {len(sampled)} images copied")

print("\nDone. Run: python inference/batch_test.py")