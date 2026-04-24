# ============================================================
# data_preprocessing/preprocess.py
# ============================================================

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

DATASET_PATH = Path("data/raw/Dataset")
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
RANDOM_SEED  = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ────────────────────────────────────────────────
# TRANSFORMS
# ────────────────────────────────────────────────

def get_transforms():
    """
    Balanced augmentation for real-world generalization
    """

    train_transform = transforms.Compose([

        # 🔥 Most important: simulate zoom + framing differences
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

        transforms.Grayscale(num_output_channels=3),

        # Spatial variations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),

        # 🔥 Controlled lighting variation (not too strong)
        transforms.ColorJitter(
    brightness=0.6,   # 🔥 increase (main fix)
    contrast=0.5,
    saturation=0.3,
    hue=0.1
),

        # 🔥 Small translation (not extreme)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),

        # 🔥 Real-world blur simulation
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


# ────────────────────────────────────────────────
# DATASET LOADING
# ────────────────────────────────────────────────

def load_dataset(dataset_path: Path, transform):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path.resolve()}\n"
            "Ensure data is placed at data/raw/Dataset/"
        )

    return datasets.ImageFolder(root=str(dataset_path), transform=transform)


# ────────────────────────────────────────────────
# STRATIFIED SPLIT
# ────────────────────────────────────────────────

def split_dataset(dataset):
    targets = dataset.targets

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=targets
    )

    return train_idx, val_idx


# ────────────────────────────────────────────────
# WEIGHTED SAMPLER (FIXED)
# ────────────────────────────────────────────────

def get_sampler(train_targets: list):
    """
    Balanced sampling using sqrt inverse weights.
    Also boosts 'defect free' slightly for real-world stability.
    """
    class_counts  = np.bincount(train_targets)

    # 🔥 FIX: less aggressive weighting
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)

    # 🔥 Boost defect free (index = 4)
    class_weights[4] *= 1.3

    # Normalize
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    sample_weights = [class_weights[t] for t in train_targets]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


# ────────────────────────────────────────────────
# MAIN — call this from train.py
# ────────────────────────────────────────────────

def get_dataloaders(dataset_path: Path = DATASET_PATH):

    train_transform, val_transform = get_transforms()

    full_dataset = load_dataset(dataset_path, train_transform)

    train_idx, val_idx = split_dataset(full_dataset)

    train_dataset = Subset(full_dataset, train_idx)

    val_full_dataset = load_dataset(dataset_path, val_transform)
    val_dataset      = Subset(val_full_dataset, val_idx)

    train_targets = [full_dataset.targets[i] for i in train_idx]
    sampler = get_sampler(train_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, full_dataset.classes, len(full_dataset.classes)


# ────────────────────────────────────────────────
# SANITY CHECK
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running preprocessing pipeline...")

    train_loader, val_loader, class_names, num_classes = get_dataloaders()

    print(f"\n✅ Classes ({num_classes}): {class_names}")
    print(f"   Train batches : {len(train_loader)}")
    print(f"   Val batches   : {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"\n   Batch shape   : {images.shape}")
    print(f"   Label shape   : {labels.shape}")
    print(f"   Pixel range   : [{images.min():.2f}, {images.max():.2f}]")

    print("\n✅ Preprocessing ready!")