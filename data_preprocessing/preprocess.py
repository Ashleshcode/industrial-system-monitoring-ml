# ============================================================
# data_preprocessing/preprocess.py
# ============================================================

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Reads from pre-split processed folders
TRAIN_DIR = Path("data/processed/train")
VAL_DIR   = Path("data/processed/val")

# Also expose raw path for weight calculation in train.py
DATASET_PATH = Path("data/processed/train")

IMAGE_SIZE    = 224
BATCH_SIZE    = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    """
    Training — strong augmentation for generalization.
    Validation — clean resize only, no augmentation.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3
            )
        ], p=0.7),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def get_dataloaders():
    """
    Loads train and val datasets from pre-split processed folders.
    No WeightedSampler here — class imbalance handled via
    loss function weights in train.py instead.
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Train folder not found: {TRAIN_DIR.resolve()}\n"
            "Expected: data/processed/train/"
        )
    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Val folder not found: {VAL_DIR.resolve()}\n"
            "Expected: data/processed/val/"
        )

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    return train_loader, val_loader, train_dataset.classes, len(train_dataset.classes)


if __name__ == "__main__":
    train_loader, val_loader, class_names, num_classes = get_dataloaders()
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape   : {images.shape}")
    print(f"Pixel range   : [{images.min():.2f}, {images.max():.2f}]")