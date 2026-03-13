"""
Skin Cancer Detection — Dataset Pipeline
Loads ISIC 2019 from Hugging Face and applies Albumentations augmentations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import yaml
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ---- ISIC 2019 Label Mapping ----
LABEL_MAP = {
    "MEL": 0,   # Melanoma
    "NV": 1,    # Melanocytic nevus
    "BCC": 2,   # Basal cell carcinoma
    "AK": 3,    # Actinic keratosis
    "BKL": 4,   # Benign keratosis
    "DF": 5,    # Dermatofibroma
    "VASC": 6,  # Vascular lesion
    "SCC": 7,   # Squamous cell carcinoma
}

LABEL_NAMES = list(LABEL_MAP.keys())

LABEL_DESCRIPTIONS = {
    "MEL": "Melanoma — most dangerous skin cancer",
    "NV": "Melanocytic nevus — common benign mole",
    "BCC": "Basal cell carcinoma — most common skin cancer",
    "AK": "Actinic keratosis — precancerous lesion",
    "BKL": "Benign keratosis — seborrheic keratosis, solar lentigo",
    "DF": "Dermatofibroma — benign fibrous growth",
    "VASC": "Vascular lesion — angiomas, angiokeratomas, etc.",
    "SCC": "Squamous cell carcinoma — second most common skin cancer",
}


def get_train_transforms(cfg):
    """Heavy augmentations for training — key to achieving >95% accuracy."""
    aug_cfg = cfg["data"]["augmentation"]
    img_size = cfg["data"]["image_size"]

    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=tuple(aug_cfg["random_resized_crop"]["scale"]),
            ratio=tuple(aug_cfg["random_resized_crop"]["ratio"]),
            p=1.0
        ),
        A.HorizontalFlip(p=aug_cfg["horizontal_flip_p"]),
        A.VerticalFlip(p=aug_cfg["vertical_flip_p"]),
        A.Affine(
            translate_percent={"x": (-aug_cfg["shift_scale_rotate"]["shift_limit"], aug_cfg["shift_scale_rotate"]["shift_limit"]),
                              "y": (-aug_cfg["shift_scale_rotate"]["shift_limit"], aug_cfg["shift_scale_rotate"]["shift_limit"])},
            scale=(1 - aug_cfg["shift_scale_rotate"]["scale_limit"], 1 + aug_cfg["shift_scale_rotate"]["scale_limit"]),
            rotate=(-aug_cfg["shift_scale_rotate"]["rotate_limit"], aug_cfg["shift_scale_rotate"]["rotate_limit"]),
            mode=0,
            p=aug_cfg["shift_scale_rotate"]["p"]
        ),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=aug_cfg["hue_saturation"]["hue_shift"],
                sat_shift_limit=aug_cfg["hue_saturation"]["sat_shift"],
                val_shift_limit=aug_cfg["hue_saturation"]["val_shift"],
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg["brightness_contrast"]["brightness_limit"],
                contrast_limit=aug_cfg["brightness_contrast"]["contrast_limit"],
                p=1.0
            ),
        ], p=0.6),
        A.OneOf([
            A.CLAHE(clip_limit=aug_cfg["clahe"]["clip_limit"], p=1.0),
            A.Sharpen(p=1.0),
            A.Emboss(p=1.0),
        ], p=0.3),
        A.GaussNoise(
            std_range=(0.02, 0.1),
            p=aug_cfg["gauss_noise"]["p"]
        ),
        A.CoarseDropout(
            num_holes_range=(1, aug_cfg["coarse_dropout"]["max_holes"]),
            hole_height_range=(1, aug_cfg["coarse_dropout"]["max_height"]),
            hole_width_range=(1, aug_cfg["coarse_dropout"]["max_width"]),
            fill=0,
            p=aug_cfg["coarse_dropout"]["p"]
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(cfg):
    """Minimal transforms for validation/test."""
    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(int(img_size * 1.15), int(img_size * 1.15)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for ISIC 2019 skin lesion classification."""

    def __init__(self, images, labels, transform=None, class_names=None):
        """
        Args:
            images: list of PIL images or HF image objects
            labels: list of integer labels
            transform: albumentations transform pipeline
            class_names: list of class name strings
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or LABEL_NAMES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image — handle both PIL and HF formats
        image = self.images[idx]
        if not isinstance(image, np.ndarray):
            image = np.array(image.convert("RGB"))

        label = self.labels[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Metadata for interpretability
        metadata = {
            "index": idx,
            "class_name": self.class_names[label] if label < len(self.class_names) else "UNKNOWN",
        }

        return image, torch.tensor(label, dtype=torch.long), metadata


def load_hf_dataset(cfg, smoke_test=False):
    """
    Load ISIC 2019 skin cancer dataset from Hugging Face.
    Returns train/val/test splits with class weights.
    """
    dataset_name = cfg["data"]["dataset_name"]
    logger.info(f"Loading dataset: {dataset_name} from Hugging Face...")

    # Load the dataset
    ds = load_dataset(dataset_name)

    # The dataset may have different split structures
    if "train" in ds and "validation" in ds and "test" in ds:
        # All 3 splits available — use them directly
        all_data = ds["train"]
        val_data = ds["validation"]
        test_data = ds["test"]
    elif "train" in ds and "test" in ds:
        all_data = ds["train"]
        val_data = None
        test_data = ds["test"]
    elif "train" in ds:
        all_data = ds["train"]
        val_data = None
        test_data = None
    else:
        all_data = ds[list(ds.keys())[0]]
        val_data = None
        test_data = None

    # Extract images and labels
    # Handle different column name conventions
    image_col = None
    label_col = None
    
    for col in all_data.column_names:
        if col.lower() in ["image", "img", "pixel_values"]:
            image_col = col
        if col.lower() in ["label", "labels", "dx", "target", "class"]:
            label_col = col

    if image_col is None or label_col is None:
        logger.info(f"Available columns: {all_data.column_names}")
        # Fallback: use first image-type column and first label column
        for col in all_data.column_names:
            if all_data.features[col].dtype == "int64" or "class" in str(all_data.features[col]):
                label_col = col
            elif "image" in str(all_data.features[col]).lower():
                image_col = col
        
        if image_col is None:
            image_col = all_data.column_names[0]
        if label_col is None:
            label_col = all_data.column_names[-1]

    logger.info(f"Using columns: image='{image_col}', label='{label_col}'")

    # For smoke test, use tiny subset
    if smoke_test:
        all_data = all_data.select(range(min(200, len(all_data))))
        if val_data is not None:
            val_data = val_data.select(range(min(50, len(val_data))))
        if test_data is not None:
            test_data = test_data.select(range(min(50, len(test_data))))

    images_all = all_data[image_col]
    labels_all = all_data[label_col]

    # Convert string labels to integers if needed
    label_to_idx = None
    if isinstance(labels_all[0], str):
        unique_labels = sorted(set(labels_all))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        labels_all = [label_to_idx[l] for l in labels_all]
        logger.info(f"Label mapping: {label_to_idx}")
    else:
        labels_all = list(labels_all)

    num_classes = len(set(labels_all))
    logger.info(f"Total samples: {len(labels_all)}, Classes: {num_classes}")

    # Compute class distribution
    class_counts = Counter(labels_all)
    logger.info(f"Class distribution: {dict(sorted(class_counts.items()))}")

    # Stratified train/val/test split
    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    test_ratio = cfg["data"]["test_split"]

    if val_data is not None and test_data is not None:
        # All 3 splits available from HF — use them directly (best case)
        images_train = images_all
        labels_train = labels_all

        images_val = val_data[image_col]
        labels_val = list(val_data[label_col])
        if isinstance(labels_val[0], str):
            labels_val = [label_to_idx[l] for l in labels_val]

        images_test = test_data[image_col]
        labels_test = list(test_data[label_col])
        if isinstance(labels_test[0], str):
            labels_test = [label_to_idx[l] for l in labels_test]

    elif test_data is not None:
        # Have train + test, need to split train → train/val
        images_test = test_data[image_col]
        labels_test = list(test_data[label_col])
        if isinstance(labels_test[0], str):
            labels_test = [label_to_idx[l] for l in labels_test]
        
        val_frac = val_ratio / (train_ratio + val_ratio)
        indices = list(range(len(labels_all)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, stratify=labels_all, random_state=cfg["project"]["seed"]
        )
        images_train = [images_all[i] for i in train_idx]
        labels_train = [labels_all[i] for i in train_idx]
        images_val = [images_all[i] for i in val_idx]
        labels_val = [labels_all[i] for i in val_idx]
    else:
        # Only one split — manually create train/val/test
        indices = list(range(len(labels_all)))
        
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, stratify=labels_all, random_state=cfg["project"]["seed"]
        )
        labels_trainval = [labels_all[i] for i in trainval_idx]

        val_frac = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac, stratify=labels_trainval, random_state=cfg["project"]["seed"]
        )

        images_train = [images_all[i] for i in train_idx]
        labels_train = [labels_all[i] for i in train_idx]
        images_val = [images_all[i] for i in val_idx]
        labels_val = [labels_all[i] for i in val_idx]
        images_test = [images_all[i] for i in test_idx]
        labels_test = [labels_all[i] for i in test_idx]

    logger.info(f"Split sizes — Train: {len(labels_train)}, Val: {len(labels_val)}, Test: {len(labels_test)}")

    # Compute class weights for loss function (inverse frequency)
    train_counts = Counter(labels_train)
    total = sum(train_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * train_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    )
    # Normalize so mean = 1
    class_weights = class_weights / class_weights.mean()
    logger.info(f"Class weights: {class_weights.tolist()}")

    return {
        "train": (images_train, labels_train),
        "val": (images_val, labels_val),
        "test": (images_test, labels_test),
        "class_weights": class_weights,
        "num_classes": num_classes,
    }


def get_sampler(labels, num_classes):
    """Create WeightedRandomSampler for class-balanced training."""
    class_counts = Counter(labels)
    total = len(labels)
    weights_per_class = {c: total / count for c, count in class_counts.items()}
    sample_weights = [weights_per_class[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_dataloaders(config_path, smoke_test=False):
    """
    Build DataLoaders from config.
    Returns dict with train/val/test DataLoaders + class_weights + num_classes.
    """
    # Load config
    if isinstance(config_path, str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config_path

    # Load dataset
    data = load_hf_dataset(cfg, smoke_test=smoke_test)

    # Create transforms
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Create datasets
    train_ds = SkinLesionDataset(
        data["train"][0], data["train"][1],
        transform=train_transform,
        class_names=cfg["data"].get("class_names", LABEL_NAMES)
    )
    val_ds = SkinLesionDataset(
        data["val"][0], data["val"][1],
        transform=val_transform,
        class_names=cfg["data"].get("class_names", LABEL_NAMES)
    )
    test_ds = SkinLesionDataset(
        data["test"][0], data["test"][1],
        transform=val_transform,
        class_names=cfg["data"].get("class_names", LABEL_NAMES)
    )

    # Sampler for class-balanced training
    train_sampler = get_sampler(data["train"][1], data["num_classes"])

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # larger batch for eval (no grads)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": data["class_weights"],
        "num_classes": data["num_classes"],
    }
