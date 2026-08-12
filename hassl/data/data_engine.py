import os
import glob
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from monai.data import PersistentDataset, Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Resized,
    ScaleIntensityRangePercentilesd
)


def get_or_create_frozen_splits(data_dir: str, image_suffix: str = ".mha", label_suffix: str = ".seg.nrrd", seed: int = 42) -> Dict[str, List[str]]:
    """Create or load a frozen split file (splits.json) with patient/volume-level holdouts (C-4 fix).

    Ensures:
      - test_ids: Frozen test set never queried or trained on.
      - val_ids: Fixed validation set across all AL rounds (comparable Dice).
      - initial_train_ids: Remaining initial labeled IDs for Round 0.
    """
    splits_file = Path(data_dir) / "splits.json"
    if splits_file.exists():
        with open(splits_file, "r") as f:
            return json.load(f)

    # Scan for initial labeled pairs
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    all_labeled_ids = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")
        lbl_path = str(Path(img_path).parent / f"{base_name}{label_suffix}")
        if not os.path.exists(lbl_path):
            lbl_path = str(data_dir_path / "labels" / f"{base_name}{label_suffix}")
        if os.path.exists(lbl_path):
            all_labeled_ids.append(base_name)

    # Seeded deterministic shuffle
    rng = random.Random(seed)
    shuffled_ids = list(all_labeled_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_test = max(1, int(n_total * 0.15)) if n_total >= 5 else 0
    n_val = max(1, int(n_total * 0.15)) if n_total >= 5 else 1

    test_ids = shuffled_ids[:n_test]
    val_ids = shuffled_ids[n_test:n_test + n_val]
    train_ids = shuffled_ids[n_test + n_val:]

    splits = {
        "test_ids": test_ids,
        "val_ids": val_ids,
        "initial_train_ids": train_ids,
        "seed": seed,
    }

    os.makedirs(data_dir, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"[DataEngine] Created frozen splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    return splits


def build_labeled_dataset(data_dir: str, image_suffix: str, label_suffix: str,
                          include_ids: Optional[List[str]] = None,
                          transform=None, cache_dir: Optional[str] = None):
    """Build dataset for labeled volumes (human + approved pseudo)."""
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    data_dicts = []
    labeled_ids = set()

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")

        if include_ids is not None and base_name not in include_ids:
            continue

        # Check in human labels directory first, then same parent
        lbl_path = str(data_dir_path / "labels" / f"{base_name}{label_suffix}")
        if not os.path.exists(lbl_path):
            lbl_path = str(Path(img_path).parent / f"{base_name}{label_suffix}")

        if os.path.exists(lbl_path):
            data_dicts.append({
                "image": img_path,
                "label": lbl_path,
                "id": base_name,
                "provenance": "human",
            })
            labeled_ids.add(base_name)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir), labeled_ids
    else:
        return Dataset(data=data_dicts, transform=transform), labeled_ids


def build_unlabeled_dataset(data_dir: str, image_suffix: str, labeled_ids: set,
                           exclude_ids: Optional[set] = None,
                           transform=None, cache_dir: Optional[str] = None):
    """Build dataset for unlabeled volumes, excluding labeled and frozen test IDs."""
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    exclude_set = set(labeled_ids)
    if exclude_ids:
        exclude_set.update(exclude_ids)

    data_dicts = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")
        if base_name not in exclude_set:
            data_dicts.append({
                "image": img_path,
                "id": base_name,
            })

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir)
    else:
        return Dataset(data=data_dicts, transform=transform)


def get_base_transforms(config, keys=["image", "label"]):
    """Build MONAI preprocessing transform chain."""
    mode = tuple(["bilinear" if k == "image" else "nearest" for k in keys])
    resize_mode = tuple(["trilinear" if k == "image" else "nearest" for k in keys])

    transforms = [
        LoadImaged(keys=keys, reader="ITKReader", image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=config.spacing, mode=mode),
        Resized(keys=keys, spatial_size=config.spatial_size, mode=resize_mode),
    ]
    if "image" in keys:
        transforms.append(ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True))

    return Compose(transforms)


def build_dataloaders(config):
    """Build train, unlabeled, and FIXED validation dataloaders (C-4 fix)."""
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
    )

    val_ids_set = set(splits.get("val_ids", []))
    test_ids_set = set(splits.get("test_ids", []))

    labeled_transforms = get_base_transforms(config, keys=["image", "label"])
    unlabeled_transforms = get_base_transforms(config, keys=["image"])

    cache_dir = getattr(config, 'cache_dir', None)

    # 1. Validation dataset (FIXED across all rounds!)
    val_ds, _ = build_labeled_dataset(
        config.data_dir,
        config.image_suffix,
        config.label_suffix,
        include_ids=splits.get("val_ids"),
        transform=labeled_transforms,
        cache_dir=cache_dir,
    )

    # 2. Labeled Training dataset (excludes val & test)
    data_dir_path = Path(config.data_dir)
    all_image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    # All labeled IDs currently available except val & test
    available_train_labeled = []
    for img_path in all_image_files:
        base_name = os.path.basename(img_path).replace(config.image_suffix, "")
        if base_name in val_ids_set or base_name in test_ids_set:
            continue
        lbl_path = str(data_dir_path / "labels" / f"{base_name}{config.label_suffix}")
        if not os.path.exists(lbl_path):
            lbl_path = str(Path(img_path).parent / f"{base_name}{config.label_suffix}")
        if os.path.exists(lbl_path):
            available_train_labeled.append(base_name)

    train_ds, train_labeled_ids = build_labeled_dataset(
        config.data_dir,
        config.image_suffix,
        config.label_suffix,
        include_ids=available_train_labeled,
        transform=labeled_transforms,
        cache_dir=cache_dir,
    )

    # 3. Unlabeled dataset (excludes all labeled + val + test)
    all_known_labeled = set(train_labeled_ids).union(val_ids_set).union(test_ids_set)
    unlabeled_ds = build_unlabeled_dataset(
        config.data_dir,
        config.image_suffix,
        labeled_ids=all_known_labeled,
        exclude_ids=test_ids_set,
        transform=unlabeled_transforms,
        cache_dir=cache_dir,
    )

    num_workers = getattr(config, 'num_workers', 2)
    batch_size = getattr(config, 'batch_size', 1)

    labeled_loader = MonaiDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(train_ds) > 0 else None
    val_loader = MonaiDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(val_ds) > 0 else None
    unlabeled_loader = MonaiDataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(unlabeled_ds) > 0 else None

    return labeled_loader, unlabeled_loader, val_loader


def build_all_volumes_loader(config):
    """Build a dataloader for ALL volumes (excluding test set), without labels for SSL."""
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
    )
    test_ids_set = set(splits.get("test_ids", []))

    unlabeled_transforms = get_base_transforms(config, keys=["image"])

    data_dir_path = Path(config.data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    data_dicts = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(config.image_suffix, "")
        if base_name not in test_ids_set:
            data_dicts.append({
                "image": img_path,
                "id": base_name,
            })

    cache_dir = getattr(config, 'cache_dir', None)
    if cache_dir is not None:
        ssl_cache = os.path.join(cache_dir, "ssl")
        os.makedirs(ssl_cache, exist_ok=True)
        dataset = PersistentDataset(data=data_dicts, transform=unlabeled_transforms, cache_dir=ssl_cache)
    else:
        dataset = Dataset(data=data_dicts, transform=unlabeled_transforms)

    return MonaiDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
