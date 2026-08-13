import os
import glob
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

from monai.data import CacheDataset, PersistentDataset, Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Resized,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    AsDiscreted
)

from .augmentations import get_strong_augmentation


def get_or_create_frozen_splits(data_dir: str, image_suffix: str = ".mha", label_suffix: str = ".seg.nrrd", seed: int = 42, patient_id_regex: Optional[str] = None) -> Dict[str, List[str]]:
    """Create or load a frozen split file (splits.json) with patient-level holdouts (C-4, V-11 fix)."""
    splits_file = Path(data_dir) / "splits.json"
    if splits_file.exists():
        with open(splits_file, "r") as f:
            return json.load(f)

    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    patient_map = {}
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")
        lbl_path = str(data_dir_path / "labels" / f"{base_name}{label_suffix}")
        if not os.path.exists(lbl_path):
            lbl_path = str(Path(img_path).parent / f"{base_name}{label_suffix}")
        if os.path.exists(lbl_path):
            # Patient-level grouping (C-4, V-11 fix): extract patient ID prefix using regex if supplied
            if patient_id_regex:
                match = re.search(patient_id_regex, base_name)
                patient_id = match.group(1) if match else base_name
            else:
                patient_id = base_name.split('_')[0] if '_' in base_name else base_name

            if patient_id not in patient_map:
                patient_map[patient_id] = []
            patient_map[patient_id].append(base_name)

    patient_ids = sorted(list(patient_map.keys()))
    n_total_vols = sum(len(vols) for vols in patient_map.values())
    n_total_p = len(patient_ids)

    # V-11 fix: Assert on patient grouping heuristic failure
    if n_total_vols >= 5:
        if n_total_p == 1:
            raise ValueError(f"[DataEngine Error] Patient extraction collapsed all {n_total_vols} volumes into 1 single patient ID '{patient_ids[0]}'. Check filename structure or set config.patient_id_regex.")
        elif n_total_p == n_total_vols:
            print(f"[DataEngine Warning] Patient ID heuristic found {n_total_p} patients for {n_total_vols} volumes (1:1 mapping). Ensure volumes are independent subjects.")

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_total_p = len(patient_ids)
    n_test_p = max(1, int(n_total_p * 0.15)) if n_total_p >= 5 else 0
    n_val_p = max(1, int(n_total_p * 0.15)) if n_total_p >= 5 else (1 if n_total_p > 1 else 0)

    test_patients = set(patient_ids[:n_test_p])
    val_patients = set(patient_ids[n_test_p:n_test_p + n_val_p])

    test_ids = [vid for pid in test_patients for vid in patient_map[pid]]
    val_ids = [vid for pid in val_patients for vid in patient_map[pid]]
    train_ids = [vid for pid in patient_ids[n_test_p + n_val_p:] for vid in patient_map[pid]]

    splits = {
        "test_ids": test_ids,
        "val_ids": val_ids,
        "initial_train_ids": train_ids,
        "seed": seed,
    }

    os.makedirs(data_dir, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"[DataEngine] Created patient-level frozen splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    return splits


def build_labeled_dataset(data_dir: str, image_suffix: str, label_suffix: str,
                          include_ids: Optional[List[str]] = None,
                          transform=None, cache_dir: Optional[str] = None,
                          manifest_path: Optional[str] = None,
                          use_cache_dataset: bool = True):
    """Build dataset for labeled volumes with strict manifest provenance gating (P-1 fix) & CacheDataset RAM caching."""
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    manifest_provenance = {}
    m_path = manifest_path or str(data_dir_path.parent / "logs" / "pool_manifest.json")
    if not os.path.exists(m_path):
        m_path = "./experiments/logs/pool_manifest.json"
    if os.path.exists(m_path):
        try:
            with open(m_path, 'r') as f:
                state = json.load(f)
                manifest_provenance = state.get('provenance', {})
        except Exception:
            pass

    data_dicts = []
    labeled_ids = set()

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")

        if include_ids is not None and base_name not in include_ids:
            continue

        lbl_path = str(data_dir_path / "labels" / f"{base_name}{label_suffix}")
        provenance = "human"

        # Check in human labels directory first
        if not os.path.exists(lbl_path):
            lbl_path = str(Path(img_path).parent / f"{base_name}{label_suffix}")

        # P-1, V6-2 & V6-3 fix: Check in approved pseudo-labels directory ONLY if manifest marks it pseudo_approved or human_corrected
        if not os.path.exists(lbl_path):
            prov_status = manifest_provenance.get(base_name)
            if prov_status in ["pseudo_approved", "human_corrected"]:
                pseudo_candidate = str(data_dir_path / "pseudo_approved" / f"{base_name}{label_suffix}")
                if not os.path.exists(pseudo_candidate):
                    pseudo_candidate = str(data_dir_path / "pseudo" / f"{base_name}{label_suffix}")
                if os.path.exists(pseudo_candidate):
                    lbl_path = pseudo_candidate
                    provenance = prov_status

        if os.path.exists(lbl_path):
            data_dicts.append({
                "image": img_path,
                "label": lbl_path,
                "id": base_name,
                "provenance": provenance,
            })
            labeled_ids.add(base_name)

    if use_cache_dataset and len(data_dicts) > 0:
        try:
            return CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, copy_cache=False), labeled_ids
        except Exception as e:
            print(f"[DataEngine Warning] CacheDataset failed ({type(e).__name__}: {e}). Falling back to PersistentDataset/Dataset.")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir), labeled_ids
    else:
        return Dataset(data=data_dicts, transform=transform), labeled_ids


def build_unlabeled_dataset(data_dir: str, image_suffix: str, labeled_ids: set,
                           exclude_ids: Optional[set] = None,
                           transform=None, cache_dir: Optional[str] = None,
                           use_cache_dataset: bool = True):
    """Build dataset for unlabeled volumes with CacheDataset RAM caching."""
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

    if use_cache_dataset and len(data_dicts) > 0:
        try:
            return CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, copy_cache=False)
        except Exception as e:
            print(f"[DataEngine Warning] CacheDataset failed ({type(e).__name__}: {e}). Falling back to PersistentDataset/Dataset.")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir)
    else:
        return Dataset(data=data_dicts, transform=transform)


def get_base_transforms(config, keys=["image", "label"], is_training: bool = False, apply_strong_aug: bool = True):
    """Build MONAI preprocessing transform chain.

    Supports two strategies controlled by config.preprocessing_mode:
    - "resize" (default): Spacingd → Resized(spatial_size). Whole-volume fixed-grid pipeline.
    - "patch": Spacingd only, then RandCropByPosNegLabeld (labeled training) or
      RandSpatialCropd (unlabeled/image-only training). Validation always uses
      Resized regardless of mode for OOM-safe whole-volume inference.
    """
    mode = tuple(["bilinear" if k == "image" else "nearest" for k in keys])
    resize_mode = tuple(["trilinear" if k == "image" else "nearest" for k in keys])
    preprocessing_mode = getattr(config, 'preprocessing_mode', 'resize')

    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=config.spacing, mode=mode),
    ]

    # Normalize label values (e.g. 255 -> 1) to clean binary masks
    if "label" in keys and getattr(config, 'num_classes', 1) == 1:
        transforms.append(AsDiscreted(keys=["label"], threshold=0.5))

    # Intensity normalisation (image only) — placed before spatial crop so
    # crop statistics reflect the normalised distribution.
    if "image" in keys:
        transforms.append(ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True
        ))

    if preprocessing_mode == "patch" and is_training:
        # Patch mode: random crop instead of resize.
        # Validation always falls through to Resized (below) for consistent
        # whole-volume inference with SlidingWindowInferer.
        patch_size = getattr(config, 'patch_size', (96, 96, 96))
        pos_neg_ratio = getattr(config, 'pos_neg_ratio', 1.0)
        if "label" in keys:
            # Labeled stream: crop centred on positive voxels
            transforms.append(RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=patch_size,
                pos=pos_neg_ratio,
                neg=1.0,
                num_samples=1,
            ))
        else:
            # Unlabeled / image-only stream: random spatial crop
            transforms.append(RandSpatialCropd(
                keys=keys,
                roi_size=patch_size,
                random_size=False,
            ))
    else:
        # "resize" mode (default), or validation in any mode
        transforms.append(Resized(keys=keys, spatial_size=config.spatial_size, mode=resize_mode))

    if is_training and apply_strong_aug:
        transforms.append(get_strong_augmentation(keys=keys))

    return Compose(transforms)


def build_dataloaders(config):
    """Build train, unlabeled, and FIXED validation dataloaders with asymmetric transforms (C-4, N-5, P-1 fix)."""
    patient_id_regex = getattr(config, 'patient_id_regex', None)
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=patient_id_regex,
    )

    val_ids_set = set(splits.get("val_ids", []))
    test_ids_set = set(splits.get("test_ids", []))

    train_transforms = get_base_transforms(config, keys=["image", "label"], is_training=True)
    val_transforms = get_base_transforms(config, keys=["image", "label"], is_training=False)

    # R12 H-2 fix: Pass apply_strong_aug=False for unlabeled stream dataloader
    # Strong augmentations are applied on-the-fly in trainer.py (_make_unlabeled_views).
    unlabeled_transforms = get_base_transforms(
        config, keys=["image"], is_training=True, apply_strong_aug=False
    )

    cache_dir = getattr(config, 'cache_dir', None)

    # 1. Validation dataset (FIXED across all rounds!)
    val_ds, _ = build_labeled_dataset(
        config.data_dir,
        config.image_suffix,
        config.label_suffix,
        include_ids=splits.get("val_ids"),
        transform=val_transforms,
        cache_dir=cache_dir,
    )

    # 2. Labeled Training dataset (excludes val & test)
    data_dir_path = Path(config.data_dir)
    all_image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    available_train_labeled = []
    for img_path in all_image_files:
        base_name = os.path.basename(img_path).replace(config.image_suffix, "")
        if base_name in val_ids_set or base_name in test_ids_set:
            continue
        available_train_labeled.append(base_name)

    train_ds, train_labeled_ids = build_labeled_dataset(
        config.data_dir,
        config.image_suffix,
        config.label_suffix,
        include_ids=available_train_labeled,
        transform=train_transforms,
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
    patient_id_regex = getattr(config, 'patient_id_regex', None)
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=patient_id_regex,
    )
    test_ids_set = set(splits.get("test_ids", []))

    unlabeled_transforms = get_base_transforms(config, keys=["image"], is_training=False)

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

    use_cache_dataset = getattr(config, 'use_cache_dataset', True)
    cache_dir = getattr(config, 'cache_dir', None)

    if use_cache_dataset and len(data_dicts) > 0:
        try:
            dataset = CacheDataset(data=data_dicts, transform=unlabeled_transforms, cache_rate=1.0, copy_cache=False, num_workers=config.num_workers)
        except Exception as e:
            print(f"[DataEngine Warning] SSL CacheDataset failed ({type(e).__name__}: {e}). Falling back to PersistentDataset/Dataset.")
            if cache_dir is not None:
                ssl_cache = os.path.join(cache_dir, "ssl")
                os.makedirs(ssl_cache, exist_ok=True)
                dataset = PersistentDataset(data=data_dicts, transform=unlabeled_transforms, cache_dir=ssl_cache)
            else:
                dataset = Dataset(data=data_dicts, transform=unlabeled_transforms)
    elif cache_dir is not None:
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
