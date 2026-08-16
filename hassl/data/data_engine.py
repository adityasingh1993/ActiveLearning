import os
import glob
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

from hassl.compat import ORIENTATIOND_RAS_LABELS

logger = logging.getLogger(__name__)


def _strip_suffix(filename: str, suffix: str) -> str:
    """Strip *suffix* from *filename* only if it appears at the trailing end.

    ``str.replace(suffix, "")`` removes ALL occurrences of the suffix, which
    corrupts volume IDs derived from filenames that contain the suffix string
    mid-name (e.g. 'sonoeq.transducer.frameacquasition.mha' with image_suffix
    '.mha' is fine, but a hypothetical 'scan.mha.backup.mha' would become
    'scan.backup' with replace, vs 'scan.mha.backup' with this helper).

    Also uses ``removesuffix`` (Python 3.9+) as the primary path, with a
    manual endswith/rstrip fallback for older Python versions.
    """
    if hasattr(filename, 'removesuffix'):
        return filename.removesuffix(suffix)
    # Fallback for Python < 3.9
    if suffix and filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename

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
    AsDiscreted,
    SpatialPadd,
    MapTransform,
)

from .augmentations import get_strong_augmentation
from .label_utils import NormalizeLabelsInDatasetd


class AspectRatioResizeWithPadd(MapTransform):
    """Resize to fit within spatial_size preserving aspect ratio, then zero-pad to spatial_size.

    Used when Spacingd is skipped (micro-spacing < 0.1mm guard). In that case, bare Resized
    squishes e.g. a 1200x800x400 ultrasound volume isotropically into 128x128x128, distorting
    anatomy and teaching the model incorrect shape priors for the target structure.

    This transform instead:
      1. Computes scale = min(target_dim / current_dim) across all spatial axes.
      2. Resizes to (D*scale, H*scale, W*scale) — all dims <= spatial_size.
      3. Symmetrically zero-pads the remaining space to reach spatial_size exactly.

    When Spacingd IS active (spacing >= 0.1mm), the existing bare Resized is used as before
    because Spacingd already normalises the voxel pitch to a uniform grid.
    """

    def __init__(self, keys, spatial_size, mode, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = tuple(spatial_size)
        self._mode_list = [mode] if isinstance(mode, str) else list(mode)

    def __call__(self, data):
        d = dict(data)
        active_keys = [k for k in self.key_iterator(d)]
        if not active_keys:
            return d

        # Derive scale from the first key's spatial shape (all keys are spatially aligned)
        ref = d[active_keys[0]]
        current_spatial = ref.shape[1:]           # drop channel dim -> (D, H, W)
        ndim = len(current_spatial)
        target = self.spatial_size[:ndim]

        # Scale by the axis that needs to shrink the most so nothing overflows
        scale = min(target[i] / max(int(current_spatial[i]), 1) for i in range(ndim))
        scaled = tuple(
            min(max(int(round(current_spatial[i] * scale)), 1), target[i])
            for i in range(ndim)
        )

        # Resize each key with its own interpolation mode
        for idx, key in enumerate(active_keys):
            mode_k = self._mode_list[idx] if idx < len(self._mode_list) else self._mode_list[-1]
            d = Resized(keys=[key], spatial_size=scaled, mode=mode_k)(d)

        # Symmetrically zero-pad all keys to the final target size
        d = SpatialPadd(keys=active_keys, spatial_size=list(target), method="symmetric", mode="constant")(d)
        return d


def get_or_create_frozen_splits(
    data_dir: str,
    image_suffix: str = ".mha",
    label_suffix: str = ".seg.nrrd",
    seed: int = 42,
    patient_id_regex: Optional[str] = None,
    val_split: int = 5,
) -> Dict[str, List[str]]:
    """Create or load a frozen split file (splits.json) with patient-level holdouts (C-4, V-11 fix).

    Args:
        val_split: Target number of patients to hold out for validation.
                   Previously this was hardcoded to 15% of total patients, which on a
                   10-patient dataset gives only 1 validation patient — erratic Dice
                   scores, unreliable early stopping, and biased best-model selection.
                   Now respects config.val_split (default 5), clamped so at least 1
                   training patient always remains.
    """
    splits_file = Path(data_dir) / "splits.json"
    if splits_file.exists():
        with open(splits_file, "r") as f:
            return json.load(f)

    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))

    patient_map = {}
    for img_path in image_files:
        base_name = _strip_suffix(os.path.basename(img_path), image_suffix)
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

    # Test split: fixed at 15% (not configurable — held out for final evaluation only)
    n_test_p = max(1, int(n_total_p * 0.15)) if n_total_p >= 5 else 0

    # Validation split: respect config.val_split, clamped so at least 1 training patient remains
    remaining_after_test = n_total_p - n_test_p
    n_val_p = min(val_split, max(0, remaining_after_test - 1))  # leave at least 1 train patient
    if n_total_p < 5:
        n_val_p = 1 if n_total_p > 1 else 0

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

    print(f"[DataEngine] Created patient-level frozen splits: {len(train_ids)} train, {len(val_ids)} val (target={val_split}), {len(test_ids)} test")
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
        base_name = _strip_suffix(os.path.basename(img_path), image_suffix)

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
        base_name = _strip_suffix(os.path.basename(img_path), image_suffix)
        if base_name not in exclude_set:
            data_dicts.append({
                "image": img_path,
                "id": base_name,
            })

    if use_cache_dataset and len(data_dicts) > 0:
        try:
            return CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, copy_cache=False), labeled_ids
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
    - "patch": Spacingd → training crop for train data, while validation keeps the
      full post-Spacingd volume and uses SlidingWindowInferer at patch_size. This keeps
      train and validation windows at the same physical voxel scale.
    """
    mode = tuple(["bilinear" if k == "image" else "nearest" for k in keys])
    resize_mode = tuple(["trilinear" if k == "image" else "nearest" for k in keys])
    preprocessing_mode = getattr(config, 'preprocessing_mode', 'resize')

    # Guard against corrupted or ultra-fine spacing values (< 0.1 mm) that cause 300GB+ OOM allocation in Spacingd
    use_spacingd = False
    if config.spacing and all(s >= 0.1 for s in config.spacing):
        use_spacingd = True
    else:
        logger.warning(
            "[HASSL DataEngine] Skipping Spacingd because config.spacing %s contains micro-spacing values (<0.1mm) "
            "that cause 300GB+ OOM memory allocations. Relying on Resized(spatial_size=%s) for safe spatial normalization.",
            getattr(config, 'spacing', None), getattr(config, 'spatial_size', (128, 128, 128))
        )

    transforms = [
        # image_only=True: metadata (affine, transform history) is embedded in the
        # MetaTensor rather than added as a separate '{filename_stem}' dict key.
        # image_only=False caused MONAI to inject the file's basename as an extra key
        # (e.g. 'SonoEQ.Transducer.frame_acquisition') into every data dict. When
        # batching two items with different filenames, list_data_collate tries to access
        # item_0's filename key from item_1's dict → KeyError in DataLoader worker 0.
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=ORIENTATIOND_RAS_LABELS),
    ]
    if use_spacingd:
        transforms.append(Spacingd(keys=keys, pixdim=config.spacing, mode=mode))

    # Label normalisation — applied after Spacingd (if any) so resampling happens first.
    if "label" in keys:
        num_classes = getattr(config, 'num_classes', 1)
        label_names = getattr(config, 'label_names', None)
        if num_classes == 1:
            # Binary: threshold to {0, 1} (handles NRRD files where foreground = 255 etc.)
            transforms.append(AsDiscreted(keys=["label"], threshold=0.5))
        elif label_names and isinstance(label_names, dict):
            # Multi-class with explicit label map: remap non-contiguous values to 0,1,2,...,N
            # e.g. label_names={0:'bg', 1:'bladder', 3:'urethra', 7:'sphincter'} -> 0,1,2,3
            transforms.append(NormalizeLabelsInDatasetd(keys=["label"], label_names=label_names))
        # else: multi-class without label_names — trust the NRRD values are already contiguous

    # Intensity normalisation (image only) — placed before spatial crop so
    # crop statistics reflect the normalised distribution.
    # channel_wise=True prevents divide-by-zero crash when a single channel is
    # fully saturated (flat intensity), which is common in ultrasound background slices.
    if "image" in keys:
        transforms.append(ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True,
            channel_wise=True
        ))

    if preprocessing_mode == "patch":
        patch_size = getattr(config, 'patch_size', (96, 96, 96))
        pos_neg_ratio = getattr(config, 'pos_neg_ratio', 1.0)

        # Both train and validation must be at least patch_size so the training crop and
        # SlidingWindowInferer can operate safely on small volumes. Validation is padded
        # only; it is NOT globally resized, preserving the post-Spacingd physical scale.
        transforms.append(SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric", mode="constant"))

        if is_training:
            if "label" in keys:
                # Labeled stream: crop centred on positive voxels with explicit negatives.
                transforms.append(RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="label",
                    spatial_size=patch_size,
                    pos=pos_neg_ratio,
                    neg=1.0,
                    num_samples=1,
                ))
            else:
                # Unlabeled / image-only stream: random spatial crop.
                transforms.append(RandSpatialCropd(
                    keys=keys,
                    roi_size=patch_size,
                    random_size=False,
                ))
    else:
        # Resize mode: spatial normalisation to a fixed whole-volume grid.
        # When Spacingd ran (use_spacingd=True): voxel pitch is already uniform so isotropic
        # resize is appropriate. When Spacingd was skipped (micro-spacing), preserve aspect ratio.
        if use_spacingd:
            transforms.append(Resized(keys=keys, spatial_size=config.spatial_size, mode=resize_mode))
        else:
            transforms.append(AspectRatioResizeWithPadd(
                keys=keys, spatial_size=config.spatial_size, mode=resize_mode
            ))

    if is_training and apply_strong_aug:
        transforms.append(get_strong_augmentation(keys=keys))

    return Compose(transforms)


def build_dataloaders(config):
    """Build train, unlabeled, and FIXED validation dataloaders with asymmetric transforms (C-4, N-5, P-1 fix)."""
    patient_id_regex = getattr(config, 'patient_id_regex', None)
    val_split = getattr(config, 'val_split', 5)
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=patient_id_regex,
        val_split=val_split,
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

    # 1. Validation dataset — FIXED across all rounds AND uses plain Dataset (not CacheDataset).
    # Reason: CacheDataset pre-runs transforms and bakes the output tensors into RAM. The cached
    # tensors' MetaTensor.applied_operations may be stale or empty because they were recorded
    # against a now-dead transform instance. Invertd then cannot replay the chain → exception.
    # Using plain Dataset ensures transforms run live each iteration, so applied_operations
    # always contains fresh entries that match self.val_transform.
    val_data_dicts: list = []
    data_dir_path_val = Path(config.data_dir)
    for img_path in sorted(glob.glob(str(data_dir_path_val / f"**/*{config.image_suffix}"), recursive=True)):
        base_name = _strip_suffix(os.path.basename(img_path), config.image_suffix)
        if base_name not in set(splits.get("val_ids", [])):
            continue
        lbl_path = str(data_dir_path_val / "labels" / f"{base_name}{config.label_suffix}")
        if not os.path.exists(lbl_path):
            lbl_path = str(Path(img_path).parent / f"{base_name}{config.label_suffix}")
        if os.path.exists(lbl_path):
            val_data_dicts.append({"image": img_path, "label": lbl_path, "id": base_name, "provenance": "human"})

    val_ds = Dataset(data=val_data_dicts, transform=val_transforms)

    # 2. Labeled Training dataset (excludes val & test)
    data_dir_path = Path(config.data_dir)
    all_image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    available_train_labeled = []
    for img_path in all_image_files:
        base_name = _strip_suffix(os.path.basename(img_path), config.image_suffix)
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
    if os.name == 'nt' and getattr(config, 'compute_mode', 'prototype') == 'prototype':
        num_workers = 0  # 0 workers on Windows prototype mode prevents multiprocess IPC memory leaks/crashes
    batch_size = getattr(config, 'batch_size', 1)

    labeled_loader = MonaiDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(train_ds) > 0 else None
    # val_loader: batch_size=1, num_workers=0 (plain Dataset, no workers needed; avoids multiprocess fork overhead)
    val_loader = MonaiDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0) if len(val_ds) > 0 else None
    unlabeled_loader = MonaiDataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(unlabeled_ds) > 0 else None

    # Return val_transforms so callers can pass the EXACT same Compose instance to Invertd.
    # Invertd matches transforms by object identity; a separately-constructed Compose fails.
    return labeled_loader, unlabeled_loader, val_loader, val_transforms


def build_all_volumes_loader(config):
    """Build a dataloader for ALL volumes (excluding test set), without labels for SSL."""
    patient_id_regex = getattr(config, 'patient_id_regex', None)
    val_split = getattr(config, 'val_split', 5)
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=patient_id_regex,
        val_split=val_split,
    )
    test_ids_set = set(splits.get("test_ids", []))

    unlabeled_transforms = get_base_transforms(config, keys=["image"], is_training=False)
    data_dir_path = Path(config.data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    data_dicts = []
    for img_path in image_files:
        base_name = _strip_suffix(os.path.basename(img_path), config.image_suffix)
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