import os
import glob
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

class SemiSupervisedDataLoader:
    """Semi-supervised loader that alternates/combines labeled and unlabeled batches"""
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __iter__(self):
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        
        while True:
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_batch = next(labeled_iter)
                
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                break
                
            yield labeled_batch, unlabeled_batch

    def __len__(self):
        return len(self.unlabeled_loader)

def build_labeled_dataset(data_dir: str, image_suffix: str, label_suffix: str, transform=None, cache_dir: Optional[str] = None):
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))
    
    data_dicts = []
    labeled_ids = set()
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")
        
        # Searching for corresponding label
        lbl_path = str(Path(img_path).parent / f"{base_name}{label_suffix}")
        if os.path.exists(lbl_path):
            data_dicts.append({
                "image": img_path,
                "label": lbl_path,
                "id": base_name
            })
            labeled_ids.add(base_name)
            
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir), labeled_ids
    else:
        return Dataset(data=data_dicts, transform=transform), labeled_ids

def build_unlabeled_dataset(data_dir: str, image_suffix: str, labeled_ids: set, transform=None, cache_dir: Optional[str] = None):
    data_dir_path = Path(data_dir)
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))
    
    data_dicts = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(image_suffix, "")
        if base_name not in labeled_ids:
            data_dicts.append({
                "image": img_path,
                "id": base_name
            })
            
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir)
    else:
        return Dataset(data=data_dicts, transform=transform)

def get_base_transforms(config, keys=["image", "label"]):
    # Base transforms using MONAI
    mode = tuple(["bilinear" if k == "image" else "nearest" for k in keys])
    resize_mode = tuple(["trilinear" if k == "image" else "nearest" for k in keys])
    
    transforms = [
        LoadImaged(keys=keys, reader="ITKReader", image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=config.spacing, mode=mode),
        Resized(keys=keys, spatial_size=config.spatial_size, mode=resize_mode)
    ]
    if "image" in keys:
        transforms.append(ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True))
        
    return Compose(transforms)

def build_dataloaders(config):
    labeled_transforms = get_base_transforms(config, keys=["image", "label"])
    unlabeled_transforms = get_base_transforms(config, keys=["image"])
    
    cache_dir = getattr(config, 'cache_dir', None)
    
    labeled_ds, labeled_ids = build_labeled_dataset(
        config.data_dir, 
        config.image_suffix, 
        config.label_suffix, 
        transform=labeled_transforms,
        cache_dir=cache_dir
    )
    
    unlabeled_ds = build_unlabeled_dataset(
        config.data_dir,
        config.image_suffix,
        labeled_ids,
        transform=unlabeled_transforms,
        cache_dir=cache_dir
    )
    
    # Validation split
    val_size = min(5, len(labeled_ds))
    train_size = len(labeled_ds) - val_size
    
    if len(labeled_ds) > 0:
        train_ds, val_ds = torch.utils.data.random_split(labeled_ds, [train_size, val_size])
    else:
        train_ds, val_ds = labeled_ds, labeled_ds
        
    num_workers = getattr(config, 'num_workers', 4)
    batch_size = getattr(config, 'batch_size', 2)
    
    labeled_loader = MonaiDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(train_ds) > 0 else None
    val_loader = MonaiDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_size > 0 else None
    unlabeled_loader = MonaiDataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) if len(unlabeled_ds) > 0 else None
    
    return labeled_loader, unlabeled_loader, val_loader


def build_all_volumes_loader(config):
    """Build a dataloader for ALL volumes (labeled + unlabeled), without labels.

    Used for SSL pre-training where we need all volumes but no labels.

    Args:
        config: HASSLConfig with data_dir, image_suffix, spatial_size, etc.

    Returns:
        DataLoader over all volumes with 'image' and 'id' keys.
    """
    unlabeled_transforms = get_base_transforms(config, keys=["image"])

    data_dir_path = Path(config.data_dir)
    # Search in both root and images/ subdirectory
    image_files = sorted(glob.glob(str(data_dir_path / f"**/*{config.image_suffix}"), recursive=True))

    data_dicts = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(config.image_suffix, "")
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

