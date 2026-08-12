import torch
import numpy as np
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd
)

def get_weak_augmentation(keys=["image", "label"]):
    """Compose of RandomFlipd (all 3 axes), small RandRotated (±5°)"""
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(keys=keys, range_x=np.pi/36, range_y=np.pi/36, range_z=np.pi/36, prob=0.5)
    ])

def get_strong_augmentation(keys=["image", "label"]):
    """Compose of RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, RandAdjustContrastd"""
    image_keys = [k for k in keys if k == "image"]
    return Compose([
        RandAffined(keys=keys, prob=0.5, rotate_range=(np.pi/18, np.pi/18, np.pi/18), 
                    translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1)),
        RandGaussianNoised(keys=image_keys, prob=0.5, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=image_keys, prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
        RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.5, 2.0))
    ])

class CutMix3d:
    """3D CutMix for volumes, works on batched dictionary."""
    def __init__(self, keys=["image", "label"], alpha=1.0, prob=0.5):
        self.keys = keys
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch_data):
        if np.random.rand() > self.prob:
            return batch_data
            
        if "image" not in batch_data:
            return batch_data
            
        images = batch_data["image"]
        B = images.shape[0]
        # Needs at least batch of 2 to cutmix
        if B < 2:
            return batch_data
            
        B, C, W, H, D = images.shape
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cut_d = int(D * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)
        
        indices = torch.randperm(B)
        
        # apply cutmix to keys
        for key in self.keys:
            if key in batch_data:
                batch_data[key] = batch_data[key].clone()
                batch_data[key][:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = batch_data[key][indices, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
            
        return batch_data
