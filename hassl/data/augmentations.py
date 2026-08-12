import torch
import numpy as np
from monai.transforms import (
    Compose,
    RandomizableTransform,
    RandFlipd,
    RandRotated,
    RandAffined,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd
)


class RandMultiplicativeSpeckleNoised(RandomizableTransform):
    """Ultrasound multiplicative Rayleigh speckle noise transform (N-6 fix)."""

    def __init__(self, keys=["image"], prob=0.5, std=0.08):
        super().__init__(prob=prob)
        self.keys = keys
        self.std = std

    def randomize(self, data=None):
        super().randomize(data)

    def __call__(self, data_dict):
        self.randomize(data_dict)
        if not self._do_transform:
            return data_dict

        data_dict = dict(data_dict)
        for k in self.keys:
            if k in data_dict:
                img = data_dict[k]
                noise_np = 1.0 + self.R.normal(0.0, self.std, size=img.shape)
                noise_tensor = torch.from_numpy(noise_np).to(dtype=img.dtype, device=img.device)
                data_dict[k] = torch.clamp(img * noise_tensor, 0.0, 1.0)
        return data_dict


def get_weak_augmentation(keys=["image", "label"]):
    """In-plane ultrasound flip (lateral axis only) + small rotation (§7 fix)."""
    return Compose([
        # In-plane lateral flips only (preserving acoustic beam propagation direction axis 0)
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(keys=keys, range_x=np.pi / 36, range_y=np.pi / 36, range_z=np.pi / 36, prob=0.5),
    ])


def get_strong_augmentation(keys=["image", "label"]):
    """Strong ultrasound augmentations: Affine + Multiplicative Speckle + Contrast (§7 fix)."""
    image_keys = [k for k in keys if k == "image"]
    return Compose([
        RandAffined(
            keys=keys, prob=0.5,
            rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
            translate_range=(5, 5, 5), scale_range=(0.05, 0.05, 0.05),
        ),
        RandMultiplicativeSpeckleNoised(keys=image_keys, prob=0.5, std=0.08),
        RandGaussianSmoothd(keys=image_keys, prob=0.3, sigma_x=(0.5, 1.2), sigma_y=(0.5, 1.2), sigma_z=(0.5, 1.2)),
        RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
        RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.7, 1.5)),
    ])


class CutMix3d:
    """3D CutMix for batched volume dictionaries."""

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
        if B < 2:
            return batch_data

        B, C, W, H, D = images.shape
        lam = np.random.beta(self.alpha, self.alpha)

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

        for key in self.keys:
            if key in batch_data:
                batch_data[key] = batch_data[key].clone()
                batch_data[key][:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = batch_data[key][indices, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]

        return batch_data
