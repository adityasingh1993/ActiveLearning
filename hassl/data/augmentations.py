import torch
import numpy as np
from monai.transforms import (
    Compose,
    MapTransform,
    RandomizableTransform,
    RandFlipd,
    RandRotated,
    RandAffined,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
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


class SafeClampIntensityd(MapTransform):
    """Clamp image intensity to [minv, maxv] after augmentation to prevent out-of-range values.

    RandScaleIntensityd and RandAdjustContrastd can push pixel values above 1.0 even when the
    input was in [0,1]. Confirmed experimentally: max=1.0179 after strong augmentation.
    Out-of-range inputs shift the activation distribution and degrade convergence significantly.
    """

    def __init__(self, keys, minv: float = 0.0, maxv: float = 1.0, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.clamp(d[key].float(), self.minv, self.maxv)
        return d


def get_spatial_augmentation(keys=["image", "label"]):
    """Pure spatial/geometric transforms (flip, rotation, affine). Shared between teacher and student (V7-1 fix)."""
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(keys=keys, range_x=np.pi / 36, range_y=np.pi / 36, range_z=np.pi / 36, prob=0.5),
        RandAffined(
            keys=keys, prob=0.5,
            rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
            translate_range=(3, 3, 3), scale_range=(0.03, 0.03, 0.03),
        ),
    ])


def get_intensity_augmentation(keys=["image"]):
    """Pure photometric/intensity transforms (speckle, blur, contrast). Applied to student view only (V7-1 fix).

    All transforms are followed by SafeClampIntensityd to ensure image values stay in [0,1].
    Without this clamp, RandScaleIntensityd and RandAdjustContrastd push values above 1.0
    (measured max=1.0179), causing out-of-distribution inputs that degrade convergence.

    RandCoarseDropoutd/RandCoarseShuffled/RandGaussianSharpend/RandGibbsNoised/
    RandKSpaceSpikeNoised were ported in from the reference DeepEdit-style pipeline's
    `random_augment` block (transforms.py get_train_pre_transforms) at the user's request.
    Dropout/shuffle hole sizing (holes=2, spatial_size=4, max_holes=8, max_spatial_size=16)
    is reused as-is from that reference since hassl's spatial_size is also (128,128,128) by
    default, so the same hole-size-relative-to-volume-size assumption holds.
    """
    image_keys = [k for k in keys if k == "image"]
    return Compose([
        RandMultiplicativeSpeckleNoised(keys=image_keys, prob=0.5, std=0.08),
        RandGaussianSmoothd(keys=image_keys, prob=0.3, sigma_x=(0.5, 1.2), sigma_y=(0.5, 1.2), sigma_z=(0.5, 1.2)),
        RandGaussianNoised(keys=image_keys, prob=0.2, mean=0.0, std=0.02),  # Additive Gaussian noise for US background robustness
        RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
        RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.7, 1.5)),
        # --- Ported from reference pipeline's random_augment block ---
        RandCoarseDropoutd(
            keys=image_keys, holes=2, spatial_size=4, dropout_holes=True,
            max_holes=8, max_spatial_size=16, prob=0.2,
        ),
        RandCoarseShuffled(
            keys=image_keys, holes=2, spatial_size=4, max_holes=8, max_spatial_size=16, prob=0.2,
        ),
        RandGaussianSharpend(keys=image_keys, prob=0.1),
        RandGibbsNoised(keys=image_keys, prob=0.1, alpha=(0.0, 1.0)),
        RandKSpaceSpikeNoised(keys=image_keys, prob=0.1),
        # --- end ported block ---
        SafeClampIntensityd(keys=image_keys, minv=0.0, maxv=1.0),  # Clamp after aug: prevents >1.0 inputs to network
    ])


def get_weak_augmentation(keys=["image", "label"]):
    """In-plane ultrasound flip (lateral axis only) + small rotation (§7 fix)."""
    return get_spatial_augmentation(keys=keys)


def get_strong_augmentation(keys=["image", "label"]):
    """Strong ultrasound augmentations: Spatial + Photometric Speckle & Contrast (§7 fix)."""
    return Compose([
        get_spatial_augmentation(keys=keys),
        get_intensity_augmentation(keys=keys),
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
