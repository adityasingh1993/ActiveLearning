"""Reusable training controls for the CenterNet -> DynUNet Stage-2 model.

The historical Stage-2 recipe used one uniformly sampled GT-box crop with a 40%-60%
margin.  This module keeps that behaviour as the default and provides opt-in controls for
two failure-oriented interventions:

* a mixture of the historical crop and a wider, noisier GT-safe crop; and
* fold-local small-bladder oversampling plus mild ultrasound appearance augmentation.

All target-derived information in this module is training-only.  Validation and inference
continue to consume frozen CenterNet crops and never use GT size or bounds.
"""

from dataclasses import dataclass
from functools import lru_cache
import copy
import math

import numpy as np
import nrrd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandomizableTransform,
)
from torch.utils.data import WeightedRandomSampler

from .augmentations import RandMultiplicativeSpeckleNoised, SafeClampIntensityd


MILD_APPEARANCE_RECIPE = {
    "multiplicative_speckle_probability": 0.30,
    "multiplicative_speckle_std": 0.05,
    "intensity_scale_probability": 0.30,
    "intensity_scale_factor": 0.08,
    "contrast_probability": 0.30,
    "contrast_gamma_range": [0.85, 1.15],
    "final_intensity_range": [0.0, 1.0],
}


@dataclass(frozen=True)
class Stage2TrainingOptions:
    """Complete training-only Stage-2 data recipe.

    ``wide_crop_probability=0``, ``small_bladder_weight=1``, and
    ``mild_appearance_augmentation=False`` reproduce the historical data path.
    """

    output_size: int = 128
    margin_min: float = 0.40
    margin_max: float = 0.60
    center_jitter: float = 0.10
    wide_crop_probability: float = 0.0
    wide_margin_min: float = 0.60
    wide_margin_max: float = 1.00
    wide_center_jitter: float = 0.20
    small_bladder_quantile: float = 1.0 / 3.0
    small_bladder_weight: float = 1.0
    mild_appearance_augmentation: bool = False

    def validate(self):
        if self.output_size < 32 or self.output_size % 16:
            raise ValueError("Stage-2 output_size must be >=32 and divisible by 16")
        if not 0.0 <= self.margin_min <= self.margin_max <= 1.0:
            raise ValueError("Stage-2 base margins must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.center_jitter <= 0.5:
            raise ValueError("Stage-2 base center jitter must be in [0, 0.5]")
        if not 0.0 <= self.wide_crop_probability <= 1.0:
            raise ValueError("wide_crop_probability must be in [0, 1]")
        if not 0.0 <= self.wide_margin_min <= self.wide_margin_max <= 2.0:
            raise ValueError("wide margins must satisfy 0 <= min <= max <= 2")
        if not 0.0 <= self.wide_center_jitter <= 0.5:
            raise ValueError("wide_center_jitter must be in [0, 0.5]")
        if not 0.0 < self.small_bladder_quantile < 1.0:
            raise ValueError("small_bladder_quantile must be in (0, 1)")
        if self.small_bladder_weight < 1.0:
            raise ValueError("small_bladder_weight must be >=1")
        return self

    @property
    def robust_enabled(self):
        return bool(
            self.wide_crop_probability > 0.0
            or self.small_bladder_weight > 1.0
            or self.mild_appearance_augmentation
        )


def add_training_recipe_extensions(recipe, options):
    """Record only enabled interventions, preserving exact legacy recipe equality."""
    options.validate()
    result = copy.deepcopy(recipe)
    if options.wide_crop_probability > 0.0:
        result["training_crop"]["wide_crop_mixture"] = {
            "probability": float(options.wide_crop_probability),
            "margin_per_side_range": [
                float(options.wide_margin_min), float(options.wide_margin_max)
            ],
            "center_jitter_fraction_of_gt_size": float(options.wide_center_jitter),
            "guarantee": "crop unioned with GT bounds so foreground is never cut",
        }
    if options.small_bladder_weight > 1.0:
        result["training_sampler"] = {
            "type": "WeightedRandomSampler",
            "replacement": True,
            "samples_per_epoch": "len(training_ids)",
            "small_definition": "bottom training-only native physical-volume quantile",
            "small_quantile": float(options.small_bladder_quantile),
            "small_case_weight": float(options.small_bladder_weight),
            "validation_labels_used": False,
        }
    if options.mild_appearance_augmentation:
        result["augmentation_after_crop"]["mild_ultrasound_appearance"] = copy.deepcopy(
            MILD_APPEARANCE_RECIPE
        )
    return result


def options_from_recipe(recipe):
    """Load robust options from either a historical or extended CV recipe."""
    crop = recipe.get("training_crop", {})
    margin_range = crop.get("margin_per_side_range", [0.40, 0.60])
    wide = crop.get("wide_crop_mixture", {})
    wide_range = wide.get("margin_per_side_range", [0.60, 1.00])
    sampler = recipe.get("training_sampler", {})
    appearance = recipe.get("augmentation_after_crop", {}).get(
        "mild_ultrasound_appearance"
    )
    crop_size = recipe.get("crop_size", [128, 128, 128])
    if not crop_size or len(set(int(x) for x in crop_size)) != 1:
        raise ValueError(f"Stage-2 recipe crop_size must be cubic, got {crop_size}")
    return Stage2TrainingOptions(
        output_size=int(crop_size[0]),
        margin_min=float(margin_range[0]),
        margin_max=float(margin_range[1]),
        center_jitter=float(crop.get("center_jitter_fraction_of_gt_size", 0.10)),
        wide_crop_probability=float(wide.get("probability", 0.0)),
        wide_margin_min=float(wide_range[0]),
        wide_margin_max=float(wide_range[1]),
        wide_center_jitter=float(
            wide.get("center_jitter_fraction_of_gt_size", 0.20)
        ),
        small_bladder_quantile=float(sampler.get("small_quantile", 1.0 / 3.0)),
        small_bladder_weight=float(sampler.get("small_case_weight", 1.0)),
        mild_appearance_augmentation=bool(appearance),
    ).validate()


def plain_tensor(value):
    return value.as_tensor() if hasattr(value, "as_tensor") else torch.as_tensor(value)


def crop_and_resize_pair(image, label, bounds, output_size):
    z0, z1, y0, y1, x0, x1 = [int(x) for x in bounds]
    image_tensor = plain_tensor(image).float()
    label_tensor = plain_tensor(label).float()
    image_crop = image_tensor[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    label_crop = label_tensor[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    if image_crop.numel() == 0 or label_crop.numel() == 0:
        raise RuntimeError(f"Empty Stage-2 crop from bounds {bounds}")
    target_size = (int(output_size),) * 3
    image_output = F.interpolate(
        image_crop.unsqueeze(0), size=target_size, mode="trilinear", align_corners=False
    )[0]
    label_output = F.interpolate(
        label_crop.unsqueeze(0), size=target_size, mode="nearest"
    )[0]
    return image_output, (label_output > 0.5).float()


class SafeMixedGTBoxCropd(RandomizableTransform):
    """Sample a base or wide GT-derived crop while guaranteeing full GT inclusion."""

    def __init__(self, options):
        super().__init__(prob=1.0)
        self.options = options.validate()

    def __call__(self, data):
        result = dict(data)
        self.randomize(None)
        label = plain_tensor(result["label"])
        spatial = label[0] if label.ndim == 4 else label
        coordinates = torch.nonzero(spatial > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            raise RuntimeError(f"Empty Stage-2 training GT: {result.get('id', '?')}")

        gt_lo = coordinates.min(dim=0).values.cpu().numpy().astype(float)
        gt_hi = coordinates.max(dim=0).values.cpu().numpy().astype(float)
        gt_size = gt_hi - gt_lo + 1.0
        gt_center = 0.5 * (gt_lo + gt_hi)

        use_wide = False
        if self.options.wide_crop_probability > 0.0:
            use_wide = bool(self.R.random() < self.options.wide_crop_probability)
        if use_wide:
            margin_min = self.options.wide_margin_min
            margin_max = self.options.wide_margin_max
            center_jitter = self.options.wide_center_jitter
        else:
            margin_min = self.options.margin_min
            margin_max = self.options.margin_max
            center_jitter = self.options.center_jitter

        margin = float(self.R.uniform(margin_min, margin_max))
        shift = self.R.uniform(low=-center_jitter, high=center_jitter, size=3) * gt_size
        proposed_center = gt_center + shift
        proposed_size = gt_size * (1.0 + 2.0 * margin)
        lo = np.floor(proposed_center - 0.5 * proposed_size).astype(int)
        hi = np.ceil(proposed_center + 0.5 * proposed_size).astype(int)

        # Noise may move the crop, but training GT is never truncated.
        lo = np.minimum(lo, gt_lo.astype(int))
        hi = np.maximum(hi, gt_hi.astype(int))
        shape = np.asarray(spatial.shape, dtype=int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shape - 1)
        bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        result["image"], result["label"] = crop_and_resize_pair(
            result["image"], result["label"], bounds, self.options.output_size
        )
        result["training_crop_bounds"] = torch.tensor(bounds, dtype=torch.int64)
        result["training_crop_margin"] = torch.tensor(margin, dtype=torch.float32)
        result["training_crop_is_wide"] = torch.tensor(int(use_wide), dtype=torch.uint8)
        result["training_crop_gt_fraction"] = result["label"].float().mean()
        return result


def build_stage2_training_transform(base_transform, options):
    """Append the shared crop, spatial, and optional mild appearance transforms."""
    options.validate()
    steps = list(getattr(base_transform, "transforms", [base_transform]))
    steps.extend([
        SafeMixedGTBoxCropd(options),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])
    if options.mild_appearance_augmentation:
        steps.extend([
            RandMultiplicativeSpeckleNoised(
                keys=["image"],
                prob=MILD_APPEARANCE_RECIPE["multiplicative_speckle_probability"],
                std=MILD_APPEARANCE_RECIPE["multiplicative_speckle_std"],
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=MILD_APPEARANCE_RECIPE["intensity_scale_factor"],
                prob=MILD_APPEARANCE_RECIPE["intensity_scale_probability"],
            ),
            RandAdjustContrastd(
                keys=["image"],
                gamma=tuple(MILD_APPEARANCE_RECIPE["contrast_gamma_range"]),
                prob=MILD_APPEARANCE_RECIPE["contrast_probability"],
            ),
            SafeClampIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ])
    return Compose(steps)


def segmentation_voxel_volume_mm3(header, fallback_spacing):
    directions = header.get("space directions")
    vectors = []
    if directions is not None:
        for item in directions:
            try:
                vector = np.asarray(item, dtype=float).reshape(-1)
            except Exception:
                continue
            if vector.size == 3 and np.all(np.isfinite(vector)):
                vectors.append(vector)
    if len(vectors) >= 3:
        value = abs(float(np.linalg.det(np.stack(vectors[:3], axis=0))))
        if math.isfinite(value) and value > 0.0:
            return value
    return float(np.prod(np.asarray(fallback_spacing, dtype=float)))


@lru_cache(maxsize=None)
def _native_bladder_ml(label_path, image_path):
    image = sitk.ReadImage(str(image_path))
    fallback_spacing = tuple(float(x) for x in image.GetSpacing())
    segmentation, header = nrrd.read(str(label_path))
    segmentation = np.squeeze(segmentation)
    if segmentation.ndim != 3:
        raise RuntimeError(
            f"Stage-2 label is not 3D after squeeze: {label_path} shape={segmentation.shape}"
        )
    foreground = int(np.count_nonzero(segmentation > 0))
    if foreground <= 0:
        raise RuntimeError(f"Empty HUMAN_GOLD label: {label_path}")
    voxel_mm3 = segmentation_voxel_volume_mm3(header, fallback_spacing)
    volume_ml = float(foreground * voxel_mm3 / 1000.0)
    if not math.isfinite(volume_ml) or volume_ml <= 0.0:
        raise RuntimeError(f"Invalid physical bladder volume for {label_path}: {volume_ml}")
    return volume_ml


def small_bladder_sampling_plan(case_ids, volumes_ml, quantile, small_weight):
    """Return ordered sample weights and an auditable training-only size split."""
    case_ids = [str(x) for x in case_ids]
    volumes_ml = {str(case_id): float(value) for case_id, value in volumes_ml.items()}
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Small-bladder sampling requires unique non-empty case IDs")
    if set(case_ids) != set(volumes_ml):
        raise ValueError("Physical-volume keys must exactly match training case IDs")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("Small-bladder quantile must be in (0, 1)")
    if float(small_weight) <= 1.0:
        raise ValueError("Small-bladder weighted sampling requires weight >1")

    values = np.asarray([float(volumes_ml[x]) for x in case_ids], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("All physical bladder volumes must be finite and positive")
    threshold = float(np.quantile(values, float(quantile)))
    small_ids = sorted(x for x in case_ids if float(volumes_ml[x]) <= threshold)
    small_set = set(small_ids)
    weights = [float(small_weight if x in small_set else 1.0) for x in case_ids]
    expected_small_fraction = float(
        sum(weights[index] for index, case_id in enumerate(case_ids) if case_id in small_set)
        / sum(weights)
    )
    audit = {
        "enabled": True,
        "definition": "bottom training-only native physical-volume quantile",
        "small_quantile": float(quantile),
        "small_threshold_ml_training_only": threshold,
        "small_case_weight": float(small_weight),
        "n_training": len(case_ids),
        "n_small_training": len(small_ids),
        "small_training_ids": small_ids,
        "training_case_volumes_ml": {
            case_id: float(volumes_ml[case_id]) for case_id in case_ids
        },
        "expected_small_draw_fraction": expected_small_fraction,
        "validation_labels_used": False,
    }
    return weights, audit


def build_small_bladder_sampler(items, options, seed):
    """Build a fixed-length replacement sampler from training labels only."""
    options.validate()
    if options.small_bladder_weight <= 1.0:
        return None, {
            "enabled": False,
            "sampler": "shuffle_without_replacement",
            "samples_per_epoch": len(items),
            "validation_labels_used": False,
        }
    case_ids = [str(item["id"]) for item in items]
    volumes = {
        str(item["id"]): _native_bladder_ml(str(item["label"]), str(item["image"]))
        for item in items
    }
    weights, audit = small_bladder_sampling_plan(
        case_ids,
        volumes,
        options.small_bladder_quantile,
        options.small_bladder_weight,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(items),
        replacement=True,
        generator=generator,
    )
    audit.update({
        "sampler": "WeightedRandomSampler(replacement=True)",
        "samples_per_epoch": len(items),
        "steps_per_epoch_preserved": True,
        "sampler_seed": int(seed),
    })
    return sampler, audit
