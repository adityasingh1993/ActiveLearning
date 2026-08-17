#!/usr/bin/env python3
"""Run the mild, anatomically constrained spatial-augmentation CV experiment.

This is a thin wrapper around scripts/train_supervised_cv.py so the original baseline
and moderate spatial-augmentation experiment remain exactly reproducible.

Mild profile (training only):
- paired image/label flips on spatial axes 1 and 2
- translation: +/-6 voxels on each spatial axis
- rotation: +/-5 degrees around the axis normal to the axes-(1,2) plane only
- scale: +/-5% on each spatial axis
- affine probability: 0.8
- image interpolation: bilinear
- label interpolation: nearest

The wrapper automatically enables --spatial-aug and writes to a separate experiment
namespace while reusing the baseline cv_splits.json through the underlying runner.
Additional CLI arguments accepted by train_supervised_cv.py can still be supplied and
will override the injected defaults when appropriate.
"""

import math
import sys
from pathlib import Path

from monai.transforms import Compose, RandAffined, RandFlipd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv


MILD_OUTPUT_DIR = "experiments/cv5_supervised_resize128_spatialaug_mild"


def mild_spatial_aug_transform(
    base_transform,
    translate_voxels=6.0,
    rotate_degrees=5.0,
    scale_fraction=0.05,
):
    """Append the mild paired spatial profile after deterministic preprocessing.

    The existing CV experiment flips spatial axes 1 and 2, so those axes define the
    intended in-plane plane for this controlled experiment. Rotation is therefore
    restricted to the single component normal to that plane rather than rotating
    independently around all three axes.
    """
    rotate_radians = math.radians(float(rotate_degrees))
    translate = float(translate_voxels)
    scale = float(scale_fraction)

    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.8,
            rotate_range=(rotate_radians, 0.0, 0.0),
            translate_range=(translate, translate, translate),
            scale_range=(scale, scale, scale),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


def main():
    # The underlying runner owns split validation, checkpoint isolation, held-out
    # evaluation, CSV writing, and baseline comparison. Patch only its augmentation
    # factory so every other experimental detail stays identical.
    cv.spatial_aug_transform = mild_spatial_aug_transform

    user_args = sys.argv[1:]
    injected = [
        "--spatial-aug",
        "--translate-voxels", "6",
        "--rotate-degrees", "5",
        "--scale-fraction", "0.05",
        "--output-dir", MILD_OUTPUT_DIR,
    ]

    # Put defaults first: argparse uses the later value for repeated scalar options,
    # allowing an explicit user-provided --output-dir or range to override if needed.
    sys.argv = [sys.argv[0]] + injected + user_args
    cv.main()


if __name__ == "__main__":
    main()
