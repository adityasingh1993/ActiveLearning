#!/usr/bin/env python3
"""Run translation-only +/-12 voxels plus intensity-scale CV ablation.

This is a thin wrapper around scripts/train_supervised_cv.py so all existing
baseline/augmentation experiments remain reproducible.

Training-only profile:
- paired image/label translation: +/-12 voxels on each spatial axis, p=0.8
- image-only intensity scaling: +/-10%, p=0.3
- NO gamma/contrast
- NO flips
- NO rotation
- NO geometric scaling
- NO noise
- image interpolation: bilinear
- label interpolation: nearest
- affine padding mode: zeros

The wrapper reuses the baseline frozen cv_splits.json through the shared runner,
keeps the same per-fold seeds/model/loss/evaluation, and writes to a separate
output namespace. The reported held-out source/threshold remain ensemble@0.50
for direct comparison with the frozen translation-only +/-12 reference.
"""

import sys
from pathlib import Path

from monai.transforms import Compose, RandAffined, RandScaleIntensityd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv


OUTPUT_DIR = "experiments/cv5_supervised_resize128_translation12_intensityscale"
INTENSITY_FACTOR = 0.10
INTENSITY_PROB = 0.3


def translation_intensity_transform(
    base_transform,
    translate_voxels=12.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """Append paired translation followed by image-only intensity scaling."""
    translate = float(translate_voxels)
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))

    return Compose(base_steps + [
        RandAffined(
            keys=["image", "label"],
            prob=0.8,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(translate, translate, translate),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=INTENSITY_FACTOR,
            prob=INTENSITY_PROB,
        ),
    ])


def main():
    # Patch only the training augmentation factory; the shared CV runner still owns
    # split provenance, deterministic validation, checkpoint isolation, fold seeds,
    # training, result persistence, and baseline comparison.
    cv.spatial_aug_transform = translation_intensity_transform

    original_run_fold = cv.run_fold

    def run_fold_with_experiment_label(args, fold_spec, output_dir):
        rows = original_run_fold(args, fold_spec, output_dir)
        for row in rows:
            row["experiment"] = "translation12_intensityscale"
        return rows

    cv.run_fold = run_fold_with_experiment_label

    user_args = sys.argv[1:]
    injected = [
        "--spatial-aug",
        "--translate-voxels", "12",
        "--rotate-degrees", "0",
        "--scale-fraction", "0",
        "--output-dir", OUTPUT_DIR,
    ]

    # Defaults first so explicit shared-runner scalar options supplied later can still
    # override them for a deliberate diagnostic. Intensity settings stay fixed so the
    # experiment has one unambiguous provenance.
    sys.argv = [sys.argv[0]] + injected + user_args

    print(
        "Appearance profile: translation +/-12 vox (p=0.8) + "
        f"intensity scale +/-{INTENSITY_FACTOR:.0%} (p={INTENSITY_PROB:.1f}); "
        "gamma/flip/rotation/geometric-scale/noise OFF"
    )
    cv.main()


if __name__ == "__main__":
    main()
