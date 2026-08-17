#!/usr/bin/env python3
"""Run the translation-only spatial-augmentation CV ablation.

This is a thin wrapper around scripts/train_supervised_cv.py so the baseline,
moderate spatial augmentation, and mild spatial augmentation experiments remain
exactly reproducible.

Translation-only profile (training only):
- NO flips
- translation: +/-8 voxels on each spatial axis
- NO rotation
- NO scale augmentation
- affine probability: 0.8
- image interpolation: bilinear
- label interpolation: nearest
- padding mode: zeros

The wrapper automatically enables --spatial-aug, reuses the baseline frozen
cv_splits.json through the underlying runner, and writes to a separate output
namespace. Model, loss, fold seeds, epoch budget, validation preprocessing, and
held-out ensemble@0.50 evaluation remain unchanged.
"""

import sys
from pathlib import Path

from monai.transforms import Compose, RandAffined

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv


TRANSLATION_OUTPUT_DIR = "experiments/cv5_supervised_resize128_translationonly"


def translation_only_transform(
    base_transform,
    translate_voxels=8.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """Append paired translation only after deterministic 128^3 preprocessing."""
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
    ])


def main():
    # Patch only the augmentation factory. The shared runner still owns the frozen
    # folds, per-fold seeds, checkpoint isolation, training, deterministic held-out
    # evaluation, result persistence, and baseline comparison.
    cv.spatial_aug_transform = translation_only_transform

    original_run_fold = cv.run_fold

    def run_fold_with_translation_label(args, fold_spec, output_dir):
        rows = original_run_fold(args, fold_spec, output_dir)
        for row in rows:
            row["experiment"] = "translation_only"
        return rows

    cv.run_fold = run_fold_with_translation_label

    user_args = sys.argv[1:]
    injected = [
        "--spatial-aug",
        "--translate-voxels", "8",
        "--rotate-degrees", "0",
        "--scale-fraction", "0",
        "--output-dir", TRANSLATION_OUTPUT_DIR,
    ]

    # Defaults are inserted first so an explicitly supplied scalar CLI option can
    # override them if a later diagnostic intentionally needs a different value.
    sys.argv = [sys.argv[0]] + injected + user_args
    cv.main()


if __name__ == "__main__":
    main()
