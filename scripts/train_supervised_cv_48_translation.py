#!/usr/bin/env python3
"""Run the frozen 48-label supervised CV experiment with translation-only augmentation.

Preparation prerequisite:
  python scripts/prepare_48_labeled_cv.py --config <config>

That command must create and freeze:
  experiments/cv5_supervised_48_translation12/cv_splits.json

Training profile:
- all labeled cases visible to the config (manifest currently expected to contain 48)
- 5-fold patient-grouped CV, seed 42 by default
- resize 128^3 / spacing from config
- DiceCE, random init, dropout=0, LR=1e-4, lambda_unsup=0
- paired image/label translation +/-12 voxels, p=0.8
- no flips, no rotation, no geometric scale, no gamma, no intensity scaling, no noise
- held-out reporting defaults to ensemble @ 0.50 for continuity with the frozen 13-case result

Use --fold 0, --fold 1, etc. Completed folds are preserved by the shared CV runner,
so it is safe to run only folds 0/1/2 now and complete folds 3/4 later without changing
split provenance.
"""

import sys
from pathlib import Path

from monai.transforms import Compose, RandAffined

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv


OUTPUT_DIR = Path("experiments/cv5_supervised_48_translation12")
SPLIT_MANIFEST = OUTPUT_DIR / "cv_splits.json"


def translation_only_transform(
    base_transform,
    translate_voxels=12.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """Append only paired translation to deterministic preprocessing."""
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
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing frozen 48-label split manifest: {SPLIT_MANIFEST}\n"
            "Run first: python scripts/prepare_48_labeled_cv.py --config <config>"
        )

    cv.spatial_aug_transform = translation_only_transform

    original_run_fold = cv.run_fold

    def run_fold_with_experiment_label(args, fold_spec, output_dir):
        rows = original_run_fold(args, fold_spec, output_dir)
        for row in rows:
            row["experiment"] = "48labels_translation12"
        return rows

    cv.run_fold = run_fold_with_experiment_label

    user_args = sys.argv[1:]
    injected = [
        "--spatial-aug",
        "--folds", "5",
        "--seed", "42",
        "--translate-voxels", "12",
        "--rotate-degrees", "0",
        "--scale-fraction", "0",
        "--output-dir", str(OUTPUT_DIR),
        "--split-manifest", str(SPLIT_MANIFEST),
    ]

    # Defaults are injected first so deliberate CLI choices such as --eval-source student
    # or --epochs can be supplied later by the user without editing this file.
    sys.argv = [sys.argv[0]] + injected + user_args

    print(
        "48-label profile: frozen 5-fold manifest | translation +/-12 vox (p=0.8) | "
        "flip/rotation/scale/gamma/intensity/noise OFF"
    )
    cv.main()


if __name__ == "__main__":
    main()
