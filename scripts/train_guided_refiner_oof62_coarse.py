#!/usr/bin/env python3
"""Train the fresh all-62 OOF coarse models used to create guided-refiner inputs.

The recipe intentionally matches the frozen Final62 supervised profile:
- DynUNet / prototype student + EMA teacher
- resize128
- DiceCE
- AdamW 1e-4
- dropout 0
- lambda_unsup 0
- paired translation +/-12 voxels, p=0.8
- no flip/rotation/scale/intensity/noise augmentation

Unlike the controlled Round-2 CV, every one of the 62 HUMAN_GOLD cases is held out exactly
once. These models are for OOF guidance generation, not for changing the production model.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv
from scripts.train_supervised_cv_47_translation import translation_only_transform

OUTPUT_DIR = Path("experiments/guided_refiner_oof62_coarse_v1")
SPLIT_MANIFEST = OUTPUT_DIR / "cv_splits.json"


def main():
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing all-62 OOF manifest: {SPLIT_MANIFEST}\n"
            "Run first: python scripts/prepare_guided_refiner_oof62.py --config <config>"
        )

    cv.spatial_aug_transform = translation_only_transform

    original_run_fold = cv.run_fold

    def run_fold_with_label(args, fold_spec, output_dir):
        rows = original_run_fold(args, fold_spec, output_dir)
        for row in rows:
            row["experiment"] = "guided_refiner_oof62_coarse_translation12"
        return rows

    cv.run_fold = run_fold_with_label

    user_args = sys.argv[1:]
    injected = [
        "--spatial-aug",
        "--folds", "5",
        "--seed", "42",
        "--translate-voxels", "12",
        "--rotate-degrees", "0",
        "--scale-fraction", "0",
        "--epochs", "100",
        "--eval-source", "teacher",
        "--eval-threshold", "0.50",
        "--output-dir", str(OUTPUT_DIR),
        "--split-manifest", str(SPLIT_MANIFEST),
    ]
    sys.argv = [sys.argv[0]] + injected + user_args

    print(
        "Guided-refiner OOF62 coarse profile: all 62 held out once | EMA evaluation | "
        "translation +/-12 vox p=0.8 | all other augmentation OFF"
    )
    cv.main()


if __name__ == "__main__":
    main()
