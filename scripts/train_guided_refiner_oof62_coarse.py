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

GPU selection
-------------
Use ``--gpu 0`` or ``--gpu 1``. CUDA_VISIBLE_DEVICES is configured before PyTorch/MONAI
are imported, so the selected physical GPU appears to the child training code as cuda:0.
This avoids changing the frozen HASSL training recipe while allowing folds to run on either GPU.
"""

import os
import sys
from pathlib import Path


def _consume_gpu_argument(argv):
    """Set CUDA_VISIBLE_DEVICES before importing any torch/MONAI code."""
    args = list(argv)
    gpu = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == "--gpu":
            if i + 1 >= len(args):
                raise SystemExit("--gpu requires 0 or 1")
            gpu = args[i + 1]
            i += 2
            continue
        if token.startswith("--gpu="):
            gpu = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1

    if gpu is not None:
        if gpu not in {"0", "1"}:
            raise SystemExit(f"--gpu must be 0 or 1, got {gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu, cleaned


SELECTED_GPU, CLEAN_ARGV = _consume_gpu_argument(sys.argv)
sys.argv = CLEAN_ARGV

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import only after CUDA_VISIBLE_DEVICES has been configured.
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

    gpu_text = f"physical GPU {SELECTED_GPU}" if SELECTED_GPU is not None else "GPU from environment/config"
    print(
        "Guided-refiner OOF62 coarse profile: all 62 held out once | EMA evaluation | "
        "translation +/-12 vox p=0.8 | all other augmentation OFF"
    )
    print(f"CUDA selection: {gpu_text} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    cv.main()


if __name__ == "__main__":
    main()
