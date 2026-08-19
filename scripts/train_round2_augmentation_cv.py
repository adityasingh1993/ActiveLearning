#!/usr/bin/env python3
"""Controlled Round-2 augmentation experiments on the frozen Final62 development design.

This runner intentionally changes ONLY the training augmentation profile relative to
experiments/round2_cv_62_translation12. It reuses the established Round-2 provenance-safe
runner, which holds out the exact same original 47 cases/folds and appends the 15 audited
Round-1/Round-2 HUMAN_GOLD cases to TRAIN ONLY.

Profiles
--------
appearance (A1)
    Existing paired translation +/-12 vox, p=0.8 plus conservative image-only appearance
    perturbations after deterministic 0..1 preprocessing:
      - gamma/contrast [0.8, 1.2], p=0.20
      - intensity scale +/-10%, p=0.15
      - intensity shift +/-0.05, p=0.15
      - Gaussian noise std<=0.02, p=0.10
      - Gaussian smoothing sigma 0.25..0.75 vox, p=0.10
    Appearance transforms are applied BEFORE translation so affine zero-padding remains zero.

mild_affine (A2)
    Existing translation +/-12 vox plus very mild rotation +/-3 degrees and scale +/-5%,
    all in one paired image/label RandAffined at p=0.8. No flips and no intensity changes.

Frozen for both profiles
------------------------
- 62 audited HUMAN_GOLD labels visible: original47 + Round1 8 + Round2 7
- exact original-47 held-out fold assignments
- resize 128^3 / spacing 0.1 mm from config
- DynUNet prototype student + EMA teacher
- DiceCE / AdamW 1e-4 / weight decay 1e-5
- dropout 0 / lambda_unsup 0 / random initialization
- 100 CV epochs / seed 42
- raw ensemble evaluation at threshold 0.50
- no SSL, no LCC, no external31 access

Examples
--------
python scripts/train_round2_augmentation_cv.py --profile appearance --config config_resize128.yaml --gpu 1
python scripts/train_round2_augmentation_cv.py --profile mild_affine --config config_resize128.yaml --gpu 0
python scripts/train_round2_augmentation_cv.py --profile appearance --config config_resize128.yaml --gpu 1 --fold 0
"""

import json
import math
import os
import sys
from pathlib import Path


def _consume_option(argv, name):
    """Consume one scalar CLI option before importing torch/MONAI-heavy modules."""
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


def _option_value(argv, name):
    for i, token in enumerate(argv[1:], start=1):
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _has_option(argv, name):
    return any(x == name or x.startswith(name + "=") for x in argv[1:])


GPU, ARGV_NO_GPU = _consume_option(sys.argv, "--gpu")
PROFILE, CLEAN_ARGV = _consume_option(ARGV_NO_GPU, "--profile")

if PROFILE not in {"appearance", "mild_affine"}:
    raise SystemExit("--profile must be one of: appearance, mild_affine")

if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

DEFAULT_AUDIT = "experiments/round2_supervised_62_translation12/round2_label_audit.json"
DEFAULT_SOURCE_CV = "experiments/cv5_supervised_47_translation12"
DEFAULT_ROUND1_CV = "experiments/round1_cv_55_translation12"
BASELINE_OUTPUT = "experiments/round2_cv_62_translation12"
OUTPUTS = {
    "appearance": "experiments/round2_cv_62_translation12_aug_appearance_v1",
    "mild_affine": "experiments/round2_cv_62_translation12_aug_mildaffine_v1",
}

# Hard-lock every comparison variable/provenance source that must stay identical to A0.
LOCKED = {
    "--epochs": "100",
    "--seed": "42",
    "--resize-size": "128",
    "--eval-source": "ensemble",
    "--eval-threshold": "0.50",
    "--audit-metadata": DEFAULT_AUDIT,
    "--source-cv-dir": DEFAULT_SOURCE_CV,
    "--round1-cv-dir": DEFAULT_ROUND1_CV,
}
for name, expected in LOCKED.items():
    explicit = _option_value(CLEAN_ARGV, name)
    if explicit is None:
        continue
    if name in {"--epochs", "--seed", "--resize-size"}:
        same = int(explicit) == int(expected)
    elif name == "--eval-threshold":
        same = abs(float(explicit) - float(expected)) <= 1e-8
    else:
        same = explicit == expected
    if not same:
        raise SystemExit(
            f"Controlled augmentation CV locks {name}={expected}; got {explicit!r}. "
            "Use a separate experiment for that change."
        )

for name, expected in LOCKED.items():
    if not _has_option(CLEAN_ARGV, name):
        CLEAN_ARGV.extend([name, expected])

if not _has_option(CLEAN_ARGV, "--output-dir"):
    CLEAN_ARGV.extend(["--output-dir", OUTPUTS[PROFILE]])

resolved_output = Path(_option_value(CLEAN_ARGV, "--output-dir"))
if resolved_output.resolve() == Path(BASELINE_OUTPUT).resolve():
    raise SystemExit(
        "Refusing to write an augmentation experiment into the frozen A0 baseline directory: "
        f"{BASELINE_OUTPUT}"
    )

sys.argv = CLEAN_ARGV

# Heavy imports only after physical GPU visibility is fixed.
from monai.transforms import (  # noqa: E402
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_active_learning_round2_cv_from_audit as round2  # noqa: E402


APPEARANCE_PROFILE = {
    "name": "A1_appearance_v1",
    "translation_voxels": 12.0,
    "translation_probability": 0.8,
    "gamma_range": [0.8, 1.2],
    "gamma_probability": 0.20,
    "intensity_scale_factor": 0.10,
    "intensity_scale_probability": 0.15,
    "intensity_shift_offset": 0.05,
    "intensity_shift_probability": 0.15,
    "gaussian_noise_mean": 0.0,
    "gaussian_noise_std_max": 0.02,
    "gaussian_noise_probability": 0.10,
    "gaussian_smooth_sigma": [0.25, 0.75],
    "gaussian_smooth_probability": 0.10,
    "rotation_degrees": 0.0,
    "geometric_scale_fraction": 0.0,
    "flips": False,
}

MILD_AFFINE_PROFILE = {
    "name": "A2_mild_affine_v1",
    "translation_voxels": 12.0,
    "translation_probability": 0.8,
    "rotation_degrees_each_axis": 3.0,
    "geometric_scale_fraction": 0.05,
    "flips": False,
    "appearance_changes": False,
}


def appearance_transform(
    base_transform,
    translate_voxels=12.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """A1: mild image-only appearance variation + the frozen paired translation."""
    del rotate_degrees, scale_fraction
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    t = float(translate_voxels)
    return Compose(
        base_steps
        + [
            RandAdjustContrastd(keys=["image"], prob=0.20, gamma=(0.8, 1.2)),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.15),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.15),
            RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.02),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.10,
                sigma_x=(0.25, 0.75),
                sigma_y=(0.25, 0.75),
                sigma_z=(0.25, 0.75),
            ),
            # Translation last: preserve clean zero padding outside the shifted FOV.
            RandAffined(
                keys=["image", "label"],
                prob=0.8,
                rotate_range=(0.0, 0.0, 0.0),
                translate_range=(t, t, t),
                scale_range=(0.0, 0.0, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]
    )


def mild_affine_transform(
    base_transform,
    translate_voxels=12.0,
    rotate_degrees=3.0,
    scale_fraction=0.05,
):
    """A2: frozen translation plus conservative paired 3-D rotation and scale."""
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    t = float(translate_voxels)
    r = math.radians(float(rotate_degrees))
    s = float(scale_fraction)
    return Compose(
        base_steps
        + [
            RandAffined(
                keys=["image", "label"],
                prob=0.8,
                rotate_range=(r, r, r),
                translate_range=(t, t, t),
                scale_range=(s, s, s),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]
    )


def _selected_output_dir(argv):
    value = _option_value(argv, "--output-dir")
    if value is None:
        raise RuntimeError("Internal error: --output-dir was not resolved")
    return Path(value)


def _write_profile_metadata(output_dir: Path, profile: str):
    payload = {
        "version": "final62_round2_augmentation_ablation_v1",
        "profile": profile,
        "profile_parameters": APPEARANCE_PROFILE if profile == "appearance" else MILD_AFFINE_PROFILE,
        "controlled_reference": BASELINE_OUTPUT,
        "source_cv": DEFAULT_SOURCE_CV,
        "round1_cv": DEFAULT_ROUND1_CV,
        "round2_audit": DEFAULT_AUDIT,
        "held_out_design": (
            "Exact original-47 fold assignments are held out; all 15 audited post-source "
            "HUMAN_GOLD labels are train-only in every fold."
        ),
        "frozen": {
            "epochs": 100,
            "seed": 42,
            "resize_size": [128, 128, 128],
            "loss": "dice_ce",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
            "ssl": False,
            "external31_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "augmentation_profile.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main():
    chosen = appearance_transform if PROFILE == "appearance" else mild_affine_transform

    # The established Round-2 runner assigns cv.spatial_aug_transform from this module-global
    # symbol inside its main(). Replacing the symbol preserves every other provenance/evaluation
    # guard while changing only the transform factory.
    round2.translation_only_transform = chosen

    # The base Round-2 runtime stores rotate/scale as 0 because A0 is translation-only. For A2,
    # set the same runtime object's fields immediately before each fold. install_cv_loader_hook
    # captures that object by reference, so both the actual transform parameters and fold log are
    # now truthful (+/-3 degrees, +/-5%).
    original_run_fold = round2.cv.run_fold

    def run_fold_with_profile_params(args, fold_spec, output_dir):
        if PROFILE == "mild_affine":
            args.rotate_degrees = 3.0
            args.scale_fraction = 0.05
        else:
            args.rotate_degrees = 0.0
            args.scale_fraction = 0.0
        return original_run_fold(args, fold_spec, output_dir)

    round2.cv.run_fold = run_fold_with_profile_params

    output_dir = _selected_output_dir(sys.argv)
    _write_profile_metadata(output_dir, PROFILE)

    print("=" * 118)
    print("FINAL62 CONTROLLED AUGMENTATION CV")
    print(f"Profile:             {PROFILE}")
    print(f"Output:              {output_dir}")
    print("Held-out cohort:     exact frozen original47 folds")
    print("Train-only extras:   Round1 8 + Round2 7 HUMAN_GOLD")
    print("Frozen:              128^3 | 100 epochs | seed42 | DiceCE | AdamW 1e-4 | EMA | ensemble@.50")
    print("SSL/LCC/external31:  OFF / OFF / NOT USED")
    print("Physical GPU:        " + (GPU if GPU is not None else "<environment/config>"))
    if PROFILE == "appearance":
        print("A1 delta:            gamma + scale/shift intensity + mild noise/smoothing; translation unchanged")
    else:
        print("A2 delta:            rotation +/-3deg + scale +/-5%; translation unchanged; no flips")
    print("=" * 118)

    round2.main()


if __name__ == "__main__":
    main()
