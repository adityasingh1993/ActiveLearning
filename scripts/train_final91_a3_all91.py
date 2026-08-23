#!/usr/bin/env python3
"""Train one deployment Final91 A3 model on all 91 audited HUMAN_GOLD cases.

Epoch count defaults to the median best epoch across the completed Final91 five-fold CV.
The deployment checkpoint is the fixed-epoch latest checkpoint; same-data validation is diagnostic
only and does not select weights. External31 is never accessed here.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    MapTransform,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
import scripts.train_supervised_cv as cv
import scripts.train_final72_screen_spatial_folds12 as spatial
from scripts.audit_round1_labels import discover_round1_cases

CV_DIR = Path("experiments/round5_cv_91_a3")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
OUTPUT = Path("experiments/final91_a3_all91")
EXPECTED_TOTAL = 91
EXPECTED_SOURCE = 47

A3_VARIANT = {
    "name": "translation4_p05_lrflip_p05",
    "spatial_aug": True,
    "translate_voxels": 4.0,
    "translate_prob": 0.5,
    "lr_flip": True,
    "lr_flip_prob": 0.5,
}

APPEARANCE_AUGMENTATION = {
    "gamma_contrast": {"probability": 0.30, "gamma": [0.7, 1.5]},
    "intensity_scale": {"probability": 0.20, "factor": 0.10},
    "intensity_shift": {"probability": 0.20, "offset": 0.05},
    "gaussian_noise": {"probability": 0.15, "mean": 0.0, "std": 0.02},
    "gaussian_smoothing": {"probability": 0.15, "sigma": [0.5, 1.0]},
    "clip_after_augmentation": [0.0, 1.0],
}


class ClipIntensityd(MapTransform):
    """Clip normalized image intensities after random appearance perturbations."""

    def __init__(self, keys, low=0.0, high=1.0):
        super().__init__(keys)
        self.low = float(low)
        self.high = float(high)

    def __call__(self, data):
        result = dict(data)
        for key in self.keys:
            result[key] = torch.clamp(result[key], self.low, self.high)
        return result


def add_mild_ultrasound_appearance_augmentation(transform):
    """Append the predeclared mild image-only ultrasound appearance augmentation bundle."""
    steps = list(getattr(transform, "transforms", [transform]))
    steps.extend([
        RandAdjustContrastd(keys=["image"], prob=0.30, gamma=(0.7, 1.5)),
        RandScaleIntensityd(keys=["image"], prob=0.20, factors=0.10),
        RandShiftIntensityd(keys=["image"], prob=0.20, offsets=0.05),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.15,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        ClipIntensityd(keys=["image"], low=0.0, high=1.0),
    ])
    return Compose(steps)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_cv_best_epochs(cv_dir: Path):
    rows = []
    for fold in range(5):
        checkpoint = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing Final91 CV checkpoint: {checkpoint}. Complete all five folds first.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(state.get("epoch", 0))
        if epoch <= 0:
            raise RuntimeError(f"Checkpoint has no valid saved epoch: {checkpoint}")
        rows.append({
            "fold": fold,
            "best_epoch": epoch,
            "best_dice": float(state.get("best_dice", float("nan"))),
            "checkpoint": str(checkpoint),
        })
    return rows


def install_all_labeled_a3_loader(cases, use_cache: bool, appearance_augmentation=False):
    ordered = sorted(cases, key=lambda x: str(x["id"]))

    def build_final_dataloaders(config):
        base_train = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        spatial.ACTIVE_VARIANT = dict(A3_VARIANT)
        train_t = spatial.screening_spatial_transform(
            base_train,
            translate_voxels=4.0,
            rotate_degrees=0.0,
            scale_fraction=0.0,
        )
        if appearance_augmentation:
            train_t = add_mild_ultrasound_appearance_augmentation(train_t)
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )
        train_ds = cv.make_dataset(ordered, train_t, use_cache)
        val_ds = cv.make_dataset(ordered, val_t, use_cache)
        train_workers = int(getattr(config, "num_workers", 0))
        pin_memory = bool(torch.cuda.is_available())
        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            shuffle=True,
            num_workers=train_workers,
            pin_memory=pin_memory,
            persistent_workers=train_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        print(
            f"Final91 data loaders: train={len(train_ds)} | val={len(val_ds)} | "
            f"RAM cache={'ON (100%)' if use_cache else 'OFF'} | "
            f"train_workers={train_workers} | val_batch=1"
        )
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_final_dataloaders
    pipeline_module.build_dataloaders = build_final_dataloaders


def main(argv=None):
    p = argparse.ArgumentParser(description="Train final A3 model on all 91 HUMAN_GOLD cases")
    p.add_argument("--config", required=True)
    p.add_argument("--cv-dir", default=str(CV_DIR))
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    p.add_argument("--output-dir", default=str(OUTPUT))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--validation-every-n-epochs",
        type=int,
        default=10,
        help="Run full cached all-91 validation at this interval, plus epoch 1 and final epoch",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--appearance-augmentation",
        action="store_true",
        help="Add the locked mild ultrasound contrast/intensity/noise/smoothing bundle",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.validation_every_n_epochs < 1:
        p.error("--validation-every-n-epochs must be >=1")

    cv_dir = Path(args.cv_dir)
    audit_path = Path(args.audit_metadata)
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    if args.appearance_augmentation and output_dir.resolve() == OUTPUT.resolve():
        p.error("--appearance-augmentation requires a separate --output-dir from locked Final91 A3")

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False) or not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Final91 audit must be passing and provenance-enforced")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(audited_ids) != EXPECTED_TOTAL or len(set(audited_ids)) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected exactly {EXPECTED_TOTAL} audited IDs")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Final91 A3 requires prototype DynUNet Student+EMA")
    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest)
    if len(source_ids) != EXPECTED_SOURCE:
        raise RuntimeError("Frozen source is not original47")
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        raise RuntimeError("Live label directory changed after Final91 audit")

    epoch_rows = read_cv_best_epochs(cv_dir)
    best_epochs = np.asarray([x["best_epoch"] for x in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif final_checkpoint.exists():
        print(f"Final91 checkpoint already exists: {final_checkpoint}")
        print("Use --overwrite only if intentionally retraining from scratch.")
        return

    cv.apply_baseline(config, resize_size=128, epochs=final_epochs)
    config.seed = int(args.seed)
    config.use_cache_dataset = True
    config.validation_every_n_epochs = int(args.validation_every_n_epochs)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    variant_name = "a4-appearance" if args.appearance_augmentation else "a3"
    config.experiment_name = f"{config.experiment_name}-final91-{variant_name}-all91"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    cases = [by_id[x] for x in audited_ids]
    install_all_labeled_a3_loader(
        cases,
        True,
        appearance_augmentation=bool(args.appearance_augmentation),
    )

    print("=" * 112)
    print(f"FINAL91 {variant_name.upper()} — FULL TRAINING ON ALL 91 HUMAN_GOLD")
    print(f"CV source:             {cv_dir}")
    print(f"Audit:                 {audit_path}")
    print(f"Total HUMAN_GOLD:      {len(audited_ids)}")
    for row in epoch_rows:
        print(f"  Fold {row['fold']}: best epoch={row['best_epoch']} | best Dice={row['best_dice']:.4f}")
    print(f"CV best epochs:        {best_epochs.tolist()}")
    print(f"Median best epoch:     {median_best_epoch}")
    print(f"Final training epochs: {final_epochs}")
    print("Dataset cache:         train + validation cached 100% in host RAM")
    print(f"Full validation:       epoch 1, every {args.validation_every_n_epochs} epochs, and final epoch")
    print("Recipe:                DynUNet | 128^3 | DiceCE | AdamW 1e-4")
    print("A3 augmentation:       translation +/-4 p=.5 + LR flip p=.5")
    print(
        "Appearance augmentation: mild gamma/intensity/noise/smoothing bundle"
        if args.appearance_augmentation else
        "Appearance augmentation: OFF"
    )
    print("Deployment prediction: Student+EMA 50/50 @ .50 | no LCC")
    print("External31:            NOT ACCESSED")
    print("=" * 112)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected fixed-epoch latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": (
            "final91_a4_appearance_all91_training_v1"
            if args.appearance_augmentation else "final91_a3_all91_training_v1"
        ),
        "variant": "A4_APPEARANCE" if args.appearance_augmentation else "A3",
        "source_cv_dir": str(cv_dir),
        "source_audit": str(audit_path),
        "source_manifest": str(source_manifest),
        "n_total_human_labels": len(audited_ids),
        "all_human_label_ids": audited_ids,
        "seed": int(args.seed),
        "cv_best_epochs": [int(x) for x in best_epochs.tolist()],
        "cv_best_dice": [float(x["best_dice"]) for x in epoch_rows],
        "median_cv_best_epoch": median_best_epoch,
        "final_training_epochs": final_epochs,
        "validation_every_n_epochs": int(args.validation_every_n_epochs),
        "validation_cache": "CacheDataset cache_rate=1.0 in host RAM",
        "epoch_selection": "user_override" if args.epochs is not None else "median_final91_cv_best_epoch",
        "deployment_checkpoint": str(final_checkpoint),
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "prediction_source": "student_teacher_50_50_raw_ensemble",
        "prediction_threshold": 0.50,
        "recipe": {
            "architecture": "DynUNet", "resize_size": [128, 128, 128], "loss": "dice_ce",
            "learning_rate": 1e-4, "dropout": 0.0, "lambda_unsup": 0.0,
            "translation_voxels": 4.0, "translation_probability": 0.5,
            "lr_flip": True, "lr_flip_probability": 0.5, "postprocessing": "raw_no_lcc",
            "appearance_augmentation": (
                APPEARANCE_AUGMENTATION if args.appearance_augmentation else None
            ),
        },
        "external31_access": False,
        "warning": "Same-data validation is diagnostic only. Deployment weight is fixed-epoch latest checkpoint.",
    }
    metadata_path = output_dir / "final_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL91 FULL MODEL COMPLETE")
    print(f"Deployment checkpoint: {final_checkpoint}")
    print(f"Metadata:              {metadata_path}")
    print("Next: one locked evaluation on frozen external31.")


if __name__ == "__main__":
    main()
