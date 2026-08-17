#!/usr/bin/env python3
"""Train the final frozen supervised segmentation model on all 47 valid labels.

The 5-fold CV experiment is used only to choose a fixed training duration. By default,
this script reads each fold's best_checkpoint.pth, extracts the saved best epoch, and uses
the median best epoch across folds. It then trains ONE model on all 47 labeled cases with
exactly the frozen translation-only recipe.

Important safeguards
--------------------
- The frozen 47-case CV manifest must match the currently visible labeled dataset.
- All 47 cases are used for gradient updates.
- Validation during this final fit reuses the same 47 cases ONLY to satisfy the existing
  trainer diagnostics. It is NOT used to choose the deployment checkpoint.
- Early stopping is disabled.
- Deployment checkpoint is final_checkpoint.pth, copied from round0_latest.pth after the
  fixed number of epochs. Do not deploy best_checkpoint.pth from this run because it is
  selected on training-set diagnostics.

Default output:
  experiments/final_supervised_47_translation12/
    checkpoints/round0_latest.pth
    checkpoints/final_checkpoint.pth
    final_training_metadata.json
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
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
import scripts.train_supervised_cv as cv
from scripts.train_supervised_cv_47_translation import translation_only_transform


DEFAULT_CV_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/final_supervised_47_translation12")
EXPECTED_CASES = 47


def read_cv_epoch_summary(cv_dir: Path, folds: int = 5):
    rows = []
    for fold in range(folds):
        checkpoint = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing CV best checkpoint: {checkpoint}\n"
                "Complete all five frozen CV folds before training the final model."
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"Malformed checkpoint: {checkpoint}")
        # HASSL save_checkpoint stores epoch+1, so this is the 1-based number of epochs
        # completed when the best validation Dice was observed.
        best_epoch = int(state.get("epoch", 0))
        if best_epoch <= 0:
            raise RuntimeError(f"Checkpoint has no valid saved epoch: {checkpoint}")
        rows.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_dice": float(state.get("best_dice", float("nan"))),
            "checkpoint": str(checkpoint),
        })
    return rows


def install_all_labeled_loader_hook(config: HASSLConfig, cases, use_cache: bool):
    by_id = {case["id"]: case for case in cases}
    ordered = [by_id[key] for key in sorted(by_id)]

    def build_final_dataloaders(_config):
        base_train = cv.ORIGINAL_GET_TRANSFORMS(
            _config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        train_t = translation_only_transform(
            base_train,
            translate_voxels=12.0,
            rotate_degrees=0.0,
            scale_fraction=0.0,
        )
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            _config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )

        train_ds = cv.make_dataset(ordered, train_t, use_cache)
        # Same labeled cases, deterministic transform. This loader is diagnostic only; fixed
        # epoch count means these metrics do not select the deployment checkpoint.
        val_ds = cv.make_dataset(ordered, val_t, use_cache)
        empty_unlabeled = Dataset([])

        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(_config, "batch_size", 1)),
            shuffle=True,
            num_workers=int(getattr(_config, "num_workers", 0)),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(empty_unlabeled, batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_final_dataloaders
    pipeline_module.build_dataloaders = build_final_dataloaders


def main():
    parser = argparse.ArgumentParser(
        description="Train one final supervised model on all 47 labels using the frozen CV recipe"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-dir", default=str(DEFAULT_CV_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override fixed epoch count. Default is median best epoch across the five CV folds.",
    )
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output_dir)
    manifest_path = cv_dir / "cv_splits.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    epoch_rows = read_cv_epoch_summary(cv_dir, folds=5)
    cv_best_epochs = np.asarray([row["best_epoch"] for row in epoch_rows], dtype=int)
    median_epoch = int(round(float(np.median(cv_best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_epoch
    if final_epochs < 1:
        parser.error("--epochs must be >= 1")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError(
            "The frozen 47-label QC feature schema was built from the prototype student/EMA-teacher "
            "recipe. Train the final model with compute_mode='prototype' for distribution continuity."
        )

    cases = cv.collect_cases(config)
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_CASES} valid labeled cases, found {len(cases)}. "
            "Resolve dataset provenance before final training."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, 5, manifest_path)

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif final_checkpoint.exists():
        print(f"Final checkpoint already exists: {final_checkpoint}")
        print("Use --overwrite only if you intentionally want to retrain from scratch.")
        return

    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-final47-translation12"

    # Reproducible final fit.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    install_all_labeled_loader_hook(config, cases, bool(getattr(config, "use_cache_dataset", True)))

    print("=" * 100)
    print("FINAL 47-LABEL SUPERVISED TRAINING")
    print(f"CV directory:       {cv_dir}")
    for row in epoch_rows:
        print(
            f"  Fold {row['fold']}: best epoch={row['best_epoch']} | "
            f"best Dice={row['best_dice']:.4f}"
        )
    print(f"CV best epochs:     {cv_best_epochs.tolist()}")
    print(f"Median best epoch:  {median_epoch}")
    print(f"Final train epochs: {final_epochs}")
    print(f"Labeled cases:      {len(cases)} / {EXPECTED_CASES}")
    print("Recipe: resize128 | DiceCE | AdamW 1e-4 | dropout=0 | lambda_unsup=0")
    print("Augmentation: paired translation +/-12 vox, p=0.8; flips/rotation/scale/intensity OFF")
    print("Checkpoint rule: deploy FIXED-EPOCH latest, not training-set 'best'")
    print("=" * 100)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected final latest checkpoint was not written: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "final_supervised_47_translation12_v1",
        "source_cv_dir": str(cv_dir),
        "source_manifest": str(manifest_path),
        "n_labeled_cases": len(cases),
        "seed": int(args.seed),
        "cv_best_epochs": [int(x) for x in cv_best_epochs.tolist()],
        "cv_best_dice": [float(row["best_dice"]) for row in epoch_rows],
        "median_cv_best_epoch": int(median_epoch),
        "final_training_epochs": int(final_epochs),
        "epoch_selection": "user_override" if args.epochs is not None else "median_cv_best_epoch",
        "deployment_checkpoint": str(final_checkpoint),
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "prediction_threshold": float(config.prediction_threshold),
        "recipe": {
            "preprocessing_mode": "resize",
            "spatial_size": [int(args.resize_size)] * 3,
            "spacing": [float(x) for x in config.spacing],
            "loss": "dice_ce",
            "optimizer": "AdamW",
            "learning_rate": float(config.train_lr),
            "weight_decay": float(config.train_weight_decay),
            "dropout": float(config.dropout),
            "lambda_unsup": float(config.lambda_unsup),
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
        },
        "warning": (
            "All 47 labeled cases were used for training. Training-set validation diagnostics were "
            "not used to select the deployment checkpoint; final_checkpoint.pth is the fixed-epoch latest."
        ),
    }
    metadata_path = output_dir / "final_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL MODEL COMPLETE")
    print(f"Deployment checkpoint: {final_checkpoint}")
    print(f"Metadata:              {metadata_path}")
    print("Next: run this checkpoint + the QC bundle over the unlabeled pool; development AUTO_ACCEPT")
    print("      decisions remain pseudo-label candidates until a future locked validation set confirms safety.")


if __name__ == "__main__":
    main()
