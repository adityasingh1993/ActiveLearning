#!/usr/bin/env python3
"""Train the final Round-1 supervised model on all audited human labels.

The fixed training duration is derived from the completed controlled Round-1 CV folds:
median(saved best epoch across fold_0..fold_4), unless --epochs explicitly overrides it.

For the current first active-learning round this is expected to be 47 frozen + 8 new = 55
human labels, but the exact count is read from the passing audit metadata rather than hard-coded.

Important safeguards:
- the current label folder must exactly match the audited new-label IDs,
- all frozen source labels must still be present,
- training uses the same frozen translation-only recipe,
- early stopping is disabled,
- deployment checkpoint is copied from fixed-epoch round0_latest.pth, not same-data best_checkpoint.pth.
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
from scripts.audit_round1_labels import discover_round1_cases
from scripts.train_supervised_cv_47_translation import translation_only_transform


DEFAULT_ROUND1_CV_DIR = Path("experiments/round1_cv_55_translation12")
DEFAULT_AUDIT = Path("experiments/round1_supervised_55_translation12/round1_label_audit.json")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/final_supervised_round1_55_translation12")
EXPECTED_SOURCE_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_round1_best_epochs(round1_cv_dir: Path):
    rows = []
    for fold in range(5):
        checkpoint = round1_cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing Round-1 CV checkpoint: {checkpoint}. Complete all five folds first."
            )
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


def install_all_labeled_loader_hook(cases, use_cache: bool):
    ordered = sorted(cases, key=lambda case: str(case["id"]))

    def build_final_dataloaders(config):
        base_train = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        train_t = translation_only_transform(
            base_train,
            translate_voxels=12.0,
            rotate_degrees=0.0,
            scale_fraction=0.0,
        )
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )

        train_ds = cv.make_dataset(ordered, train_t, use_cache)
        # Same-data validation is diagnostic only. Fixed epoch count selects deployment.
        val_ds = cv.make_dataset(ordered, val_t, use_cache)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            shuffle=True,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_final_dataloaders
    pipeline_module.build_dataloaders = build_final_dataloaders


def main():
    parser = argparse.ArgumentParser(
        description="Train final Round-1 model on all audited human labels using median Round-1 CV best epoch"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--round1-cv-dir", default=str(DEFAULT_ROUND1_CV_DIR))
    parser.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    round1_cv_dir = Path(args.round1_cv_dir)
    audit_path = Path(args.audit_metadata)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Audit metadata does not record a passing label audit")
    new_ids = sorted(str(x) for x in audit.get("new_human_label_ids", []))
    expected_new = int(audit.get("expected_new_human_labels", len(new_ids)))
    if expected_new != len(new_ids):
        raise RuntimeError(
            f"Audit metadata inconsistent: expected_new={expected_new}, discovered={len(new_ids)}"
        )
    expected_total = EXPECTED_SOURCE_CASES + expected_new
    if int(audit.get("n_current_valid_human_labels", expected_total)) != expected_total:
        raise RuntimeError("Audit total does not equal 47 frozen + audited new labels")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Round-1 recipe requires prototype student + EMA teacher")

    _, source_ids, by_id, current_new_ids = discover_round1_cases(config, source_manifest_path)
    if len(source_ids) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_CASES} frozen source labels")
    if sorted(current_new_ids) != new_ids:
        raise RuntimeError(
            "Current label folder no longer matches the audited Round-1 set. "
            f"Audited={new_ids} Current={sorted(current_new_ids)}"
        )
    if len(by_id) != expected_total:
        raise RuntimeError(f"Expected exactly {expected_total} audited labels, found {len(by_id)}")

    epoch_rows = read_round1_best_epochs(round1_cv_dir)
    best_epochs = np.asarray([row["best_epoch"] for row in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        parser.error("--epochs must be >=1")

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif final_checkpoint.exists():
        print(f"Final Round-1 checkpoint already exists: {final_checkpoint}")
        print("Use --overwrite only if intentionally retraining from scratch.")
        return

    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-final-round1-{expected_total}labels-translation12"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    cases = [by_id[case_id] for case_id in sorted(by_id)]
    install_all_labeled_loader_hook(cases, bool(getattr(config, "use_cache_dataset", True)))

    print("=" * 104)
    print("FINAL ROUND-1 SUPERVISED TRAINING")
    print(f"Round-1 CV:        {round1_cv_dir}")
    print(f"Audit:             {audit_path}")
    print(f"Frozen labels:     {len(source_ids)}")
    print(f"New human labels:  {len(new_ids)}")
    print(f"Total labels:      {len(by_id)}")
    for row in epoch_rows:
        print(
            f"  Fold {row['fold']}: best epoch={row['best_epoch']} | best Dice={row['best_dice']:.4f}"
        )
    print(f"CV best epochs:    {best_epochs.tolist()}")
    print(f"Median best epoch: {median_best_epoch}")
    print(f"Final epochs:      {final_epochs}")
    print("Recipe: resize128 | DiceCE | AdamW 1e-4 | dropout=0 | lambda_unsup=0")
    print("Augmentation: paired translation +/-12 vox, p=0.8; flips/rotation/scale/intensity OFF")
    print("Deployment rule: fixed-epoch latest checkpoint; same-data best is diagnostic only")
    print("=" * 104)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected fixed-epoch latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "final_supervised_round1_from_audit_v1",
        "source_round1_cv_dir": str(round1_cv_dir),
        "source_audit": str(audit_path),
        "source_manifest": str(source_manifest_path),
        "n_frozen_source_labels": len(source_ids),
        "n_new_human_labels": len(new_ids),
        "n_total_human_labels": len(by_id),
        "new_human_label_ids": new_ids,
        "seed": int(args.seed),
        "cv_best_epochs": [int(x) for x in best_epochs.tolist()],
        "cv_best_dice": [float(row["best_dice"]) for row in epoch_rows],
        "median_cv_best_epoch": median_best_epoch,
        "final_training_epochs": final_epochs,
        "epoch_selection": "user_override" if args.epochs is not None else "median_round1_cv_best_epoch",
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
            "All audited human labels are used for gradient updates. Same-data validation diagnostics do not select "
            "the deployment checkpoint; final_checkpoint.pth is copied from the fixed-epoch latest checkpoint."
        ),
    }
    metadata_path = output_dir / "final_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL ROUND-1 MODEL COMPLETE")
    print(f"Deployment checkpoint: {final_checkpoint}")
    print(f"Metadata:              {metadata_path}")
    print("Next: pair this final checkpoint with a QC model rebuilt from Round-1 OOF predictions.")


if __name__ == "__main__":
    main()
