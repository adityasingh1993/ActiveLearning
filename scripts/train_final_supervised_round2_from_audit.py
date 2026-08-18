#!/usr/bin/env python3
"""Train the final Round-2 supervised model on all audited human labels.

The fixed training duration is derived from the completed controlled Round-2 CV folds:
median(saved best epoch across fold_0..fold_4), unless --epochs explicitly overrides it.

The exact label count is provenance-derived from the passing Round-2 audit. For the current
controlled cycle this is expected to be 47 frozen + 8 Round-1 + 7 Round-2 = 62 human labels,
but no count is hard-coded.

Safeguards
----------
- the current central human-label set must exactly match the passing Round-2 audit,
- Round-2 audit selection provenance must be enforced,
- pseudo-labels and external-validation labels are excluded,
- training uses the frozen translation-only supervised recipe,
- early stopping is disabled,
- final_checkpoint.pth is copied from the fixed-epoch latest checkpoint; same-data validation
  is diagnostic only and does not select deployment weights.
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

DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/final_supervised_round2_62_translation12")
EXPECTED_SOURCE_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_cv_best_epochs(round2_cv_dir: Path):
    rows = []
    for fold in range(5):
        checkpoint = round2_cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing Round-2 CV checkpoint: {checkpoint}. Complete all five folds first."
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
    p = argparse.ArgumentParser(
        description="Train final Round-2 model on all provenance-safe audited human labels"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    round2_cv_dir = Path(args.round2_cv_dir)
    audit_path = Path(args.audit_metadata)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata does not record a passing label audit")
    if not audit.get("selection_provenance_enforced", False):
        raise RuntimeError(
            "Round-2 audit did not enforce selection provenance. Refusing controlled final training."
        )
    unselected = [str(x) for x in audit.get("unselected_new_label_ids", [])]
    if unselected:
        raise RuntimeError(f"Round-2 audit contains unselected new labels: {unselected}")

    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    round1_ids = sorted(str(x) for x in audit.get("round1_human_label_ids", []))
    round2_ids = sorted(str(x) for x in audit.get("round2_new_human_label_ids", []))
    n_total = int(audit.get("n_current_valid_human_labels", len(audited_ids)))
    if not audited_ids or len(audited_ids) != n_total:
        raise RuntimeError(
            f"Round-2 audit ID/count mismatch: ids={len(audited_ids)}, reported total={n_total}"
        )
    if n_total != EXPECTED_SOURCE_CASES + len(round1_ids) + len(round2_ids):
        raise RuntimeError(
            "Round-2 audit total is inconsistent with source + Round-1 + Round-2 human labels"
        )
    if len(round2_ids) < 1:
        raise RuntimeError("Round-2 audit contains no newly added human labels")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Frozen supervised recipe requires prototype student + EMA teacher")

    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest_path)
    if len(source_ids) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_CASES} frozen source labels")
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        added = sorted(set(current_ids) - set(audited_ids))
        missing = sorted(set(audited_ids) - set(current_ids))
        raise RuntimeError(
            "Current central human-label set no longer matches the passing Round-2 audit.\n"
            f"Unexpected current labels: {added}\nMissing audited labels: {missing}\n"
            "Re-run the provenance-safe Round-2 audit before final training."
        )

    epoch_rows = read_cv_best_epochs(round2_cv_dir)
    best_epochs = np.asarray([row["best_epoch"] for row in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif final_checkpoint.exists():
        print(f"Final Round-2 checkpoint already exists: {final_checkpoint}")
        print("Use --overwrite only if intentionally retraining from scratch.")
        return

    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-final-round2-{n_total}labels-translation12"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    cases = [by_id[case_id] for case_id in audited_ids]
    install_all_labeled_loader_hook(cases, bool(getattr(config, "use_cache_dataset", True)))

    print("=" * 108)
    print("FINAL ROUND-2 SUPERVISED TRAINING")
    print(f"Round-2 CV:          {round2_cv_dir}")
    print(f"Round-2 audit:       {audit_path}")
    print(f"Frozen source:       {len(source_ids)}")
    print(f"Round-1 human:       {len(round1_ids)}")
    print(f"Round-2 new human:   {len(round2_ids)}")
    print(f"Total human labels:  {n_total}")
    for row in epoch_rows:
        print(f"  Fold {row['fold']}: best epoch={row['best_epoch']} | best Dice={row['best_dice']:.4f}")
    print(f"CV best epochs:      {best_epochs.tolist()}")
    print(f"Median best epoch:   {median_best_epoch}")
    print(f"Final epochs:        {final_epochs}")
    print("Recipe: resize128 | DiceCE | AdamW 1e-4 | dropout=0 | lambda_unsup=0")
    print("Augmentation: paired translation +/-12 vox, p=0.8; other augmentations OFF")
    print("Segmentation operating point remains raw ensemble threshold=0.50")
    print("Deployment rule: fixed-epoch latest checkpoint; same-data best is diagnostic only")
    print("=" * 108)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected fixed-epoch latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "final_supervised_round2_from_audit_v1",
        "source_round2_cv_dir": str(round2_cv_dir),
        "source_round2_audit": str(audit_path),
        "source_manifest": str(source_manifest_path),
        "n_frozen_source_labels": len(source_ids),
        "n_round1_human_labels": len(round1_ids),
        "n_round2_new_human_labels": len(round2_ids),
        "n_total_human_labels": n_total,
        "round1_human_label_ids": round1_ids,
        "round2_new_human_label_ids": round2_ids,
        "all_human_label_ids": audited_ids,
        "selected_unlabeled_ids": [str(x) for x in audit.get("selected_unlabeled_ids", [])],
        "seed": int(args.seed),
        "cv_best_epochs": [int(x) for x in best_epochs.tolist()],
        "cv_best_dice": [float(row["best_dice"]) for row in epoch_rows],
        "median_cv_best_epoch": median_best_epoch,
        "final_training_epochs": final_epochs,
        "epoch_selection": "user_override" if args.epochs is not None else "median_round2_cv_best_epoch",
        "deployment_checkpoint": str(final_checkpoint),
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "prediction_threshold": 0.50,
        "prediction_source": "student_teacher_50_50_raw_ensemble",
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
            "All and only provenance-audited human labels are used for gradient updates. Same-data validation "
            "does not choose the deployment checkpoint; final_checkpoint.pth is copied from the fixed-epoch latest checkpoint."
        ),
    }
    metadata_path = output_dir / "final_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL ROUND-2 MODEL COMPLETE")
    print(f"Deployment checkpoint: {final_checkpoint}")
    print(f"Metadata:              {metadata_path}")
    print("Next: compare final Round-1 vs final Round-2 on the frozen external 31-case benchmark.")


if __name__ == "__main__":
    main()
