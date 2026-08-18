#!/usr/bin/env python3
"""Train the oracle-ROI Final62 model with the frozen supervised recipe.

Only the input field of view changes: cases are GT-cropped by build_oracle_roi_dataset.py.
Architecture, loss, optimizer, translation augmentation, threshold convention, and fixed-epoch
selection are kept aligned with the controlled Final62 experiment.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.pipeline as pipeline_module
import scripts.train_supervised_cv as cv
from scripts.train_final_supervised_round2_from_audit import (
    install_all_labeled_loader_hook,
    read_cv_best_epochs,
    read_json,
)

DEFAULT_ROI_DIR = Path("experiments/oracle_roi_dataset_62_v1")
DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/oracle_roi_final62_v1")


def main():
    p = argparse.ArgumentParser(description="Train Final62 on GT-derived oracle ROIs")
    p.add_argument("--config", required=True)
    p.add_argument("--roi-dir", default=str(DEFAULT_ROI_DIR))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    roi_dir = Path(args.roi_dir)
    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    roi_meta_path = roi_dir / "oracle_roi_dataset_metadata.json"
    if not roi_meta_path.exists():
        raise FileNotFoundError(f"Build oracle ROI dataset first: {roi_meta_path}")

    audit = read_json(audit_path)
    roi_meta = json.loads(roi_meta_path.read_text(encoding="utf-8"))
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    roi_ids = sorted(str(x) for x in roi_meta.get("case_ids", []))
    if roi_ids != audited_ids:
        raise RuntimeError("Oracle ROI case IDs do not exactly match the audited Final62 human-label set")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Oracle ROI A/B requires prototype student + EMA teacher training")

    cases = []
    for case_id in audited_ids:
        image = roi_dir / f"{case_id}{config.image_suffix}"
        label = roi_dir / "labels" / f"{case_id}{config.label_suffix}"
        if not image.exists() or not label.exists():
            raise FileNotFoundError(f"Missing oracle ROI pair for {case_id}: {image} | {label}")
        cases.append({"image": str(image), "label": str(label), "id": case_id})

    epoch_rows = read_cv_best_epochs(Path(args.round2_cv_dir))
    best_epochs = np.asarray([int(r["best_epoch"]) for r in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif final_checkpoint.exists():
            print(f"Oracle ROI final checkpoint already exists: {final_checkpoint}")
            print("Use --overwrite only if intentionally retraining.")
            return
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")

    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-oracle-roi-final62"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    install_all_labeled_loader_hook(cases, bool(getattr(config, "use_cache_dataset", True)))

    print("=" * 108)
    print("ORACLE ROI FINAL62 TRAINING — EXPERIMENT ONLY")
    print(f"Cases:              {len(cases)} audited HUMAN_GOLD")
    print(f"ROI dataset:        {roi_dir}")
    print(f"Margin each side:   {float(roi_meta.get('margin_fraction_each_side', float('nan'))):.2f}")
    print(f"CV best epochs:     {best_epochs.tolist()}")
    print(f"Median best epoch:  {median_best_epoch}")
    print(f"Final epochs:       {final_epochs}")
    print("Frozen recipe: DynUNet | resize128 | DiceCE | AdamW 1e-4 | dropout=0 | lambda_unsup=0")
    print("Augmentation: paired translation +/-12 vox, p=0.8; other augmentations OFF")
    print("ONLY experimental change: GT-derived ROI field of view")
    print("=" * 108)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "oracle_roi_final62_v1",
        "purpose": "oracle ROI upper-bound feasibility experiment",
        "source_audit": str(audit_path),
        "source_roi_dataset": str(roi_dir),
        "n_human_labels": len(cases),
        "case_ids": audited_ids,
        "oracle_margin_fraction_each_side": float(roi_meta.get("margin_fraction_each_side", 0.40)),
        "seed": int(args.seed),
        "cv_best_epochs_from_full_volume_round2": [int(x) for x in best_epochs.tolist()],
        "median_cv_best_epoch": median_best_epoch,
        "final_training_epochs": final_epochs,
        "epoch_selection": "user_override" if args.epochs is not None else "median_full_volume_round2_cv_best_epoch",
        "deployment_checkpoint": str(final_checkpoint),
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "evaluation_mode_planned": "EMA only at threshold 0.50",
        "controlled_difference": "GT-derived ROI field of view only",
        "warning": (
            "This model is trained on oracle GT-derived crops and is not a production model. It tests whether "
            "localization/resolution is a major bottleneck for the offline active-learning workflow."
        ),
    }
    metadata_path = output_dir / "oracle_roi_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nORACLE ROI TRAINING COMPLETE")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
