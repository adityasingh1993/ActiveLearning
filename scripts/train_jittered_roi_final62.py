#!/usr/bin/env python3
"""Train a robust offline ROI refiner on jittered GT-containing crops.

This experiment keeps the same DynUNet/loss/optimizer/EMA recipe as the oracle-ROI model but
changes the crop distribution. Each of the 62 audited HUMAN_GOLD source cases contributes the
same number of deterministic ROI variants, including one nominal symmetric crop plus several
asymmetric/context-jittered crops.

To avoid confounding crop diversity with a larger optimization budget, the default epoch count
is scaled so the approximate number of sample-level optimizer steps matches the original
62-case full-volume training duration:

    robust_epochs ~= median_full_volume_best_epoch * 62 / n_generated_samples

Use --epochs only for a deliberate follow-up experiment.
"""

import argparse
import json
import math
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

DEFAULT_ROI_DIR = Path("experiments/jittered_roi_dataset_62_v1")
DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/jittered_roi_final62_v1")


def main():
    p = argparse.ArgumentParser(description="Train robust Final62 offline ROI refiner")
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
    roi_meta_path = roi_dir / "jittered_roi_dataset_metadata.json"
    if not roi_meta_path.exists():
        raise FileNotFoundError(f"Build jittered ROI dataset first: {roi_meta_path}")

    audit = read_json(audit_path)
    roi_meta = json.loads(roi_meta_path.read_text(encoding="utf-8"))
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")

    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    source_ids = sorted(str(x) for x in roi_meta.get("source_case_ids", []))
    if source_ids != audited_ids:
        raise RuntimeError("Jittered ROI source IDs do not exactly match audited Final62 HUMAN_GOLD IDs")

    generated_ids = [str(x) for x in roi_meta.get("generated_ids", [])]
    variants_per_case = int(roi_meta.get("variants_per_case", 0))
    expected_generated = len(audited_ids) * variants_per_case
    if variants_per_case < 2 or len(generated_ids) != expected_generated:
        raise RuntimeError(
            f"Jittered ROI metadata is inconsistent: variants={variants_per_case}, "
            f"generated={len(generated_ids)}, expected={expected_generated}"
        )
    if len(set(generated_ids)) != len(generated_ids):
        raise RuntimeError("Duplicate generated IDs in jittered ROI metadata")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Jittered ROI experiment requires prototype student + EMA teacher")

    cases = []
    for generated_id in generated_ids:
        image = roi_dir / f"{generated_id}{config.image_suffix}"
        label = roi_dir / "labels" / f"{generated_id}{config.label_suffix}"
        if not image.exists() or not label.exists():
            raise FileNotFoundError(f"Missing jittered ROI pair: {image} | {label}")
        cases.append({"image": str(image), "label": str(label), "id": generated_id})

    epoch_rows = read_cv_best_epochs(Path(args.round2_cv_dir))
    best_epochs = np.asarray([int(r["best_epoch"]) for r in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))

    # Match approximate sample-level update budget to the original Final62 training. With
    # equal variants/source case this is equivalent to median_best_epoch / variants_per_case.
    reference_sample_exposures = median_best_epoch * len(audited_ids)
    matched_epochs = max(1, int(round(reference_sample_exposures / max(len(cases), 1))))
    final_epochs = int(args.epochs) if args.epochs is not None else matched_epochs
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif final_checkpoint.exists():
            print(f"Jittered ROI checkpoint already exists: {final_checkpoint}")
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
    config.experiment_name = f"{config.experiment_name}-jittered-roi-final62"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    install_all_labeled_loader_hook(cases, bool(getattr(config, "use_cache_dataset", True)))

    nominal_total_samples = median_best_epoch * len(audited_ids)
    robust_total_samples = final_epochs * len(cases)

    print("=" * 112)
    print("JITTERED ROI FINAL62 TRAINING — OFFLINE REFINER EXPERIMENT")
    print(f"Source HUMAN_GOLD cases:       {len(audited_ids)}")
    print(f"Variants per source case:      {variants_per_case}")
    print(f"Generated training samples:    {len(cases)}")
    print(f"Full-volume median best epoch: {median_best_epoch}")
    print(f"Matched robust epochs:         {matched_epochs}")
    print(f"Final robust epochs:           {final_epochs}")
    print(f"Reference sample exposures:    {nominal_total_samples}")
    print(f"Robust sample exposures:       {robust_total_samples}")
    print("Recipe: DynUNet | resize128 | DiceCE | AdamW 1e-4 | dropout=0 | lambda_unsup=0")
    print("Augmentation: paired translation +/-12 vox, p=0.8; other model augmentations OFF")
    print("Controlled change: GT-containing asymmetric ROI/context jitter during refiner training")
    print("Production model remains unchanged and single-stage.")
    print("=" * 112)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "jittered_roi_final62_v1",
        "purpose": "robust offline predicted-ROI refinement experiment",
        "source_audit": str(audit_path),
        "source_roi_dataset": str(roi_dir),
        "n_source_human_labels": len(audited_ids),
        "source_case_ids": audited_ids,
        "variants_per_case": variants_per_case,
        "n_generated_training_samples": len(cases),
        "seed": int(args.seed),
        "full_volume_cv_best_epochs": [int(x) for x in best_epochs.tolist()],
        "median_full_volume_best_epoch": median_best_epoch,
        "matched_epoch_count": matched_epochs,
        "final_training_epochs": final_epochs,
        "epoch_selection": "user_override" if args.epochs is not None else "matched_sample_exposure_budget",
        "reference_sample_exposures": int(nominal_total_samples),
        "robust_sample_exposures": int(robust_total_samples),
        "deployment_checkpoint": str(final_checkpoint),
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "evaluation_mode_planned": "EMA only at threshold 0.50 inside automatically predicted ROIs",
        "controlled_difference": "asymmetric GT-containing crop/context jitter during offline ROI-refiner training",
        "production_architecture_change": False,
        "warning": (
            "This checkpoint is an offline refinement experiment, not a production segmentation model. "
            "Its value must be judged by predicted-ROI external evaluation, not same-data validation."
        ),
    }
    metadata_path = output_dir / "jittered_roi_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nJITTERED ROI TRAINING COMPLETE")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
