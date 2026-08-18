#!/usr/bin/env python3
"""Train the final ROI experiment: a residual 2-channel guided refiner on existing OOF47 crops.

No new coarse CV is trained. This script reuses:
  experiments/guided_refiner_oof47_roi_dataset_v1/

Inputs
------
channel 0: ultrasound predicted-ROI crop
channel 1: leakage-safe OOF Round2 EMA probability crop

The network does NOT predict a fresh segmentation logit. It predicts delta_logit and returns:

    corrected_logit = logit(coarse_probability) + delta_logit

The delta segmentation heads are zero-initialized, so training starts from the existing coarse
prediction and learns only corrections. This remains an offline ROI feasibility experiment;
production remains a one-stage segmentation model.
"""

import os
import sys


def _consume_gpu_argument(argv):
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

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.pipeline as pipeline_module
import hassl.training.trainer as trainer_module
import scripts.train_supervised_cv as cv
from scripts.residual_guided_refiner_utils import build_residual_guided_dynunet
from scripts.train_final_supervised_round2_from_audit import read_cv_best_epochs
from scripts.train_guided_refiner_oof47_2ch import install_guided_loader_hook, read_csv


DEFAULT_DATASET_DIR = Path("experiments/guided_refiner_oof47_roi_dataset_v1")
DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/residual_guided_refiner_oof47_2ch_v1")
EXPECTED_CASES = 47


def main():
    p = argparse.ArgumentParser(description="Train residual 2-channel guided OOF47 ROI refiner")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "guided_refiner_oof47_roi_manifest.csv"
    metadata_path = dataset_dir / "guided_refiner_oof47_roi_metadata.json"
    rows = read_csv(manifest_path)
    if len(rows) != EXPECTED_CASES or len({str(r["case_id"]) for r in rows}) != EXPECTED_CASES:
        raise RuntimeError("Residual guided refiner dataset must contain exactly 47 unique OOF cases")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Residual guided refiner requires prototype DynUNet configuration")

    epoch_rows = read_cv_best_epochs(Path(args.round2_cv_dir))
    best_epochs = np.asarray([int(r["best_epoch"]) for r in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    cases = []
    for row in rows:
        for key in ["image_path", "coarse_probability_path", "label_path"]:
            if not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
        cases.append({
            "id": str(row["case_id"]),
            "image": str(row["image_path"]),
            "coarse": str(row["coarse_probability_path"]),
            "label": str(row["label_path"]),
        })

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif final_checkpoint.exists():
            print(f"Residual guided checkpoint already exists: {final_checkpoint}")
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
    config.experiment_name = f"{config.experiment_name}-residual-guided-oof47-2ch"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    # Patch only this process. Stable one-channel production code remains untouched.
    trainer_module.build_network = build_residual_guided_dynunet
    cv.build_network = build_residual_guided_dynunet
    install_guided_loader_hook(cases, config, args.resize_size)

    probe = build_residual_guided_dynunet(config.unet_backbone, config.num_classes, config.dropout)
    n_zero_heads = int(getattr(probe, "n_zero_initialized_heads", 0))
    del probe

    print("=" * 120)
    print("RESIDUAL 2-CHANNEL GUIDED ROI REFINER — OOF47 FINAL ROI EXPERIMENT")
    print(f"Cases:               {len(cases)} true-OOF predicted ROIs")
    print("Channels:            ultrasound + OOF EMA probability")
    print("Residual rule:       corrected_logit = coarse_logit + delta_logit")
    print(f"Zero-init heads:     {n_zero_heads} delta segmentation head(s)")
    print(f"CV best epochs:      {best_epochs.tolist()}")
    print(f"Median best epoch:   {median_best_epoch}")
    print(f"Final epochs:        {final_epochs}")
    print(f"Physical GPU:        {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("Recipe:              residual 2ch DynUNet | resize128 | DiceCE | AdamW 1e-4 | EMA | dropout=0")
    print("Augmentation:        paired image/coarse/label translation +/-12 vox, p=0.8")
    print("Validation:          same-data diagnostic only; fixed epoch selects final checkpoint")
    print("=" * 120)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected fixed-epoch latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    training_meta = {
        "version": "residual_guided_refiner_oof47_2ch_v1",
        "purpose": "final offline ROI residual-correction feasibility experiment",
        "n_training_cases": len(cases),
        "source_dataset": str(dataset_dir),
        "input_channels": ["ultrasound", "leakage_safe_oof_ema_probability"],
        "in_channels": 2,
        "residual_rule": "corrected_logit = logit(coarse_probability) + delta_logit",
        "coarse_probability_clip_epsilon": 1e-4,
        "zero_initialized_delta_segmentation_heads": n_zero_heads,
        "seed": int(args.seed),
        "cv_best_epochs_from_round2": [int(x) for x in best_epochs.tolist()],
        "final_training_epochs": final_epochs,
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "deployment_checkpoint": str(final_checkpoint),
        "prediction_threshold": 0.50,
        "dataset_localization_diagnostics": {
            "median_gt_coverage": dataset_meta.get("median_gt_coverage"),
            "gt_coverage_lt_090": dataset_meta.get("gt_coverage_lt_090"),
            "gt_coverage_lt_050": dataset_meta.get("gt_coverage_lt_050"),
        },
        "warning": (
            "This residual 2-channel model is an OFFLINE refiner experiment. It does not replace or alter "
            "the single-stage production segmentation requirement."
        ),
    }
    train_meta_path = output_dir / "residual_guided_refiner_training_metadata.json"
    train_meta_path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    print("\nRESIDUAL GUIDED REFINER TRAINING COMPLETE")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Metadata:   {train_meta_path}")


if __name__ == "__main__":
    main()
