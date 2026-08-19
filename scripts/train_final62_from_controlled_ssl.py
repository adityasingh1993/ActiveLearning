#!/usr/bin/env python3
"""Fine-tune the exact Final62 supervised recipe from the controlled SSL checkpoint.

The intended comparison is one-variable-at-a-time:

    Final62 resize128, no SSL   vs   Final62 resize128, controlled SSL

All 62 provenance-audited HUMAN_GOLD labels, fixed epoch rule, augmentation, optimizer,
threshold, and EMA behavior remain the same as the established Final62 baseline.

Important transfer rules
------------------------
- SSL reconstruction/deep-supervision output heads are NOT transferred.
- Compatible representation weights are transferred into the student.
- EMA teacher is synchronized to the SSL-initialized student BEFORE fine-tuning.
- No unlabeled loss is used in this final supervised fit (lambda_unsup=0).
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
import hassl.data.data_engine as data_engine
from hassl.ssl.controlled import initialize_trainer_from_ssl
from hassl.tracking import ExperimentTracker
from hassl.training.trainer import HASSLTrainer
import scripts.train_supervised_cv as cv
from scripts.audit_round1_labels import discover_round1_cases
from scripts.train_final_supervised_round2_from_audit import (
    DEFAULT_AUDIT,
    DEFAULT_ROUND2_CV_DIR,
    DEFAULT_SOURCE_MANIFEST,
    EXPECTED_SOURCE_CASES,
    install_all_labeled_loader_hook,
    read_cv_best_epochs,
    read_json,
)


DEFAULT_SSL_CHECKPOINT = Path(
    "experiments/ssl_pretrain_final62_pool_128_v1/checkpoints/ssl_pretrained_controlled.pth"
)
DEFAULT_OUTPUT_DIR = Path("experiments/final_supervised_round2_62_translation12_ssl128")


def main():
    p = argparse.ArgumentParser(description="Train Final62 resize128 from controlled SSL initialization")
    p.add_argument("--config", required=True)
    p.add_argument("--pretrained", default=str(DEFAULT_SSL_CHECKPOINT))
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.resize_size != 128:
        p.error("This controlled SSL comparison is intentionally frozen to --resize-size 128")

    pretrained = Path(args.pretrained)
    if not pretrained.exists():
        raise FileNotFoundError(
            f"Controlled SSL checkpoint not found: {pretrained}\n"
            "Run scripts/train_controlled_ssl_final62_128.py first."
        )

    round2_cv_dir = Path(args.round2_cv_dir)
    audit_path = Path(args.audit_metadata)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata does not record a passing label audit")
    if not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Round-2 audit did not enforce selection provenance")
    unselected = [str(x) for x in audit.get("unselected_new_label_ids", [])]
    if unselected:
        raise RuntimeError(f"Round-2 audit contains unselected new labels: {unselected}")

    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    round1_ids = sorted(str(x) for x in audit.get("round1_human_label_ids", []))
    round2_ids = sorted(str(x) for x in audit.get("round2_new_human_label_ids", []))
    n_total = int(audit.get("n_current_valid_human_labels", len(audited_ids)))
    if len(audited_ids) != n_total or n_total != 62:
        raise RuntimeError(f"Controlled Final62 experiment requires exactly 62 audited labels, found {n_total}")
    if n_total != EXPECTED_SOURCE_CASES + len(round1_ids) + len(round2_ids):
        raise RuntimeError("Audit total is inconsistent with source + Round-1 + Round-2 labels")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Controlled Final62 recipe requires prototype student + EMA teacher")

    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest_path)
    if len(source_ids) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_CASES} frozen source labels")
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        added = sorted(set(current_ids) - set(audited_ids))
        missing = sorted(set(audited_ids) - set(current_ids))
        raise RuntimeError(
            "Current central HUMAN_GOLD set no longer matches the passing Final62 audit.\n"
            f"Unexpected labels: {added}\nMissing labels: {missing}"
        )

    epoch_rows = read_cv_best_epochs(round2_cv_dir)
    best_epochs = np.asarray([row["best_epoch"] for row in epoch_rows], dtype=int)
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
            print(f"Final62 SSL checkpoint already exists: {final_checkpoint}")
            print("Use --overwrite only if intentionally retraining.")
            return
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")

    # Exact frozen Final62 recipe. This sets resize mode, DiceCE, LR 1e-4, dropout=0,
    # lambda_unsup=0, and the requested fixed epoch budget.
    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-final62-resize128-controlled-ssl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    cases = [by_id[case_id] for case_id in audited_ids]
    install_all_labeled_loader_hook(cases, bool(getattr(config, "use_cache_dataset", True)))
    labeled_loader, unlabeled_loader, val_loader, val_transform = data_engine.build_dataloaders(config)
    if len(labeled_loader.dataset) != 62 or len(val_loader.dataset) != 62:
        raise RuntimeError(
            f"Expected same-data diagnostic train/val loaders of 62; got "
            f"train={len(labeled_loader.dataset)}, val={len(val_loader.dataset)}"
        )
    if unlabeled_loader is None or len(unlabeled_loader.dataset) != 0:
        raise RuntimeError("Controlled Final62 fit must have an empty unlabeled loader")

    tracker = ExperimentTracker(
        backend=config.tracker,
        project=config.project_name,
        run_name=f"{config.experiment_name}_train_round0",
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_config(config.to_dict())
    tracker.log_metrics({"al_round": 0, "ssl_initialized": 1}, step=0)

    trainer = HASSLTrainer(
        config=config,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        tracker=tracker,
        pretrained_weights=None,
        val_transform=val_transform,
    )
    transfer_report = initialize_trainer_from_ssl(trainer, pretrained)

    print("=" * 116)
    print("FINAL62 CONTROLLED SSL FINE-TUNING — RESIZE128")
    print(f"HUMAN_GOLD labels:        {n_total}")
    print(f"SSL checkpoint:           {pretrained}")
    print(f"Transferred tensors:      {transfer_report['transferred_tensor_count']}/{transfer_report['target_tensor_count']}")
    print(f"Skipped SSL task heads:   {transfer_report['skipped_task_head_count']}")
    print(f"Shape-mismatch skips:     {transfer_report['skipped_shape_count']}")
    print("EMA teacher sync:         PASS — copied from SSL-initialized student")
    print(f"CV best epochs:           {best_epochs.tolist()}")
    print(f"Median/final epochs:      {median_best_epoch}/{final_epochs}")
    print("Recipe:                   SAME Final62 | resize128 | DiceCE | AdamW 1e-4 | EMA | dropout=0")
    print("Augmentation:             translation +/-12 vox p=0.8; other augmentation OFF")
    print(f"Physical GPU:             {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("Only intended change:     SSL representation initialization")
    print("=" * 116)

    trainer.train(num_epochs=final_epochs)
    latest = checkpoint_dir / "round0_latest.pth"
    trainer.save_checkpoint(str(latest), epoch=final_epochs - 1)
    tracker.finish()

    if not latest.exists():
        raise FileNotFoundError(latest)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "final62_controlled_ssl128_v1",
        "comparison_baseline": "Final62 resize128 no-SSL external31 EMA mean Dice 0.7367",
        "source_ssl_checkpoint": str(pretrained),
        "ssl_transfer_report": transfer_report,
        "source_round2_cv_dir": str(round2_cv_dir),
        "source_round2_audit": str(audit_path),
        "source_manifest": str(source_manifest_path),
        "n_total_human_labels": n_total,
        "all_human_label_ids": audited_ids,
        "seed": int(args.seed),
        "cv_best_epochs": [int(x) for x in best_epochs.tolist()],
        "median_cv_best_epoch": median_best_epoch,
        "final_training_epochs": final_epochs,
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "deployment_checkpoint": str(final_checkpoint),
        "prediction_threshold": 0.50,
        "recipe": {
            "preprocessing_mode": "resize",
            "spatial_size": [128, 128, 128],
            "spacing": [float(x) for x in config.spacing],
            "loss": "dice_ce",
            "optimizer": "AdamW",
            "learning_rate": float(config.train_lr),
            "weight_decay": float(config.train_weight_decay),
            "dropout": float(config.dropout),
            "lambda_unsup": float(config.lambda_unsup),
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "ema_decay": float(config.ema_decay),
        },
        "controlled_difference": (
            "Compared with Final62 baseline, only SSL feature initialization is intended to change. "
            "SSL task-specific output heads are skipped and EMA teacher is synchronized after transfer."
        ),
    }
    metadata_path = output_dir / "final_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL62 CONTROLLED SSL MODEL COMPLETE")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Metadata:   {metadata_path}")
    print("Next: run benchmark_final62_student_ema_ensemble_external31.py with this checkpoint.")


if __name__ == "__main__":
    main()
