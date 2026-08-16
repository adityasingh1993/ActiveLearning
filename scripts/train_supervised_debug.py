#!/usr/bin/env python3
"""Run HASSL supervised diagnostics with deterministic Train Dice every epoch.

Use this while isolating the base segmentation path. It keeps the production trainer,
but wraps validation so the student is also evaluated on the frozen labeled training
split with the exact same deterministic preprocessing and sliding-window scale used
for validation.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
from hassl.data.data_engine import get_or_create_frozen_splits, get_base_transforms, _strip_suffix
from hassl.training.trainer import HASSLTrainer
from hassl.pipeline import run_train


def _training_cases(config):
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=getattr(config, "patient_id_regex", None),
        val_split=getattr(config, "val_split", 5),
    )
    held_out = set(splits.get("val_ids", [])) | set(splits.get("test_ids", []))

    root = Path(config.data_dir)
    cases = []
    for img_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(img_path), config.image_suffix)
        if case_id in held_out:
            continue

        label_path = root / "labels" / f"{case_id}{config.label_suffix}"
        if not label_path.exists():
            label_path = Path(img_path).parent / f"{case_id}{config.label_suffix}"
        if label_path.exists():
            cases.append({"image": img_path, "label": str(label_path), "id": case_id})
    return cases


def _main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def _get_train_eval_loader(trainer):
    loader = getattr(trainer, "_deterministic_train_eval_loader", None)
    if loader is not None:
        return loader

    cases = _training_cases(trainer.config)
    transform = get_base_transforms(
        trainer.config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    loader = DataLoader(
        Dataset(cases, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    trainer._deterministic_train_eval_loader = loader
    return loader


@torch.no_grad()
def _compute_train_metrics(trainer):
    loader = _get_train_eval_loader(trainer)
    if len(loader.dataset) == 0:
        return {
            "train_dice_eval": float("nan"),
            "train_precision_eval": float("nan"),
            "train_recall_eval": float("nan"),
            "train_gt_fg_eval": float("nan"),
            "train_pred_fg_eval": float("nan"),
        }

    trainer.net_A.eval()
    roi_size = (
        tuple(trainer.config.patch_size)
        if getattr(trainer.config, "preprocessing_mode", "resize") == "patch"
        else tuple(trainer.config.spatial_size)
    )
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)
    threshold = float(getattr(trainer.config, "prediction_threshold", 0.5))

    dices, precisions, recalls, gt_fracs, pred_fracs = [], [], [], [], []

    for batch in loader:
        image = batch["image"].to(trainer.device)
        target = batch["label"].float().to(trainer.device)

        with torch.amp.autocast(trainer.device_type, enabled=(trainer.device_type == "cuda")):
            logits = _main_prediction(inferer(image, trainer.net_A))

        if trainer.num_classes != 1:
            raise NotImplementedError("Supervised diagnostic Train Dice currently supports num_classes=1.")

        pred = (torch.sigmoid(logits) > threshold).float()
        tp = (pred * target).sum().item()
        pred_sum = pred.sum().item()
        gt_sum = target.sum().item()

        dices.append((2.0 * tp + 1e-5) / (pred_sum + gt_sum + 1e-5))
        precisions.append((tp + 1e-5) / (pred_sum + 1e-5))
        recalls.append((tp + 1e-5) / (gt_sum + 1e-5))
        gt_fracs.append(float(target.mean().item()))
        pred_fracs.append(float(pred.mean().item()))

    return {
        "train_dice_eval": float(np.mean(dices)),
        "train_precision_eval": float(np.mean(precisions)),
        "train_recall_eval": float(np.mean(recalls)),
        "train_gt_fg_eval": float(np.mean(gt_fracs)),
        "train_pred_fg_eval": float(np.mean(pred_fracs)),
    }


def _install_train_metric_hook():
    original_validate = HASSLTrainer.validate

    def validate_with_train_metrics(self, epoch=0, should_log_image=False):
        train_metrics = _compute_train_metrics(self)
        print(
            f"  [TRAIN EVAL] Epoch {epoch:3d} | "
            f"Dice: {train_metrics['train_dice_eval']:.4f} | "
            f"Prec: {train_metrics['train_precision_eval']:.4f} | "
            f"Rec: {train_metrics['train_recall_eval']:.4f} | "
            f"GT_FG: {train_metrics['train_gt_fg_eval']:.5f} | "
            f"PRED_FG: {train_metrics['train_pred_fg_eval']:.5f}"
        )

        metrics = original_validate(self, epoch=epoch, should_log_image=should_log_image)
        metrics.update(train_metrics)
        return metrics

    HASSLTrainer.validate = validate_with_train_metrics


def _install_labeled_only_loader_hook():
    """Return an empty DataLoader instead of None when no unlabeled cases exist.

    The production trainer already understands an empty unlabeled dataset, but the current
    pipeline status print accesses ``unlabeled_loader.dataset`` unconditionally. Supplying
    an empty loader keeps this diagnostic compatible without inventing unlabeled samples.
    """
    original_build = data_engine.build_dataloaders

    def build_with_empty_unlabeled(config):
        labeled_loader, unlabeled_loader, val_loader, val_transforms = original_build(config)
        if unlabeled_loader is None:
            unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return labeled_loader, unlabeled_loader, val_loader, val_transforms

    data_engine.build_dataloaders = build_with_empty_unlabeled


def main():
    parser = argparse.ArgumentParser(description="Run supervised HASSL debug training with deterministic Train Dice")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--pretrained", default=None)
    args = parser.parse_args()

    config = HASSLConfig.from_yaml(args.config)
    _install_train_metric_hook()
    _install_labeled_only_loader_hook()

    print("=" * 60)
    print("HASSL supervised diagnostic run")
    print(f"Config: {args.config}")
    print(f"Patch size: {config.patch_size}")
    print(f"Spacing: {config.spacing}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print("Train Dice will be evaluated deterministically every epoch.")
    print("=" * 60)

    run_train(config, round_num=args.round, pretrained_weights=args.pretrained)


if __name__ == "__main__":
    main()
