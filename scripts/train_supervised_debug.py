#!/usr/bin/env python3
"""Run a minimal supervised HASSL diagnostic with deterministic Train Dice.

This script intentionally removes semi-supervised and augmentation complexity so we can
answer one question first: can the base binary segmentation model fit the labeled data?
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Allow direct execution via `python scripts/train_supervised_debug.py` from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
from hassl.data.data_engine import get_or_create_frozen_splits, get_base_transforms, _strip_suffix
from hassl.training.trainer import HASSLTrainer


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
    only_case_id = getattr(config, "_debug_overfit_case_id", None)

    root = Path(config.data_dir)
    cases = []
    for img_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(img_path), config.image_suffix)
        if case_id in held_out:
            continue
        if only_case_id is not None and case_id != only_case_id:
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


def _select_nonempty_train_index(config, labeled_dataset, deterministic_transform):
    """Find a training case whose deterministic transformed label contains foreground."""
    source_data = getattr(labeled_dataset, "data", None)
    if source_data is None:
        raise RuntimeError("Overfit-one diagnostic requires a dataset exposing its source `data` list.")

    for idx, case in enumerate(source_data):
        sample = deterministic_transform(dict(case))
        label = torch.as_tensor(sample["label"])
        fg_voxels = float(label.sum().item())
        total_voxels = int(label.numel())
        if fg_voxels > 0:
            case_id = case.get("id") or _strip_suffix(os.path.basename(case["image"]), config.image_suffix)
            fg_fraction = fg_voxels / max(1, total_voxels)
            return idx, str(case_id), fg_voxels, total_voxels, fg_fraction

    raise RuntimeError("Could not find a non-empty labeled training case for --overfit-one.")


def _install_minimal_data_hook(overfit_one=False, overfit_repeats=16):
    """Disable strong augmentation, normalize empty unlabeled data, optionally overfit one case."""
    original_get_transforms = data_engine.get_base_transforms

    def get_transforms_no_strong_aug(config, keys=["image", "label"], is_training=False, apply_strong_aug=True):
        return original_get_transforms(
            config,
            keys=keys,
            is_training=is_training,
            apply_strong_aug=False,
        )

    data_engine.get_base_transforms = get_transforms_no_strong_aug

    original_build = data_engine.build_dataloaders

    def build_with_empty_unlabeled(config):
        labeled_loader, unlabeled_loader, val_loader, val_transforms = original_build(config)

        if overfit_one:
            deterministic_transform = original_get_transforms(
                config,
                keys=["image", "label"],
                is_training=False,
                apply_strong_aug=False,
            )
            idx, case_id, fg_voxels, total_voxels, fg_fraction = _select_nonempty_train_index(
                config, labeled_loader.dataset, deterministic_transform
            )
            config._debug_overfit_case_id = case_id

            # Repeat the same source case to create multiple optimizer steps per epoch.
            # In patch mode this also produces fresh random crops on each __getitem__.
            # In resize mode it deliberately repeats the identical whole 128^3 volume so
            # we can test whether the network can memorize one fixed image/label pair.
            repeated = Subset(labeled_loader.dataset, [idx] * int(overfit_repeats))
            labeled_loader = DataLoader(
                repeated,
                batch_size=1,
                shuffle=True,
                num_workers=config.num_workers,
            )
            print(
                f"  [OVERFIT-ONE] case={case_id} | deterministic FG={fg_voxels:.0f}/{total_voxels} "
                f"({fg_fraction:.5%}) | updates/epoch={len(repeated)}"
            )

        if unlabeled_loader is None or len(unlabeled_loader.dataset) == 0:
            unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return labeled_loader, unlabeled_loader, val_loader, val_transforms

    data_engine.build_dataloaders = build_with_empty_unlabeled
    pipeline_module.build_dataloaders = build_with_empty_unlabeled


def _apply_diagnostic_overrides(config):
    """Make the supervised sanity check intentionally simple and easy to overfit."""
    config.loss_type = "dice_ce"
    config.include_boundary = False
    config.lambda_unsup = 0.0
    config.pos_neg_ratio = 1.0
    config.dropout = 0.0
    config.train_lr = 1e-4
    config.lr_scheduler = "none"
    config.lr_warmup_epochs = 0


def _append_namespace(path_value, suffix):
    path = Path(path_value)
    return str(path.with_name(path.name + suffix))


def _apply_random_init_namespace(config):
    """Use isolated state so random-init A/B cannot auto-resume or auto-load SSL weights."""
    config.checkpoint_dir = _append_namespace(config.checkpoint_dir, "_random_init")
    config.cache_dir = _append_namespace(config.cache_dir, "_random_init")
    config.experiment_name = f"{config.experiment_name}-random-init"


def _apply_resize_namespace(config, resize_size):
    """Switch the diagnostic to whole-volume Spacingd -> Resize training with isolated state."""
    resize_size = int(resize_size)
    config.preprocessing_mode = "resize"
    config.spatial_size = (resize_size, resize_size, resize_size)
    config.checkpoint_dir = _append_namespace(config.checkpoint_dir, f"_resize{resize_size}")
    config.cache_dir = _append_namespace(config.cache_dir, f"_resize{resize_size}")
    config.experiment_name = f"{config.experiment_name}-resize{resize_size}"


def _apply_overfit_one_namespace(config):
    """Keep one-case sanity-check checkpoints separate and prevent val early stopping."""
    config.checkpoint_dir = _append_namespace(config.checkpoint_dir, "_overfit1")
    config.cache_dir = _append_namespace(config.cache_dir, "_overfit1")
    config.experiment_name = f"{config.experiment_name}-overfit1"
    config.use_early_stopping = False


def main():
    parser = argparse.ArgumentParser(description="Run minimal supervised HASSL debug training with deterministic Train Dice")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Force random initialization and use isolated cache/checkpoint directories for a clean A/B test.",
    )
    parser.add_argument(
        "--overfit-one",
        action="store_true",
        help="Train repeatedly on one confirmed non-empty labeled case as a decisive pipeline sanity check.",
    )
    parser.add_argument(
        "--overfit-repeats",
        type=int,
        default=16,
        help="Number of optimizer updates per epoch from the selected case (default: 16).",
    )
    parser.add_argument(
        "--resize-overfit",
        action="store_true",
        help="Use whole-volume Spacingd -> Resize training for the overfit diagnostic instead of patch crops.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=128,
        help="Cubic whole-volume resize dimension for --resize-overfit (default: 128).",
    )
    args = parser.parse_args()

    if args.random_init and args.pretrained:
        parser.error("--random-init and --pretrained are mutually exclusive")
    if args.overfit_repeats < 1:
        parser.error("--overfit-repeats must be >= 1")
    if args.resize_size < 16:
        parser.error("--resize-size must be >= 16")
    if args.resize_overfit and not args.overfit_one:
        parser.error("--resize-overfit requires --overfit-one so the resize test stays a controlled one-case diagnostic")

    config = HASSLConfig.from_yaml(args.config)
    _apply_diagnostic_overrides(config)
    if args.random_init:
        _apply_random_init_namespace(config)
    if args.resize_overfit:
        _apply_resize_namespace(config, args.resize_size)
    if args.overfit_one:
        _apply_overfit_one_namespace(config)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    _install_train_metric_hook()
    _install_minimal_data_hook(overfit_one=args.overfit_one, overfit_repeats=args.overfit_repeats)

    init_mode = "RANDOM INIT" if args.random_init else (f"PRETRAINED: {args.pretrained}" if args.pretrained else "DEFAULT/NO EXPLICIT PRETRAIN")

    print("=" * 60)
    print("HASSL minimal supervised diagnostic run")
    print(f"Config: {args.config}")
    print(f"Initialization: {init_mode}")
    print(f"Preprocessing mode: {config.preprocessing_mode}")
    print(f"Spacing: {config.spacing}")
    if config.preprocessing_mode == "resize":
        print(f"Whole-volume resize: {tuple(config.spatial_size)}")
    else:
        print(f"Patch size: {config.patch_size}")
        print(f"Pos/neg ratio: {config.pos_neg_ratio}:1")
    print(f"Loss: {config.loss_type}")
    print(f"Dropout: {config.dropout}")
    print(f"LR: {config.train_lr} (fixed)")
    print("Strong augmentation: disabled")
    if args.overfit_one:
        print(f"Overfit-one: enabled ({args.overfit_repeats} updates/epoch; early stopping disabled)")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Cache dir: {config.cache_dir}")
    print("Train Dice will be evaluated deterministically every epoch.")
    print("=" * 60)

    # In random-init mode we deliberately pass None. The isolated checkpoint directory
    # prevents run_train from finding either ssl_pretrained.pth or round0_latest.pth.
    pretrained_weights = None if args.random_init else args.pretrained
    pipeline_module.run_train(config, round_num=args.round, pretrained_weights=pretrained_weights)


if __name__ == "__main__":
    main()
