#!/usr/bin/env python3
"""Evaluate student segmentation quality on the frozen labeled training split.

This diagnostic deliberately uses deterministic validation-style preprocessing rather
than the random/augmented training transform. In patch mode, the full post-Spacingd
volume is evaluated with SlidingWindowInferer using config.patch_size, so Train Dice
and Validation Dice are measured at the same physical voxel scale.
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
from hassl.data.data_engine import get_or_create_frozen_splits, get_base_transforms, _strip_suffix
from hassl.training.trainer import build_network


def _build_training_cases(config: HASSLConfig):
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
        if not label_path.exists():
            continue

        cases.append({"image": img_path, "label": str(label_path), "id": case_id})
    return cases


def _main_prediction(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def evaluate(config_path: str, checkpoint_path: str):
    config = HASSLConfig.from_yaml(config_path)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    cases = _build_training_cases(config)
    if not cases:
        raise RuntimeError("No labeled training cases were found after excluding val/test IDs.")

    # Deterministic transform. In patch mode this now preserves the full post-Spacingd
    # volume (padding only when needed); there is no random crop or strong augmentation.
    transform = get_base_transforms(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    loader = DataLoader(Dataset(cases, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    model = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state["net_A"] if isinstance(state, dict) and "net_A" in state else state
    model.load_state_dict(model_state)
    model.eval()

    roi_size = (
        tuple(config.patch_size)
        if getattr(config, "preprocessing_mode", "resize") == "patch"
        else tuple(config.spatial_size)
    )
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)
    threshold = float(getattr(config, "prediction_threshold", 0.5))

    dices, precisions, recalls = [], [], []
    gt_fracs, pred_fracs = [], []

    print(f"Evaluating {len(cases)} labeled TRAIN cases")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ROI size: {roi_size} | threshold={threshold:.3f}\n")

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = _main_prediction(inferer(image, model))

            if config.num_classes != 1:
                raise NotImplementedError(
                    "This diagnostic currently targets the binary HASSL segmentation path (num_classes=1)."
                )

            pred = (torch.sigmoid(logits) > threshold).float()

            tp = (pred * target).sum().item()
            pred_sum = pred.sum().item()
            gt_sum = target.sum().item()

            dice = (2.0 * tp + 1e-5) / (pred_sum + gt_sum + 1e-5)
            precision = (tp + 1e-5) / (pred_sum + 1e-5)
            recall = (tp + 1e-5) / (gt_sum + 1e-5)
            gt_fg = float(target.mean().item())
            pred_fg = float(pred.mean().item())

            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)
            gt_fracs.append(gt_fg)
            pred_fracs.append(pred_fg)

            print(
                f"{case_id}: Dice={dice:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | "
                f"GT_FG={gt_fg:.5f} | PRED_FG={pred_fg:.5f} | "
                f"GTvox={gt_sum:.0f} | PredVox={pred_sum:.0f}"
            )

    print("\nTRAIN SUMMARY")
    print(f"Dice:    {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Prec:    {np.mean(precisions):.4f}")
    print(f"Recall:  {np.mean(recalls):.4f}")
    print(f"GT_FG:   {np.mean(gt_fracs):.5f}")
    print(f"PRED_FG: {np.mean(pred_fracs):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deterministic Train Dice for a HASSL checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to HASSL YAML config")
    parser.add_argument(
        "--checkpoint",
        default="./experiments/checkpoints/best_checkpoint.pth",
        help="Path to checkpoint containing net_A",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
