#!/usr/bin/env python3
"""Evaluate one frozen HASSL checkpoint on the held-out test split.

This script is intended to be run only after model source and prediction threshold have
been selected on the validation split. It does not train or tune anything.
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import HausdorffDistanceMetric

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_or_create_frozen_splits, get_base_transforms, _strip_suffix
from hassl.training.trainer import build_network


def _main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def _test_cases(config):
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=getattr(config, "patient_id_regex", None),
        val_split=getattr(config, "val_split", 5),
    )
    test_ids = set(splits.get("test_ids", []))
    root = Path(config.data_dir)
    cases = []

    for img_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(img_path), config.image_suffix)
        if case_id not in test_ids:
            continue

        label_path = root / "labels" / f"{case_id}{config.label_suffix}"
        if not label_path.exists():
            label_path = Path(img_path).parent / f"{case_id}{config.label_suffix}"
        if label_path.exists():
            cases.append({"image": img_path, "label": str(label_path), "id": case_id})
    return cases


def _spacing_from_image(image, config):
    try:
        affine = image.affine[0] if image.affine.ndim == 3 else image.affine
        return tuple(float(torch.linalg.norm(affine[:3, i]).item()) for i in range(3))
    except Exception:
        return tuple(float(x) for x in config.spacing)


def _metrics(pred, target, spacing):
    tp = float((pred * target).sum().item())
    pred_sum = float(pred.sum().item())
    gt_sum = float(target.sum().item())
    eps = 1e-5

    dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
    precision = (tp + eps) / (pred_sum + eps)
    recall = (tp + eps) / (gt_sum + eps)
    pred_fg = float(pred.mean().item())
    gt_fg = float(target.mean().item())
    rve = abs(pred_sum - gt_sum) / (gt_sum + 1e-8) * 100.0

    hd95 = float("nan")
    try:
        hd = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
        hd(y_pred=pred, y=target, spacing=spacing)
        value = float(hd.aggregate().item())
        if math.isfinite(value):
            hd95 = value
    except Exception:
        pass

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_fg": gt_fg,
        "pred_fg": pred_fg,
        "rve": rve,
        "hd95": hd95,
        "gt_vox": gt_sum,
        "pred_vox": pred_sum,
    }


def evaluate(config_path, checkpoint_path, source, threshold):
    config = HASSLConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    cases = _test_cases(config)
    if not cases:
        raise RuntimeError("No held-out test cases found in the frozen split.")

    transform = get_base_transforms(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    loader = DataLoader(Dataset(cases, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "net_A" not in state:
        raise RuntimeError("Expected a HASSL checkpoint containing 'net_A'.")

    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()

    if source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"Checkpoint has no EMA teacher weights, cannot evaluate source={source!r}.")

    roi_size = tuple(config.patch_size) if getattr(config, "preprocessing_mode", "resize") == "patch" else tuple(config.spatial_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)

    print("=" * 72)
    print("HASSL held-out TEST evaluation")
    print(f"Config:      {config_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Test cases:  {len(cases)}")
    print(f"Source:      {source}")
    print(f"Threshold:   {threshold:.3f}")
    print(f"ROI size:    {roi_size}")
    print("=" * 72)

    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            spacing = _spacing_from_image(image, config)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                s_logits = _main_prediction(inferer(image, student))
                s_prob = torch.sigmoid(s_logits)

                if source == "student":
                    prob = s_prob
                else:
                    t_logits = _main_prediction(inferer(image, teacher))
                    t_prob = torch.sigmoid(t_logits)
                    prob = t_prob if source == "teacher" else 0.5 * (s_prob + t_prob)

            pred = (prob > float(threshold)).float()
            m = _metrics(pred, target, spacing)
            rows.append(m)

            print(
                f"{case_id}: Dice={m['dice']:.4f} | Prec={m['precision']:.4f} | "
                f"Rec={m['recall']:.4f} | GT_FG={m['gt_fg']:.5f} | "
                f"PRED_FG={m['pred_fg']:.5f} | RVE={m['rve']:.1f}% | "
                f"HD95={m['hd95']:.2f}mm | GTvox={m['gt_vox']:.0f} | PredVox={m['pred_vox']:.0f}"
            )

    if len(rows) > 1:
        print("\nTEST MEAN")
        for key in ("dice", "precision", "recall", "gt_fg", "pred_fg", "rve", "hd95"):
            vals = np.asarray([r[key] for r in rows], dtype=float)
            print(f"{key}: {np.nanmean(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frozen HASSL checkpoint on held-out test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", choices=["student", "teacher", "ensemble"], default="teacher")
    parser.add_argument("--threshold", type=float, default=0.98)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, args.source, args.threshold)
