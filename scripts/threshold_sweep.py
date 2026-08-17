#!/usr/bin/env python3
"""Sweep binary prediction thresholds on the frozen validation split.

Loads one HASSL checkpoint, runs inference once per validation case, then reuses the
probability maps to compare student, EMA teacher, and their 50/50 ensemble across
multiple thresholds. No training is performed.
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
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


def _validation_cases(config):
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=getattr(config, "patient_id_regex", None),
        val_split=getattr(config, "val_split", 5),
    )
    val_ids = set(splits.get("val_ids", []))
    root = Path(config.data_dir)
    cases = []
    for img_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(img_path), config.image_suffix)
        if case_id not in val_ids:
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


def _case_metrics(pred, target, spacing):
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
        "pred_fg": pred_fg,
        "gt_fg": gt_fg,
        "rve": rve,
        "hd95": hd95,
    }


def evaluate(config_path, checkpoint_path, thresholds):
    config = HASSLConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    cases = _validation_cases(config)
    if not cases:
        raise RuntimeError("No validation cases found in the frozen split.")

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

    roi_size = tuple(config.patch_size) if getattr(config, "preprocessing_mode", "resize") == "patch" else tuple(config.spatial_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)

    sources = ["student"]
    if teacher is not None:
        sources += ["teacher", "ensemble"]

    results = {
        source: {float(t): defaultdict(list) for t in thresholds}
        for source in sources
    }

    print("=" * 88)
    print("HASSL validation threshold sweep")
    print(f"Config:      {config_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Val cases:   {len(cases)}")
    print(f"ROI size:    {roi_size}")
    print(f"Sources:     {', '.join(sources)}")
    print(f"Thresholds:  {', '.join(f'{t:.2f}' for t in thresholds)}")
    print("=" * 88)

    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            target = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            spacing = _spacing_from_image(image, config)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                s_logits = _main_prediction(inferer(image, student))
                s_prob = torch.sigmoid(s_logits)
                probs = {"student": s_prob}

                if teacher is not None:
                    t_logits = _main_prediction(inferer(image, teacher))
                    t_prob = torch.sigmoid(t_logits)
                    probs["teacher"] = t_prob
                    probs["ensemble"] = 0.5 * (s_prob + t_prob)

            for source, prob in probs.items():
                for threshold in thresholds:
                    pred = (prob > float(threshold)).float()
                    metrics = _case_metrics(pred, target, spacing)
                    for key, value in metrics.items():
                        results[source][float(threshold)][key].append(value)

            print(f"[{idx}/{len(cases)}] {case_id} complete")

    print("\nMEAN VALIDATION RESULTS")
    print("Source     Thr    Dice    Prec    Recall  PRED_FG   RVE(%)   HD95(mm)")
    print("-" * 78)

    best = None
    for source in sources:
        for threshold in thresholds:
            values = results[source][float(threshold)]
            mean = {}
            for key, items in values.items():
                arr = np.asarray(items, dtype=float)
                mean[key] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

            print(
                f"{source:<10} {threshold:>4.2f}  "
                f"{mean['dice']:>6.4f}  {mean['precision']:>6.4f}  {mean['recall']:>6.4f}  "
                f"{mean['pred_fg']:>7.5f}  {mean['rve']:>7.1f}  {mean['hd95']:>8.2f}"
            )

            candidate = (mean["dice"], source, float(threshold), mean)
            if best is None or candidate[0] > best[0]:
                best = candidate
        print()

    if best is not None:
        dice, source, threshold, mean = best
        print("BEST BY MEAN VALIDATION DICE")
        print(
            f"source={source} | threshold={threshold:.2f} | Dice={dice:.4f} | "
            f"Prec={mean['precision']:.4f} | Recall={mean['recall']:.4f} | "
            f"PRED_FG={mean['pred_fg']:.5f} | RVE={mean['rve']:.1f}% | HD95={mean['hd95']:.2f}mm"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep prediction thresholds on HASSL validation data")
    parser.add_argument("--config", required=True, help="Config used by the resized supervised baseline")
    parser.add_argument("--checkpoint", required=True, help="Frozen best_checkpoint.pth to evaluate")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        help="Thresholds to sweep (default: 0.30 0.40 0.50 0.60 0.70 0.80 0.90)",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.thresholds)
