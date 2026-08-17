#!/usr/bin/env python3
"""Visualize a frozen CV false-positive failure in model space.

This script is diagnostic only. It does not train, modify checkpoints, tune the
reporting threshold, or overwrite CV result files.

For the requested held-out case it reproduces deterministic inference from the
existing CV best checkpoint and exports axial/coronal/sagittal views selected by:
- maximum GT foreground
- maximum predicted foreground
- maximum false-positive foreground

Each selected slice shows the normalized ultrasound image, GT/prediction overlays,
false-positive mask, and probability map. It also creates a Fold training-reference
sheet using the maximum-GT axial slice of each training case so the failure can be
compared with the appearances seen during training.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.analyze_cv_false_positives as fp
import scripts.train_supervised_cv as cv


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_resize128_translation12")
DEFAULT_SPLIT_MANIFEST = Path("experiments/cv5_supervised_resize128/cv_splits.json")
DEFAULT_FOCUS_CASE = "80d0955124466d9b82337e7a17a8a2b5de9f4ec9244be0daa6eeb6f5014989d6"

AXES = {0: "axial", 1: "coronal", 2: "sagittal"}


def get_slice(volume, axis, index):
    if axis == 0:
        value = volume[index, :, :]
    elif axis == 1:
        value = volume[:, index, :]
    elif axis == 2:
        value = volume[:, :, index]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.rot90(value)


def max_slice(mask, axis):
    mask = np.asarray(mask)
    if axis == 0:
        scores = mask.sum(axis=(1, 2))
    elif axis == 1:
        scores = mask.sum(axis=(0, 2))
    elif axis == 2:
        scores = mask.sum(axis=(0, 1))
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    if scores.size == 0 or float(scores.max()) <= 0:
        return int(mask.shape[axis] // 2)
    return int(np.argmax(scores))


def safe_contour(ax, mask, color, label=None, linewidth=1.2):
    if np.any(mask):
        contour = ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)
        if label and contour.collections:
            contour.collections[0].set_label(label)


def normalize_for_display(image_slice):
    arr = np.asarray(image_slice, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo = float(np.percentile(arr[finite], 1))
    hi = float(np.percentile(arr[finite], 99))
    if hi <= lo:
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def save_focus_panel(image, target, pred, prob, axis, index, criterion, source, threshold, out_path):
    image_sl = normalize_for_display(get_slice(image, axis, index))
    gt_sl = get_slice(target, axis, index).astype(bool)
    pred_sl = get_slice(pred, axis, index).astype(bool)
    prob_sl = get_slice(prob, axis, index)
    fp_sl = np.logical_and(pred_sl, ~gt_sl)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(image_sl, cmap="gray", vmin=0, vmax=1)
    safe_contour(axes[0], gt_sl, "lime", "GT")
    safe_contour(axes[0], pred_sl, "red", "Pred")
    axes[0].set_title("Image + GT / prediction")

    axes[1].imshow(image_sl, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(pred_sl.astype(float), cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    safe_contour(axes[1], gt_sl, "lime")
    axes[1].set_title("Prediction (red), GT outline")

    axes[2].imshow(image_sl, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(fp_sl.astype(float), cmap="Reds", alpha=0.55, vmin=0, vmax=1)
    safe_contour(axes[2], gt_sl, "lime")
    axes[2].set_title(f"False positive mask ({int(fp_sl.sum())} vox/slice)")

    axes[3].imshow(image_sl, cmap="gray", vmin=0, vmax=1)
    heat = axes[3].imshow(prob_sl, cmap="magma", alpha=0.65, vmin=0, vmax=1)
    safe_contour(axes[3], gt_sl, "lime")
    fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Foreground probability")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{AXES[axis]} | {criterion} | slice={index} | source={source} | threshold={threshold:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def image_stats(case_id, image, target, role):
    target = target.astype(bool, copy=False)
    inside = image[target]
    outside = image[~target]
    center = fp.centroid(target)

    def mean_or_nan(values):
        return float(values.mean()) if values.size else float("nan")

    def std_or_nan(values):
        return float(values.std()) if values.size else float("nan")

    return {
        "case_id": case_id,
        "role": role,
        "gt_vox": int(target.sum()),
        "gt_fg": float(target.mean()),
        "gt_centroid_d": center[0],
        "gt_centroid_h": center[1],
        "gt_centroid_w": center[2],
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "gt_intensity_mean": mean_or_nan(inside),
        "gt_intensity_std": std_or_nan(inside),
        "background_intensity_mean": mean_or_nan(outside),
        "background_intensity_std": std_or_nan(outside),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_reference_sheet(loader, focus_image, focus_target, focus_case, out_path):
    examples = [(focus_case, focus_image, focus_target, "HELD-OUT FAILURE")]
    stats_rows = [image_stats(focus_case, focus_image, focus_target, "held_out")]

    for batch in loader:
        image = batch["image"][0, 0].detach().cpu().numpy()
        target = batch["label"][0, 0].detach().cpu().numpy() > 0.5
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
        examples.append((case_id, image, target, "fold training"))
        stats_rows.append(image_stats(case_id, image, target, "train"))

    cols = 3
    rows = int(math.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (case_id, image, target, role) in zip(axes, examples):
        index = max_slice(target, axis=0)
        image_sl = normalize_for_display(get_slice(image, 0, index))
        gt_sl = get_slice(target, 0, index).astype(bool)
        ax.imshow(image_sl, cmap="gray", vmin=0, vmax=1)
        safe_contour(ax, gt_sl, "lime")
        ax.set_title(f"{case_id[:10]}...\n{role} | axial GT-max={index}")
        ax.axis("off")

    for ax in axes[len(examples):]:
        ax.axis("off")

    fig.suptitle("Held-out failure vs Fold training appearances (model space)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stats_rows


def main():
    parser = argparse.ArgumentParser(
        description="Export visual diagnostics for a frozen CV false-positive case"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--focus-case", default=DEFAULT_FOCUS_CASE)
    parser.add_argument(
        "--source",
        choices=["student", "teacher", "ensemble"],
        default="ensemble",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, args.resize_size, epochs=100)
    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}

    manifest_path = Path(args.split_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, args.folds, manifest_path)

    if args.focus_case not in by_id:
        raise KeyError(f"Focus case not found in labeled cases: {args.focus_case}")

    fold_spec = next(
        (spec for spec in manifest["folds"] if args.focus_case in spec["val_ids"]),
        None,
    )
    if fold_spec is None:
        raise RuntimeError(f"Focus case is not held out by any fold: {args.focus_case}")
    fold_idx = int(fold_spec["fold"])

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.experiment_dir) / "false_positive_analysis" / "visuals" / args.focus_case[:12]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(
        roi_size=tuple(config.spatial_size),
        sw_batch_size=1,
        overlap=0.25,
    )

    checkpoint = Path(args.experiment_dir) / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")

    focus_loader = DataLoader(
        Dataset([by_id[args.focus_case]], transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(focus_loader))
    image_t = batch["image"].to(device)
    target_t = batch["label"].float().to(device)

    student, teacher = fp.load_models(config, checkpoint, device)
    if args.source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"source={args.source} requested but checkpoint has no teacher")

    with torch.no_grad(), torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
        student_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
        if args.source == "student":
            prob_t = student_prob
        else:
            teacher_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
            prob_t = teacher_prob if args.source == "teacher" else 0.5 * (student_prob + teacher_prob)

    image = image_t[0, 0].detach().float().cpu().numpy()
    target = target_t[0, 0].detach().cpu().numpy() > 0.5
    prob = prob_t[0, 0].detach().float().cpu().numpy()
    pred = prob > args.threshold
    false_positive = np.logical_and(pred, ~target)

    metrics = fp.scalar_case_metrics(pred, target)
    prob_metrics = fp.probability_stats(prob, target)
    selected = {}
    masks = {
        "max_gt": target,
        "max_prediction": pred,
        "max_false_positive": false_positive,
    }

    for axis, axis_name in AXES.items():
        selected[axis_name] = {}
        for criterion, mask in masks.items():
            index = max_slice(mask, axis)
            selected[axis_name][criterion] = index
            save_focus_panel(
                image,
                target,
                pred,
                prob,
                axis,
                index,
                criterion,
                args.source,
                args.threshold,
                output_dir / f"{axis_name}_{criterion}.png",
            )

    train_ids = fold_spec.get("train_ids")
    if train_ids is None:
        held_out = set(fold_spec["val_ids"])
        train_ids = [case["id"] for case in cases if case["id"] not in held_out]
    train_cases = [by_id[case_id] for case_id in train_ids]
    train_loader = DataLoader(
        Dataset(train_cases, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    stats_rows = save_training_reference_sheet(
        train_loader,
        image,
        target,
        args.focus_case,
        output_dir / f"fold_{fold_idx}_training_reference_axial.png",
    )
    write_csv(output_dir / "appearance_stats.csv", stats_rows)

    report = {
        "focus_case": args.focus_case,
        "fold": fold_idx,
        "source": args.source,
        "threshold": args.threshold,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "probability_stats": prob_metrics,
        "selected_slices": selected,
        "output_dir": str(output_dir),
        "note": "Diagnostic visualization only; do not use held-out views to tune and re-report unbiased CV performance.",
    }
    (output_dir / "visual_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("Frozen CV false-positive visual diagnostic")
    print(f"Focus case: {args.focus_case}")
    print(f"Fold: {fold_idx} | source={args.source} | threshold={args.threshold:.2f}")
    print(
        f"Dice={metrics['dice']:.4f} | Prec={metrics['precision']:.4f} | "
        f"Rec={metrics['recall']:.4f} | PredVox={metrics['pred_vox']} | RVE={metrics['rve']:.1f}%"
    )
    print(f"Outputs: {output_dir}")
    print(f"  {output_dir / 'visual_diagnostic.json'}")
    print(f"  {output_dir / 'appearance_stats.csv'}")
    print(f"  {output_dir / f'fold_{fold_idx}_training_reference_axial.png'}")
    print("  plus 9 axial/coronal/sagittal diagnostic panels")
    print("=" * 100)


if __name__ == "__main__":
    main()
