#!/usr/bin/env python3
"""Visualize one HASSL test prediction in model space.

Shows image, GT, prediction, error map, and probability map for axial/coronal/sagittal
slices selected by maximum GT foreground. Also prints overlap metrics and probability
statistics inside/outside the GT to diagnose zero-Dice failures.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms, _strip_suffix
from hassl.training.trainer import build_network


def main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def find_case(config, case_id):
    root = Path(config.data_dir)
    image_path = None

    for path in glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True):
        found_id = _strip_suffix(os.path.basename(path), config.image_suffix)
        if found_id == case_id:
            image_path = path
            break

    if image_path is None:
        raise FileNotFoundError(f"Could not find image for case: {case_id}")

    label_path = root / "labels" / f"{case_id}{config.label_suffix}"
    if not label_path.exists():
        label_path = Path(image_path).parent / f"{case_id}{config.label_suffix}"
    if not label_path.exists():
        raise FileNotFoundError(f"Could not find label for case: {case_id}")

    return {"image": str(image_path), "label": str(label_path), "id": case_id}


def normalize_slice(image):
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1, 99])
    return np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)


def overlay_mask(image, mask, color):
    rgb = np.stack([image] * 3, axis=-1)
    overlay = rgb.copy()
    for c in range(3):
        overlay[..., c][mask] = color[c]
    alpha = 0.55
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * overlay[mask]
    return rgb


def make_error_map(gt, pred):
    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)
    out[tp] = [0.0, 1.0, 0.0]
    out[fp] = [1.0, 0.0, 0.0]
    out[fn] = [0.0, 0.3, 1.0]
    return out


def get_axis_slice(volume, axis, index):
    if axis == "axial":
        return volume[index, :, :]
    if axis == "coronal":
        return volume[:, index, :]
    if axis == "sagittal":
        return volume[:, :, index]
    raise ValueError(axis)


def best_gt_slice(gt, axis):
    if axis == "axial":
        sums = gt.sum(axis=(1, 2))
    elif axis == "coronal":
        sums = gt.sum(axis=(0, 2))
    elif axis == "sagittal":
        sums = gt.sum(axis=(0, 1))
    else:
        raise ValueError(axis)
    return int(np.argmax(sums)) if sums.max() > 0 else len(sums) // 2


def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    eps = 1e-8
    return {
        "dice": (2 * tp + eps) / (pred_sum + gt_sum + eps),
        "precision": (tp + eps) / (pred_sum + eps),
        "recall": (tp + eps) / (gt_sum + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_vox": gt_sum,
        "pred_vox": pred_sum,
    }


def probability_statistics(prob, gt):
    gt = gt.astype(bool)
    inside = prob[gt]
    outside = prob[~gt]

    print("\nPROBABILITY ANALYSIS")
    print("-" * 60)
    if len(inside) > 0:
        print("Inside GT:")
        print(f"  Mean: {inside.mean():.6f}")
        print(f"  Max:  {inside.max():.6f}")
        print(f"  P95:  {np.percentile(inside, 95):.6f}")
        print(f"  P99:  {np.percentile(inside, 99):.6f}")
    if len(outside) > 0:
        print("\nOutside GT:")
        print(f"  Mean: {outside.mean():.6f}")
        print(f"  Max:  {outside.max():.6f}")
        print(f"  P99:  {np.percentile(outside, 99):.6f}")


def visualize(image, gt, pred, prob, case_id, threshold, source, output_path):
    axes = ["axial", "coronal", "sagittal"]
    fig, plots = plt.subplots(3, 5, figsize=(20, 12))

    for row, axis in enumerate(axes):
        index = best_gt_slice(gt, axis)
        img_slice = get_axis_slice(image, axis, index)
        gt_slice = get_axis_slice(gt, axis, index).astype(bool)
        pred_slice = get_axis_slice(pred, axis, index).astype(bool)
        prob_slice = get_axis_slice(prob, axis, index)
        img_norm = normalize_slice(img_slice)

        plots[row, 0].imshow(img_norm, cmap="gray")
        plots[row, 0].set_title(f"{axis.capitalize()} image\nslice={index}")

        plots[row, 1].imshow(overlay_mask(img_norm, gt_slice, [0, 1, 0]))
        plots[row, 1].set_title("GT")

        plots[row, 2].imshow(overlay_mask(img_norm, pred_slice, [1, 0, 0]))
        plots[row, 2].set_title(f"Prediction\nthr={threshold:.3f}")

        plots[row, 3].imshow(make_error_map(gt_slice, pred_slice))
        plots[row, 3].set_title("Error\nGreen=TP Red=FP Blue=FN")

        im = plots[row, 4].imshow(prob_slice, vmin=0, vmax=1, cmap="inferno")
        plots[row, 4].set_title(f"{source.capitalize()} probability")

        for col in range(5):
            plots[row, col].axis("off")

    plt.colorbar(im, ax=plots[:, 4], fraction=0.02, pad=0.02)
    plt.suptitle(f"Test case: {case_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved visualization: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a frozen HASSL test prediction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source", choices=["student", "teacher", "ensemble"], default="teacher")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--output", default="test_prediction_visualization.png")
    args = parser.parse_args()

    config = HASSLConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"Device: {device}")

    case = find_case(config, args.case_id)
    print("\nCASE")
    print(f"Image: {case['image']}")
    print(f"Label: {case['label']}")

    transform = get_base_transforms(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    loader = DataLoader(Dataset([case], transform=transform), batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "net_A" not in checkpoint:
        raise RuntimeError("Expected a HASSL checkpoint containing 'net_A'.")

    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(checkpoint["net_A"])
    student.eval()

    teacher = None
    if "teacher" in checkpoint:
        teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(checkpoint["teacher"])
        teacher.eval()

    if args.source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError("Checkpoint does not contain teacher weights.")

    roi_size = tuple(config.patch_size) if config.preprocessing_mode == "patch" else tuple(config.spatial_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)

    batch = next(iter(loader))
    image = batch["image"].to(device)
    target = batch["label"].float().to(device)

    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            student_logits = main_prediction(inferer(image, student))
            student_prob = torch.sigmoid(student_logits)

            if args.source == "student":
                prob = student_prob
            else:
                teacher_logits = main_prediction(inferer(image, teacher))
                teacher_prob = torch.sigmoid(teacher_logits)
                prob = teacher_prob if args.source == "teacher" else 0.5 * (student_prob + teacher_prob)

    pred = (prob > args.threshold).float()

    image_np = image[0, 0].detach().cpu().numpy()
    gt_np = target[0, 0].detach().cpu().numpy() > 0.5
    pred_np = pred[0, 0].detach().cpu().numpy() > 0.5
    prob_np = prob[0, 0].detach().float().cpu().numpy()

    metrics = compute_metrics(pred_np, gt_np)
    print("\nSEGMENTATION METRICS")
    print("-" * 60)
    print(f"Dice:        {metrics['dice']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"TP voxels:   {metrics['tp']}")
    print(f"FP voxels:   {metrics['fp']}")
    print(f"FN voxels:   {metrics['fn']}")
    print(f"GT voxels:   {metrics['gt_vox']}")
    print(f"Pred voxels: {metrics['pred_vox']}")

    probability_statistics(prob_np, gt_np)
    visualize(
        image=image_np,
        gt=gt_np,
        pred=pred_np,
        prob=prob_np,
        case_id=args.case_id,
        threshold=args.threshold,
        source=args.source,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
