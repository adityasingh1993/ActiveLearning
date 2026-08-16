#!/usr/bin/env python3
"""Audit labeled HASSL volumes for dataset/domain-shift diagnostics.

Reports, for every labeled case:
- frozen split assignment (train/val/test)
- native image shape and affine-derived spacing
- native image intensity percentiles
- post-Orientationd/Spacingd shape and GT foreground size
- final deterministic model-space shape and GT foreground size
- model-space GT centroid (voxel and normalized coordinates)

The goal is to identify acquisition/FOV/spacing/intensity/target-location outliers before
changing the model after a held-out generalization failure.
"""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import ORIENTATIOND_RAS_LABELS
from hassl.config import HASSLConfig
from hassl.data.data_engine import (
    NormalizeLabelsInDatasetd,
    _strip_suffix,
    get_base_transforms,
    get_or_create_frozen_splits,
)


def _find_labeled_cases(config):
    root = Path(config.data_dir)
    cases = []
    for image_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(image_path), config.image_suffix)
        label_path = root / "labels" / f"{case_id}{config.label_suffix}"
        if not label_path.exists():
            label_path = Path(image_path).parent / f"{case_id}{config.label_suffix}"
        if label_path.exists():
            cases.append({"image": image_path, "label": str(label_path), "id": case_id})
    return cases


def _split_lookup(config):
    splits = get_or_create_frozen_splits(
        config.data_dir,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        seed=config.seed,
        patient_id_regex=getattr(config, "patient_id_regex", None),
        val_split=getattr(config, "val_split", 5),
    )
    lookup = {}
    for case_id in splits.get("initial_train_ids", []):
        lookup[case_id] = "train"
    for case_id in splits.get("val_ids", []):
        lookup[case_id] = "val"
    for case_id in splits.get("test_ids", []):
        lookup[case_id] = "test"
    return lookup


def _native_transform():
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])


def _post_spacing_transform(config):
    transforms = [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=ORIENTATIOND_RAS_LABELS),
    ]

    use_spacingd = bool(config.spacing) and all(float(s) >= 0.1 for s in config.spacing)
    if use_spacingd:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=config.spacing,
                mode=("bilinear", "nearest"),
            )
        )

    num_classes = getattr(config, "num_classes", 1)
    label_names = getattr(config, "label_names", None)
    if num_classes == 1:
        transforms.append(AsDiscreted(keys=["label"], threshold=0.5))
    elif label_names and isinstance(label_names, dict):
        transforms.append(NormalizeLabelsInDatasetd(keys=["label"], label_names=label_names))

    return Compose(transforms)


def _shape_spatial(tensor):
    shape = tuple(int(v) for v in tensor.shape)
    return shape[-3:] if len(shape) >= 3 else shape


def _shape_str(shape):
    return "x".join(str(int(v)) for v in shape)


def _spacing_from_tensor(tensor):
    try:
        affine = tensor.affine
        if torch.is_tensor(affine) and affine.ndim == 3:
            affine = affine[0]
        spacing = [float(torch.linalg.norm(affine[:3, i]).item()) for i in range(3)]
        return tuple(spacing)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))


def _mask3d(label):
    label_t = torch.as_tensor(label)
    mask = label_t > 0.5
    while mask.ndim > 3:
        mask = mask.any(dim=0)
    return mask


def _mask_stats(label):
    mask = _mask3d(label)
    total = int(mask.numel())
    fg = int(mask.sum().item())
    frac = fg / max(total, 1)

    if fg == 0:
        centroid_vox = (float("nan"),) * 3
        centroid_norm = (float("nan"),) * 3
    else:
        coords = torch.nonzero(mask, as_tuple=False).float()
        centroid = coords.mean(dim=0)
        centroid_vox = tuple(float(v.item()) for v in centroid)
        dims = torch.tensor(mask.shape, dtype=torch.float32)
        denom = torch.clamp(dims - 1.0, min=1.0)
        norm = centroid / denom
        centroid_norm = tuple(float(v.item()) for v in norm)

    return fg, total, frac, centroid_vox, centroid_norm


def _percentiles(image):
    arr = torch.as_tensor(image).detach().cpu().float().numpy().reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"),) * 3
    return tuple(float(v) for v in np.percentile(arr, [1, 50, 99]))


def _fmt(v, digits=5):
    if isinstance(v, float) and not math.isfinite(v):
        return "nan"
    return f"{float(v):.{digits}f}"


def _row_for_case(case, split, native_t, spaced_t, model_t):
    native = native_t(dict(case))
    spaced = spaced_t(dict(case))
    model = model_t(dict(case))

    native_shape = _shape_spatial(native["image"])
    native_spacing = _spacing_from_tensor(native["image"])
    native_p1, native_p50, native_p99 = _percentiles(native["image"])
    native_fg, native_total, native_frac, _, _ = _mask_stats(native["label"])

    spaced_shape = _shape_spatial(spaced["image"])
    spaced_fg, spaced_total, spaced_frac, _, _ = _mask_stats(spaced["label"])

    model_shape = _shape_spatial(model["image"])
    model_fg, model_total, model_frac, centroid_vox, centroid_norm = _mask_stats(model["label"])
    model_p1, model_p50, model_p99 = _percentiles(model["image"])

    return {
        "case_id": case["id"],
        "split": split,
        "native_shape": _shape_str(native_shape),
        "native_spacing_x": native_spacing[0],
        "native_spacing_y": native_spacing[1],
        "native_spacing_z": native_spacing[2],
        "native_intensity_p1": native_p1,
        "native_intensity_p50": native_p50,
        "native_intensity_p99": native_p99,
        "native_gt_vox": native_fg,
        "native_gt_fraction": native_frac,
        "post_spacing_shape": _shape_str(spaced_shape),
        "post_spacing_gt_vox": spaced_fg,
        "post_spacing_gt_fraction": spaced_frac,
        "model_shape": _shape_str(model_shape),
        "model_gt_vox": model_fg,
        "model_gt_fraction": model_frac,
        "centroid_d": centroid_vox[0],
        "centroid_h": centroid_vox[1],
        "centroid_w": centroid_vox[2],
        "centroid_d_norm": centroid_norm[0],
        "centroid_h_norm": centroid_norm[1],
        "centroid_w_norm": centroid_norm[2],
        "model_intensity_p1": model_p1,
        "model_intensity_p50": model_p50,
        "model_intensity_p99": model_p99,
        "image_path": case["image"],
        "label_path": case["label"],
    }


def _print_rows(rows, focus_case=None):
    print("\nLABELED DATASET AUDIT")
    print("=" * 150)
    print(
        f"{'split':<6} {'case':<18} {'native shape':<15} {'spacing':<24} "
        f"{'native GT%':>10} {'spaced shape':<15} {'model GT%':>10} "
        f"{'centroid norm (D,H,W)':<28} {'native p1/p50/p99':<28}"
    )
    print("-" * 150)

    for r in rows:
        case_display = r["case_id"][:18]
        spacing = "/".join(_fmt(r[k], 4) for k in ("native_spacing_x", "native_spacing_y", "native_spacing_z"))
        centroid = "(" + ",".join(_fmt(r[k], 3) for k in ("centroid_d_norm", "centroid_h_norm", "centroid_w_norm")) + ")"
        intensity = "/".join(_fmt(r[k], 2) for k in ("native_intensity_p1", "native_intensity_p50", "native_intensity_p99"))
        mark = "  <== FOCUS" if focus_case and r["case_id"] == focus_case else ""
        print(
            f"{r['split']:<6} {case_display:<18} {r['native_shape']:<15} {spacing:<24} "
            f"{100.0*r['native_gt_fraction']:>9.4f}% {r['post_spacing_shape']:<15} "
            f"{100.0*r['model_gt_fraction']:>9.4f}% {centroid:<28} {intensity:<28}{mark}"
        )


def _focus_report(rows, focus_case):
    if not focus_case:
        return
    focus = next((r for r in rows if r["case_id"] == focus_case), None)
    if focus is None:
        print(f"\n[WARN] Focus case {focus_case!r} was not found in the labeled dataset.")
        return

    train_rows = [r for r in rows if r["split"] == "train"]
    if not train_rows:
        return

    keys = [
        "native_spacing_x", "native_spacing_y", "native_spacing_z",
        "native_intensity_p1", "native_intensity_p50", "native_intensity_p99",
        "native_gt_fraction", "post_spacing_gt_fraction", "model_gt_fraction",
        "centroid_d_norm", "centroid_h_norm", "centroid_w_norm",
    ]

    print("\nFOCUS CASE VS TRAIN DISTRIBUTION")
    print("=" * 84)
    print(f"Focus: {focus_case} ({focus['split']})")
    print(f"{'metric':<28} {'focus':>12} {'train min':>12} {'train median':>14} {'train max':>12}")
    print("-" * 84)

    for key in keys:
        vals = np.asarray([float(r[key]) for r in train_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        fv = float(focus[key])
        print(
            f"{key:<28} {_fmt(fv, 6):>12} {_fmt(np.min(vals), 6):>12} "
            f"{_fmt(np.median(vals), 6):>14} {_fmt(np.max(vals), 6):>12}"
        )

    print("\nInterpretation hint: values outside the train min/max are direct acquisition/target-distribution outliers;")
    print("large shifts within the range can still matter with only seven training cases.")


def main():
    parser = argparse.ArgumentParser(description="Audit labeled HASSL dataset for domain-shift/outlier diagnostics")
    parser.add_argument("--config", required=True, help="Config used for the supervised baseline")
    parser.add_argument("--focus-case", default=None, help="Optional case ID to compare directly against train distribution")
    parser.add_argument(
        "--output",
        default="experiments/logs/labeled_dataset_audit.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    config = HASSLConfig.from_yaml(args.config)
    cases = _find_labeled_cases(config)
    if not cases:
        raise RuntimeError("No labeled cases found.")

    split_lookup = _split_lookup(config)
    native_t = _native_transform()
    spaced_t = _post_spacing_transform(config)
    model_t = get_base_transforms(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )

    rows = []
    for idx, case in enumerate(cases, start=1):
        split = split_lookup.get(case["id"], "unknown")
        print(f"[{idx}/{len(cases)}] auditing {case['id']} ({split})")
        rows.append(_row_for_case(case, split, native_t, spaced_t, model_t))

    split_order = {"train": 0, "val": 1, "test": 2, "unknown": 3}
    rows.sort(key=lambda r: (split_order.get(r["split"], 9), r["case_id"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _print_rows(rows, focus_case=args.focus_case)
    _focus_report(rows, args.focus_case)

    counts = {name: sum(1 for r in rows if r["split"] == name) for name in ("train", "val", "test", "unknown")}
    print(f"\nCases: {len(rows)} | train={counts['train']} val={counts['val']} test={counts['test']} unknown={counts['unknown']}")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
