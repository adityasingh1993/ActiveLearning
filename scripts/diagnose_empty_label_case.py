#!/usr/bin/env python3
"""Diagnose a labeled case that becomes empty after deterministic preprocessing.

This is a read-only data diagnostic. It reports label voxel counts and bounding-box
extent at native load, after orientation/spacing, and after final model-space resize.
It also tests several candidate resize sizes so we can tell whether a tiny target is
being sampled away by the 128^3 nearest-neighbor label resize.

Do not use this script to silently choose a higher resolution from held-out model
performance; it is only a preprocessing/data-integrity check.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
import scripts.audit_labeled_dataset as audit
import scripts.train_supervised_cv as cv


DEFAULT_CASE = "967fd49a334a6809669c28ddc13f5151c733cdabc898646518bc9d734628ee1f"


def mask3d(label):
    mask = torch.as_tensor(label) > 0.5
    while mask.ndim > 3:
        mask = mask.any(dim=0)
    return mask


def bbox_stats(mask):
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return {
            "vox": 0,
            "bbox": None,
            "bbox_size": None,
            "centroid": None,
        }
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    size = maxs - mins + 1
    centroid = coords.float().mean(dim=0)
    return {
        "vox": int(mask.sum().item()),
        "bbox": tuple((int(lo), int(hi)) for lo, hi in zip(mins.tolist(), maxs.tolist())),
        "bbox_size": tuple(int(v) for v in size.tolist()),
        "centroid": tuple(float(v) for v in centroid.tolist()),
    }


def shape3d(value):
    shape = tuple(int(v) for v in value.shape)
    return shape[-3:]


def spacing_from(value):
    try:
        affine = value.affine
        if torch.is_tensor(affine) and affine.ndim == 3:
            affine = affine[0]
        return tuple(float(torch.linalg.norm(affine[:3, i]).item()) for i in range(3))
    except Exception:
        return (float("nan"),) * 3


def print_stage(name, sample):
    label = sample["label"]
    image = sample["image"]
    stats = bbox_stats(mask3d(label))
    print(f"\n{name}")
    print("-" * 92)
    print(f"image shape: {shape3d(image)}")
    print(f"label shape: {shape3d(label)}")
    print(f"spacing:     {spacing_from(image)}")
    print(f"GT voxels:   {stats['vox']}")
    print(f"GT bbox:     {stats['bbox']}")
    print(f"GT bbox size:{stats['bbox_size']}")
    print(f"GT centroid: {stats['centroid']}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Trace where a labeled mask becomes empty during deterministic preprocessing"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument(
        "--resize-sizes",
        nargs="+",
        type=int,
        default=[128, 160, 192, 256],
        help="Diagnostic model-space sizes to test after the same spacing/orientation pipeline.",
    )
    args = parser.parse_args()

    config = HASSLConfig.from_yaml(args.config)
    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}
    if args.case_id not in by_id:
        raise RuntimeError(
            f"Case {args.case_id} not found among {len(cases)} labeled image/label pairs."
        )
    case = by_id[args.case_id]

    print("=" * 100)
    print("EMPTY-LABEL PREPROCESSING DIAGNOSTIC")
    print(f"Case:  {args.case_id}")
    print(f"Image: {case['image']}")
    print(f"Label: {case['label']}")
    print(f"Config spacing: {tuple(config.spacing) if config.spacing else None}")
    print(f"Config preprocessing_mode: {getattr(config, 'preprocessing_mode', 'resize')}")
    print(f"Config spatial_size: {tuple(config.spatial_size)}")
    print("=" * 100)

    native = audit._native_transform()(dict(case))
    spaced = audit._post_spacing_transform(config)(dict(case))

    native_stats = print_stage("1) NATIVE LOAD", native)
    spaced_stats = print_stage("2) AFTER ORIENTATION + SPACING + LABEL BINARIZATION", spaced)

    resize_rows = []
    for size in args.resize_sizes:
        test_config = copy.deepcopy(config)
        test_config.preprocessing_mode = "resize"
        test_config.spatial_size = (int(size), int(size), int(size))
        transform = get_base_transforms(
            test_config,
            keys=["image", "label"],
            is_training=False,
            apply_strong_aug=False,
        )
        sample = transform(dict(case))
        stats = print_stage(f"3) FINAL MODEL SPACE @ {size}^3", sample)
        resize_rows.append((int(size), stats["vox"], stats["bbox_size"]))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"Native GT voxels:       {native_stats['vox']}")
    print(f"Post-spacing GT voxels: {spaced_stats['vox']}")
    for size, vox, bbox_size in resize_rows:
        status = "EMPTY" if vox == 0 else "OK"
        print(f"Resize {size:>3}^3: GT voxels={vox:<8} bbox={bbox_size}  [{status}]")

    if native_stats["vox"] == 0:
        print("\nDIAGNOSIS: the source annotation itself is empty or not being decoded as foreground.")
    elif spaced_stats["vox"] == 0:
        print("\nDIAGNOSIS: foreground is lost during orientation/spacing/binarization; inspect label geometry/metadata.")
    elif resize_rows and resize_rows[0][1] == 0:
        surviving = [size for size, vox, _ in resize_rows if vox > 0]
        print("\nDIAGNOSIS: foreground survives spacing but is sampled away by final fixed-grid resizing.")
        if surviving:
            print(f"First tested size with surviving foreground: {surviving[0]}^3")
        else:
            print("Foreground remained empty at every tested model-space size; inspect label/image geometry alignment.")
    else:
        print("\nDIAGNOSIS: foreground is non-empty at the configured model size; rerun the 48-case audit after git pull.")


if __name__ == "__main__":
    main()
