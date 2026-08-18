#!/usr/bin/env python3
"""Build robust GT-containing ROI crops for offline refinement training.

Unlike the oracle dataset, which always places the target in a nearly symmetric GT-derived
crop, this dataset deliberately changes target position and surrounding context while always
retaining the complete HUMAN_GOLD mask.

For every audited source case:
  * variant 0 uses the original symmetric oracle margin (default 0.40 each side),
  * remaining variants draw independent low/high margins for X/Y/Z,
  * each side margin is sampled as a fraction of that axis' GT bounding-box extent,
  * all crops are clipped to the native image but are guaranteed to contain the full GT.

Independent side margins naturally shift the target away from the crop centre and vary crop
scale/context. This is experiment-only training data for the OFFLINE active-learning refiner;
it does not change the single-stage production model.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("build_jittered_roi_dataset.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import discover_round1_cases
from scripts.oracle_roi_utils import geometry_equal

DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/jittered_roi_dataset_62_v1")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bbox_xyz(mask_zyx):
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise RuntimeError(f"Expected non-empty 3D binary mask, got {mask.shape}")
    zz, yy, xx = np.where(mask)
    lo = np.asarray([xx.min(), yy.min(), zz.min()], dtype=int)
    hi = np.asarray([xx.max(), yy.max(), zz.max()], dtype=int)
    extent = hi - lo + 1
    return lo, hi, extent


def deterministic_case_rng(global_seed, case_id):
    digest = hashlib.sha256(f"{int(global_seed)}:{case_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def make_variant(image, label, case_id, variant, seed, nominal_margin, min_margin, max_margin):
    if not geometry_equal(image, label):
        raise RuntimeError(f"Image/label geometry mismatch for {case_id}")

    label_arr = np.asarray(sitk.GetArrayFromImage(label))
    binary = label_arr > 0
    lo, hi, extent = bbox_xyz(binary)
    full = np.asarray(image.GetSize(), dtype=int)

    if variant == 0:
        low_frac = np.full(3, float(nominal_margin), dtype=float)
        high_frac = np.full(3, float(nominal_margin), dtype=float)
        variant_type = "NOMINAL_SYMMETRIC"
    else:
        rng = deterministic_case_rng(seed + 1009 * variant, case_id)
        low_frac = rng.uniform(float(min_margin), float(max_margin), size=3)
        high_frac = rng.uniform(float(min_margin), float(max_margin), size=3)
        variant_type = "JITTERED_ASYMMETRIC"

    low_margin = np.ceil(extent.astype(float) * low_frac).astype(int)
    high_margin = np.ceil(extent.astype(float) * high_frac).astype(int)
    start = np.maximum(0, lo - low_margin)
    stop = np.minimum(full, hi + 1 + high_margin)
    size = stop - start
    if np.any(size <= 0):
        raise RuntimeError(f"Invalid jittered ROI for {case_id}: start={start}, size={size}")

    # Because start <= GT min and stop > GT max by construction, clipping at the full image
    # boundary cannot remove GT voxels. Verify this explicitly for every generated crop.
    crop_image = sitk.RegionOfInterest(
        image, size=tuple(int(x) for x in size), index=tuple(int(x) for x in start)
    )
    crop_label = sitk.RegionOfInterest(
        label, size=tuple(int(x) for x in size), index=tuple(int(x) for x in start)
    )
    crop_binary = np.asarray(sitk.GetArrayFromImage(crop_label)) > 0
    if int(crop_binary.sum()) != int(binary.sum()):
        raise RuntimeError(
            f"Jittered crop lost GT for {case_id}: full={int(binary.sum())}, crop={int(crop_binary.sum())}"
        )

    gt_center = 0.5 * (lo.astype(float) + hi.astype(float))
    crop_center = start.astype(float) + 0.5 * np.maximum(size - 1, 0)
    center_offset_frac = (gt_center - crop_center) / np.maximum(extent.astype(float), 1.0)
    crop_vox = int(np.prod(size.astype(np.int64)))
    full_vox = int(np.prod(full.astype(np.int64)))
    gt_vox = int(binary.sum())

    meta = {
        "variant_type": variant_type,
        "roi_start_x": int(start[0]),
        "roi_start_y": int(start[1]),
        "roi_start_z": int(start[2]),
        "roi_size_x": int(size[0]),
        "roi_size_y": int(size[1]),
        "roi_size_z": int(size[2]),
        "gt_bbox_x": int(extent[0]),
        "gt_bbox_y": int(extent[1]),
        "gt_bbox_z": int(extent[2]),
        "low_margin_frac_x": float(low_frac[0]),
        "low_margin_frac_y": float(low_frac[1]),
        "low_margin_frac_z": float(low_frac[2]),
        "high_margin_frac_x": float(high_frac[0]),
        "high_margin_frac_y": float(high_frac[1]),
        "high_margin_frac_z": float(high_frac[2]),
        "gt_center_offset_bbox_frac_x": float(center_offset_frac[0]),
        "gt_center_offset_bbox_frac_y": float(center_offset_frac[1]),
        "gt_center_offset_bbox_frac_z": float(center_offset_frac[2]),
        "roi_gt_fraction": float(gt_vox / max(crop_vox, 1)),
        "roi_volume_fraction_of_full": float(crop_vox / max(full_vox, 1)),
    }
    return crop_image, crop_label, meta


def main():
    p = argparse.ArgumentParser(description="Build jittered GT-containing ROI training dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--variants-per-case", type=int, default=4)
    p.add_argument("--nominal-margin", type=float, default=0.40)
    p.add_argument("--min-margin", type=float, default=0.40)
    p.add_argument("--max-margin", type=float, default=1.00)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.variants_per_case < 2:
        p.error("--variants-per-case must be >=2 so the dataset contains actual crop jitter")
    if args.nominal_margin < 0 or args.min_margin < 0 or args.max_margin < args.min_margin:
        p.error("Require nominal-margin>=0 and 0<=min-margin<=max-margin")

    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    expected_n = int(audit.get("n_current_valid_human_labels", len(audited_ids)))
    if not audited_ids or len(audited_ids) != expected_n:
        raise RuntimeError("Round-2 audit ID/count mismatch")

    config = HASSLConfig.from_yaml(args.config)
    _, _, by_id, _ = discover_round1_cases(config, Path(args.source_manifest))
    if sorted(str(x) for x in by_id) != audited_ids:
        raise RuntimeError("Current labeled dataset no longer matches the passing Round-2 audit")

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_index, case_id in enumerate(audited_ids, start=1):
        case = by_id[case_id]
        image = sitk.ReadImage(str(case["image"]))
        label = sitk.ReadImage(str(case["label"]))
        for variant in range(int(args.variants_per_case)):
            crop_image, crop_label, meta = make_variant(
                image=image,
                label=label,
                case_id=case_id,
                variant=variant,
                seed=int(args.seed),
                nominal_margin=float(args.nominal_margin),
                min_margin=float(args.min_margin),
                max_margin=float(args.max_margin),
            )
            generated_id = f"{case_id}__roi{variant:02d}"
            image_out = output_dir / f"{generated_id}{config.image_suffix}"
            label_out = labels_dir / f"{generated_id}{config.label_suffix}"
            sitk.WriteImage(crop_image, str(image_out), useCompression=True)
            sitk.WriteImage(crop_label, str(label_out), useCompression=True)
            rows.append({
                "generated_id": generated_id,
                "source_case_id": case_id,
                "variant": variant,
                "source_image": str(case["image"]),
                "source_label": str(case["label"]),
                "roi_image": str(image_out),
                "roi_label": str(label_out),
                **meta,
            })
        print(f"[{source_index:2d}/{len(audited_ids)}] {case_id} | variants={args.variants_per_case}")

    manifest_path = output_dir / "jittered_roi_manifest.csv"
    write_csv(manifest_path, rows)
    offset_abs = np.asarray([
        [abs(float(r[f"gt_center_offset_bbox_frac_{a}"])) for a in "xyz"] for r in rows
    ])
    crop_fracs = np.asarray([float(r["roi_volume_fraction_of_full"]) for r in rows], dtype=float)
    payload = {
        "version": "jittered_roi_dataset_62_v1",
        "purpose": "robust offline ROI refinement training with GT-retaining crop perturbations",
        "source_round2_audit": str(audit_path),
        "n_source_cases": len(audited_ids),
        "source_case_ids": audited_ids,
        "variants_per_case": int(args.variants_per_case),
        "n_generated_samples": len(rows),
        "generated_ids": [str(r["generated_id"]) for r in rows],
        "nominal_margin_each_side": float(args.nominal_margin),
        "jitter_margin_range_each_side": [float(args.min_margin), float(args.max_margin)],
        "seed": int(args.seed),
        "median_abs_target_center_offset_bbox_fraction_xyz": np.median(offset_abs, axis=0).tolist(),
        "max_abs_target_center_offset_bbox_fraction_xyz": np.max(offset_abs, axis=0).tolist(),
        "median_roi_volume_fraction_of_full": float(np.median(crop_fracs)),
        "warning": (
            "Ground truth is used to guarantee that training crops contain the target. This dataset is for the "
            "offline ROI refiner only and does not define production inference behaviour."
        ),
    }
    metadata_path = output_dir / "jittered_roi_dataset_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("JITTERED ROI DATASET COMPLETE")
    print(f"Source HUMAN_GOLD cases:      {len(audited_ids)}")
    print(f"Variants per case:            {args.variants_per_case}")
    print(f"Generated ROI samples:        {len(rows)}")
    print(f"Margin range each side:       {args.min_margin:.2f} .. {args.max_margin:.2f} x GT bbox")
    print(f"Median crop/full volume:      {np.median(crop_fracs):.3f}")
    print(
        "Median |target offset|/bbox: "
        + ", ".join(f"{x:.3f}" for x in np.median(offset_abs, axis=0))
    )
    print(f"Dataset:                      {output_dir}")
    print(f"Manifest:                     {manifest_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
