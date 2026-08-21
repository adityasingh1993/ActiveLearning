#!/usr/bin/env python3
"""Quantify bladder size/occupancy and relate it to frozen OOF segmentation performance.

This is a diagnostic only; target-derived size information is never used for model inference.

Outputs
-------
- all72_bladder_size_profile.csv
- oof_case_metrics_with_size.csv
- oof_size_group_summary.csv
- bladder_size_summary.json

Size groups are terciles of native physical bladder volume computed across the audited Final72
HUMAN_GOLD set. The same fixed thresholds are then used to summarize the supplied OOF metrics
(default: Final72 A0 / original47). This makes later A3/resolution comparisons use identical
small/medium/large definitions.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import nrrd
import SimpleITK as sitk
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_final72_screen_spatial_folds12 as spatial_screen

DEFAULT_METRICS = Path("experiments/round3_cv_72_translation12/cv_results.csv")
DEFAULT_OUTPUT = Path("experiments/final72_bladder_size_diagnostic")
SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def seg_voxel_volume_mm3(header, fallback_spacing):
    directions = header.get("space directions")
    vectors = []
    if directions is not None:
        for item in directions:
            try:
                vec = np.asarray(item, dtype=float).reshape(-1)
            except Exception:
                continue
            if vec.size == 3 and np.all(np.isfinite(vec)):
                vectors.append(vec)
    if len(vectors) >= 3:
        value = abs(float(np.linalg.det(np.stack(vectors[:3], axis=0))))
        if math.isfinite(value) and value > 0:
            return value
    return float(np.prod(np.asarray(fallback_spacing, dtype=float)))


def size_group(volume, q1, q2):
    if volume <= q1:
        return "SMALL"
    if volume <= q2:
        return "MEDIUM"
    return "LARGE"


def summarize_group(rows, group):
    subset = [r for r in rows if r["size_group"] == group]
    if not subset:
        return {"size_group": group, "n": 0}

    def arr(key):
        return np.asarray([float(r[key]) for r in subset], dtype=float)

    return {
        "size_group": group,
        "n": len(subset),
        "mean_native_bladder_ml": float(np.mean(arr("native_bladder_ml"))),
        "median_post128_fg_fraction_pct": float(np.median(arr("post128_fg_fraction_pct"))),
        "mean_dice": float(np.mean(arr("dice"))),
        "median_dice": float(np.median(arr("dice"))),
        "mean_precision": float(np.mean(arr("precision"))),
        "mean_recall": float(np.mean(arr("recall"))),
        "mean_hd95_mm": float(np.nanmean(arr("hd95"))),
        "median_abs_rve_pct": float(np.median(np.abs(arr("signed_rve_pct")))),
        "dice_lt_0p70": int(np.sum(arr("dice") < 0.70)),
        "dice_lt_0p50": int(np.sum(arr("dice") < 0.50)),
        "dice_ge_0p80": int(np.sum(arr("dice") >= 0.80)),
    }


def main():
    p = argparse.ArgumentParser(description="Final72 bladder size/occupancy diagnostic")
    p.add_argument("--config", required=True)
    p.add_argument("--metrics-csv", default=str(DEFAULT_METRICS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if int(config.num_classes) != 1:
        raise RuntimeError("This diagnostic is locked to binary bladder segmentation")

    source_manifest_path = Path(args.source_cv_dir) / "cv_splits.json"
    source_manifest, _, _ = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    source_ids = set(str(x) for x in source_manifest["all_case_ids"])

    cases = cv.collect_cases(config)
    by_id = {str(x["id"]): x for x in cases}
    if len(by_id) != 72:
        raise RuntimeError(f"Expected exact audited Final72 discovery, found {len(by_id)} cases")

    # Deterministic 128^3 transform: same preprocessing definition used by the frozen models.
    cv.apply_baseline(config, resize_size=128, epochs=100)
    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )

    profiles = []
    for case_id in sorted(by_id):
        case = by_id[case_id]
        image_path = str(case["image"])
        label_path = str(case["label"])

        image = sitk.ReadImage(image_path)
        if image.GetDimension() != 3:
            raise RuntimeError(f"{case_id}: expected 3D image, got {image.GetDimension()}D")
        native_size_xyz = tuple(int(x) for x in image.GetSize())
        native_spacing_xyz = tuple(float(x) for x in image.GetSpacing())
        native_fov_mm3 = float(np.prod(np.asarray(native_size_xyz) * np.asarray(native_spacing_xyz)))

        seg, header = nrrd.read(label_path)
        seg = np.squeeze(seg)
        if seg.ndim != 3:
            raise RuntimeError(f"{case_id}: label is not 3D after squeeze: {seg.shape}")
        native_fg_vox = int(np.count_nonzero(seg > 0))
        if native_fg_vox <= 0:
            raise RuntimeError(f"{case_id}: empty HUMAN_GOLD label")
        voxel_mm3 = seg_voxel_volume_mm3(header, native_spacing_xyz)
        bladder_mm3 = native_fg_vox * voxel_mm3
        bladder_ml = bladder_mm3 / 1000.0
        native_occupancy_pct = 100.0 * bladder_mm3 / max(native_fov_mm3, 1e-12)

        sample = transform({"image": image_path, "label": label_path, "id": case_id})
        label128 = np.asarray(sample["label"].detach().cpu())
        post_fg = int(np.count_nonzero(label128 > 0.5))
        post_total = int(label128.size)
        post_fraction_pct = 100.0 * post_fg / max(post_total, 1)

        profiles.append({
            "case_id": case_id,
            "is_original47": int(case_id in source_ids),
            "native_size_x": native_size_xyz[0],
            "native_size_y": native_size_xyz[1],
            "native_size_z": native_size_xyz[2],
            "native_spacing_x_mm": native_spacing_xyz[0],
            "native_spacing_y_mm": native_spacing_xyz[1],
            "native_spacing_z_mm": native_spacing_xyz[2],
            "native_fov_ml": native_fov_mm3 / 1000.0,
            "native_fg_voxels": native_fg_vox,
            "native_bladder_ml": bladder_ml,
            "native_bladder_fov_fraction_pct": native_occupancy_pct,
            "post128_fg_voxels": post_fg,
            "post128_fg_fraction_pct": post_fraction_pct,
        })

    volumes = np.asarray([float(r["native_bladder_ml"]) for r in profiles], dtype=float)
    q1, q2 = [float(x) for x in np.quantile(volumes, [1.0 / 3.0, 2.0 / 3.0])]
    for row in profiles:
        row["size_group"] = size_group(float(row["native_bladder_ml"]), q1, q2)
    write_csv(output_dir / "all72_bladder_size_profile.csv", profiles)

    metrics = read_csv(Path(args.metrics_csv))
    metric_by_id = {str(r["case_id"]): r for r in metrics}
    if set(metric_by_id) != source_ids:
        raise RuntimeError(
            "Metrics CSV must contain the exact frozen original47 held-out cases. "
            f"Found={len(metric_by_id)} expected={len(source_ids)}"
        )

    profile_by_id = {r["case_id"]: r for r in profiles}
    merged = []
    for case_id in sorted(source_ids):
        p_row = profile_by_id[case_id]
        m_row = metric_by_id[case_id]
        gt_vox = float(m_row.get("gt_vox", 0) or 0)
        pred_vox = float(m_row.get("pred_vox", 0) or 0)
        signed_rve = 100.0 * (pred_vox - gt_vox) / max(gt_vox, 1e-8)
        merged.append({
            **p_row,
            "fold": int(m_row["fold"]),
            "dice": float(m_row["dice"]),
            "precision": float(m_row["precision"]),
            "recall": float(m_row["recall"]),
            "hd95": float(m_row["hd95"]),
            "signed_rve_pct": signed_rve,
            "prediction_source": m_row.get("source", ""),
            "threshold": m_row.get("threshold", ""),
        })
    write_csv(output_dir / "oof_case_metrics_with_size.csv", merged)

    group_rows = [summarize_group(merged, g) for g in ("SMALL", "MEDIUM", "LARGE")]
    write_csv(output_dir / "oof_size_group_summary.csv", group_rows)

    dice = np.asarray([float(r["dice"]) for r in merged], dtype=float)
    log_volume = np.log10(np.asarray([float(r["native_bladder_ml"]) for r in merged], dtype=float) + 1e-12)
    occupancy = np.asarray([float(r["post128_fg_fraction_pct"]) for r in merged], dtype=float)
    rho_volume, p_volume = spearmanr(log_volume, dice)
    rho_occ, p_occ = spearmanr(occupancy, dice)

    summary = {
        "version": "final72_bladder_size_diagnostic_v1",
        "n_human_gold": len(profiles),
        "n_oof_cases": len(merged),
        "size_group_basis": "terciles of native physical bladder volume across audited Final72",
        "small_max_ml": q1,
        "medium_max_ml": q2,
        "native_bladder_ml": {
            "min": float(np.min(volumes)),
            "median": float(np.median(volumes)),
            "max": float(np.max(volumes)),
        },
        "spearman_log_native_volume_vs_dice": {"rho": float(rho_volume), "p": float(p_volume)},
        "spearman_post128_occupancy_vs_dice": {"rho": float(rho_occ), "p": float(p_occ)},
        "groups": group_rows,
        "warning": "Target-derived size groups are diagnostic only and must not be used at deployment without an independent size estimator.",
    }
    (output_dir / "bladder_size_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 112)
    print("FINAL72 BLADDER SIZE / OCCUPANCY DIAGNOSTIC")
    print(f"Physical-volume terciles: SMALL <= {q1:.4f} mL | MEDIUM <= {q2:.4f} mL | LARGE > {q2:.4f} mL")
    for row in group_rows:
        print(
            f"{row['size_group']:>6}: n={row['n']:2d} | Dice={row.get('mean_dice', float('nan')):.4f} | "
            f"Prec={row.get('mean_precision', float('nan')):.4f} | Rec={row.get('mean_recall', float('nan')):.4f} | "
            f"post128 FG={row.get('median_post128_fg_fraction_pct', float('nan')):.4f}% | "
            f"Dice<.70={row.get('dice_lt_0p70', 0)}"
        )
    print(f"Spearman log(volume) vs Dice: rho={rho_volume:+.3f}, p={p_volume:.4g}")
    print(f"Spearman post128 occupancy vs Dice: rho={rho_occ:+.3f}, p={p_occ:.4g}")
    print(f"Outputs: {output_dir}")
    print("=" * 112)


if __name__ == "__main__":
    main()
