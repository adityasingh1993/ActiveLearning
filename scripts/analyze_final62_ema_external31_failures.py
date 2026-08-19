#!/usr/bin/env python3
"""Analyze failure modes of the trusted Final62 128^3 EMA model on frozen external31.

This is an analysis-only tool. External GT is used only to understand failure anatomy after the
Final62 128^3 EMA operating point was already frozen. It does not tune threshold, checkpoint,
training, or QC policy.

For every external case the script reruns EMA inference with the same strict native-grid MONAI
inversion used by the established benchmark and derives:
  - Dice / precision / recall / signed RVE / FP / FN
  - predicted and GT connected-component statistics
  - physical centroid displacement relative to GT extent
  - fraction of voxel errors that lie close to the opposing surface
  - QC bucket / original QC predictions from the pool manifest

Failures (Dice < 0.70 by default) are assigned a deterministic descriptive class:
  LOCALIZATION_FAILURE, FRAGMENTED_FP, UNDER_SEGMENTATION,
  OVER_SEGMENTATION, BOUNDARY_DOMINANT, or MIXED_ERROR.

The classes are descriptive diagnostics, not learned clinical labels and not deployment rules.

Outputs
-------
  external31_ema_failure_analysis_all.csv
  external31_ema_failure_analysis_failures.csv
  external31_ema_failure_analysis_summary.json
  external31_ema_failure_analysis_report.md
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    from scipy import ndimage as ndi
except ImportError as exc:
    raise ImportError("This analysis requires scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

DEFAULT_CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_POOL_MANIFEST = Path(
    "experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv"
)
DEFAULT_ROUND2_AUDIT = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_final62_ema_failure_analysis")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
            writer.writerow({key: row.get(key, "") for key in fields})


def safe_float(value):
    try:
        text = str(value).strip()
        return float(text) if text else float("nan")
    except Exception:
        return float("nan")


def connected_component_stats(mask):
    mask = np.asarray(mask, dtype=bool)
    vox = int(mask.sum())
    if vox == 0:
        return 0, 0, 0.0
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labeled.ravel())[1:]
    largest = int(counts.max()) if counts.size else 0
    return int(n), largest, float(largest / max(vox, 1))


def centroid_zyx(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def bbox_diag_mm(mask, spacing_zyx):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return float("nan")
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    return float(np.linalg.norm(extent.astype(float) * spacing_zyx))


def centroid_distance_mm(pred, gt, spacing_zyx):
    pc = centroid_zyx(pred)
    gc = centroid_zyx(gt)
    if pc is None or gc is None:
        return float("nan")
    return float(np.linalg.norm((pc - gc) * spacing_zyx))


def boundary_error_stats(pred, gt, spacing_zyx, boundary_mm):
    """Return error fractions close to the opposing mask in physical mm."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    fp = pred & ~gt
    fn = ~pred & gt
    n_fp = int(fp.sum())
    n_fn = int(fn.sum())
    n_err = n_fp + n_fn
    if n_err == 0:
        return 1.0, 1.0, 1.0

    if gt.any():
        dist_to_gt = ndi.distance_transform_edt(~gt, sampling=spacing_zyx)
        fp_near = int((fp & (dist_to_gt <= boundary_mm)).sum())
    else:
        fp_near = 0

    if pred.any():
        dist_to_pred = ndi.distance_transform_edt(~pred, sampling=spacing_zyx)
        fn_near = int((fn & (dist_to_pred <= boundary_mm)).sum())
    else:
        fn_near = 0

    fp_frac = float(fp_near / n_fp) if n_fp else 1.0
    fn_frac = float(fn_near / n_fn) if n_fn else 1.0
    overall = float((fp_near + fn_near) / n_err)
    return fp_frac, fn_frac, overall


def classify_failure(row):
    """Deterministic descriptive failure taxonomy; GT-derived and analysis-only."""
    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    signed_rve = float(row["signed_rve_pct"])
    fp = int(row["fp_vox"])
    fn = int(row["fn_vox"])
    pred_vox = int(row["pred_vox"])
    gt_vox = int(row["gt_vox"])
    pred_cc = int(row["pred_components"])
    lcf = float(row["pred_largest_component_fraction"])
    centroid_norm = float(row["centroid_distance_over_gt_diag"])
    boundary_frac = float(row["boundary_error_fraction"])

    if pred_vox == 0 and gt_vox > 0:
        return "UNDER_SEGMENTATION"

    overlap_coeff = float(row["overlap_coefficient"])
    if (
        dice < 0.20
        or overlap_coeff < 0.25
        or (
            math.isfinite(centroid_norm)
            and centroid_norm > 0.75
            and precision < 0.40
            and recall < 0.40
        )
    ):
        return "LOCALIZATION_FAILURE"

    if pred_cc >= 3 and lcf < 0.80 and fp > fn:
        return "FRAGMENTED_FP"

    if signed_rve <= -20.0 or recall + 0.15 < precision or fn > 1.5 * max(fp, 1):
        return "UNDER_SEGMENTATION"

    if signed_rve >= 20.0 or precision + 0.15 < recall or fp > 1.5 * max(fn, 1):
        return "OVER_SEGMENTATION"

    if boundary_frac >= 0.70:
        return "BOUNDARY_DOMINANT"

    return "MIXED_ERROR"


def markdown_report(rows, failures, summary):
    lines = [
        "# Final62 128³ EMA — external31 failure analysis",
        "",
        "This report is **analysis-only**. External GT is used to describe already-observed failures; "
        "none of these GT-derived labels should be used as deployment features.",
        "",
        "## Overall",
        "",
        f"- Cases: {summary['n_cases']}",
        f"- Mean Dice: {summary['mean_dice']:.4f}",
        f"- Failures Dice < {summary['failure_dice_threshold']:.2f}: {summary['n_failures']}",
        f"- High quality Dice >= 0.80: {summary['n_high_quality']}",
        f"- GT-volume quartiles: Q1={summary['gt_volume_q1_mm3']:.3f} mm³, Q3={summary['gt_volume_q3_mm3']:.3f} mm³",
        "",
        "## Failure classes",
        "",
    ]
    for name, count in sorted(summary["failure_class_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {name}: {count}")

    lines.extend([
        "",
        "## Failed cases",
        "",
        "| case | Dice | class | size | precision | recall | RVE % | pred CC | largest CC frac | centroid/GT diag | boundary-error frac | QC bucket | QC predicted Dice | QC P(fail) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for r in failures:
        lines.append(
            "| {case_id} | {dice:.4f} | {failure_class} | {gt_size_group} | {precision:.3f} | "
            "{recall:.3f} | {signed_rve_pct:+.1f} | {pred_components} | "
            "{pred_largest_component_fraction:.3f} | {centroid_distance_over_gt_diag:.3f} | "
            "{boundary_error_fraction:.3f} | {qc_bucket} | {qc_predicted_dice:.3f} | "
            "{qc_failure_probability:.3f} |".format(**r)
        )

    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- `LOCALIZATION_FAILURE`: very weak overlap and/or large center displacement.",
        "- `FRAGMENTED_FP`: multiple predicted components with substantial foreground outside the dominant component.",
        "- `UNDER_SEGMENTATION`: FN/recall/negative-volume-error dominant.",
        "- `OVER_SEGMENTATION`: FP/precision/positive-volume-error dominant.",
        "- `BOUNDARY_DOMINANT`: most FP/FN voxels lie within the configured physical boundary band.",
        "- `MIXED_ERROR`: no single heuristic dominates.",
        "",
        "These are descriptive heuristics for deciding what correction mechanism to investigate next, not ground-truth clinical categories.",
    ])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Analyze trusted Final62 128^3 EMA failures on frozen external31")
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External GT root; repeat if needed")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    p.add_argument("--boundary-mm", type=float, default=0.50)
    p.add_argument("--gpu", choices=["0", "1"], default=None)
    args = p.parse_args()

    if args.gpu is not None:
        # CUDA_VISIBLE_DEVICES must normally be set before torch import. Keep this argument only
        # for provenance/display; prefer prefixing the command with CUDA_VISIBLE_DEVICES.
        print(
            "NOTE: --gpu is informational here because torch is already imported. "
            "Prefer CUDA_VISIBLE_DEVICES=<id> before python when selecting a physical GPU."
        )

    if not 0.0 < args.failure_dice < args.high_quality_dice <= 1.0:
        p.error("Require 0 < failure-dice < high-quality-dice <= 1")
    if args.boundary_mm <= 0:
        p.error("--boundary-mm must be > 0")

    checkpoint = Path(args.checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, pool_manifest, audit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Final62 EMA analysis requires prototype student + EMA checkpoint")
    cv.apply_baseline(config, resize_size=128, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata is not marked passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError(
            "EXTERNAL/TRAINING LEAKAGE: external31 overlaps Final62 human training IDs: "
            + ", ".join(overlap)
        )

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=False, to_tensor=True,
    )
    loader = DataLoader(
        Dataset([{"image": c["image"], "id": c["id"]} for c in cases], transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Final62 checkpoint has no EMA teacher")
    del student
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 124)
    print("FINAL62 128^3 EMA — FROZEN EXTERNAL31 FAILURE ANALYSIS")
    print(f"Checkpoint:       {checkpoint}")
    print(f"Cases:            {len(cases)}")
    print(f"Training overlap: {len(overlap)}")
    print(f"Failure cutoff:   Dice < {args.failure_dice:.2f}")
    print(f"Boundary band:    {args.boundary_mm:.2f} mm")
    print(f"Device:           {device}")
    print("External GT is analysis-only; no model/QC threshold fitting is performed.")
    print("=" * 124)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, teacher))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
            source_image, prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)
            pred = prob_zyx > 0.50
            metrics = binary_metrics(pred, gt)

            spacing_zyx = np.asarray(tuple(reversed(source_image.GetSpacing())), dtype=float)
            voxel_volume_mm3 = float(np.prod(np.asarray(source_image.GetSpacing(), dtype=float)))
            pred_cc, pred_largest, pred_lcf = connected_component_stats(pred)
            gt_cc, gt_largest, gt_lcf = connected_component_stats(gt)
            cdist = centroid_distance_mm(pred, gt, spacing_zyx)
            gt_diag = bbox_diag_mm(gt, spacing_zyx)
            cdist_norm = float(cdist / gt_diag) if math.isfinite(cdist) and gt_diag > 0 else float("nan")
            fp_near, fn_near, boundary_err = boundary_error_stats(
                pred, gt, spacing_zyx, float(args.boundary_mm)
            )
            tp = int(metrics["tp_vox"])
            pred_vox = int(metrics["pred_vox"])
            gt_vox = int(metrics["gt_vox"])
            overlap_coeff = float(tp / max(min(pred_vox, gt_vox), 1))
            error_vox = int(metrics["fp_vox"]) + int(metrics["fn_vox"])
            fp_error_fraction = float(int(metrics["fp_vox"]) / max(error_vox, 1))

            row = {
                "case_id": case_id,
                "dice": float(metrics["dice"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "signed_rve_pct": float(metrics["signed_rve_pct"]),
                "abs_rve_pct": float(metrics["abs_rve_pct"]),
                "tp_vox": tp,
                "fp_vox": int(metrics["fp_vox"]),
                "fn_vox": int(metrics["fn_vox"]),
                "pred_vox": pred_vox,
                "gt_vox": gt_vox,
                "gt_volume_mm3": float(gt_vox * voxel_volume_mm3),
                "pred_volume_mm3": float(pred_vox * voxel_volume_mm3),
                "error_vox": error_vox,
                "fp_error_fraction": fp_error_fraction,
                "overlap_coefficient": overlap_coeff,
                "pred_components": pred_cc,
                "pred_largest_component_vox": pred_largest,
                "pred_largest_component_fraction": pred_lcf,
                "gt_components": gt_cc,
                "gt_largest_component_vox": gt_largest,
                "gt_largest_component_fraction": gt_lcf,
                "centroid_distance_mm": cdist,
                "gt_bbox_diag_mm": gt_diag,
                "centroid_distance_over_gt_diag": cdist_norm,
                "fp_near_boundary_fraction": fp_near,
                "fn_near_boundary_fraction": fn_near,
                "boundary_error_fraction": boundary_err,
                "boundary_band_mm": float(args.boundary_mm),
                "qc_bucket": str(case.get("qc_bucket", "")),
                "qc_predicted_dice": safe_float(case.get("original_predicted_dice", "")),
                "qc_failure_probability": safe_float(case.get("original_failure_probability", "")),
                "is_failure": int(float(metrics["dice"]) < float(args.failure_dice)),
                "is_high_quality": int(float(metrics["dice"]) >= float(args.high_quality_dice)),
            }
            rows.append(row)
            print(
                f"[{index:2d}/{len(cases)}] {case_id[:12]} | Dice={row['dice']:.4f} "
                f"P/R={row['precision']:.3f}/{row['recall']:.3f} | RVE={row['signed_rve_pct']:+.1f}% "
                f"| CC={pred_cc} LCF={pred_lcf:.2f} | centroid/diag={cdist_norm:.2f}"
            )

    gt_vols = np.asarray([float(r["gt_volume_mm3"]) for r in rows], dtype=float)
    q1 = float(np.quantile(gt_vols, 0.25))
    q3 = float(np.quantile(gt_vols, 0.75))
    for row in rows:
        v = float(row["gt_volume_mm3"])
        row["gt_size_group"] = "SMALL" if v <= q1 else ("LARGE" if v >= q3 else "MID")
        row["failure_class"] = classify_failure(row) if row["is_failure"] else "NON_FAILURE"

    rows.sort(key=lambda r: (float(r["dice"]), str(r["case_id"])))
    failures = [r for r in rows if r["is_failure"]]
    class_counts = Counter(r["failure_class"] for r in failures)
    size_counts = Counter(r["gt_size_group"] for r in failures)
    qc_bucket_counts = Counter(r["qc_bucket"] for r in failures)

    summary = {
        "version": "final62_128_ema_external31_failure_analysis_v1",
        "checkpoint": str(checkpoint),
        "resolution": [128, 128, 128],
        "threshold": 0.50,
        "external_gt_usage": "analysis_only_no_tuning",
        "n_cases": len(rows),
        "mean_dice": float(np.mean([float(r["dice"]) for r in rows])),
        "median_dice": float(np.median([float(r["dice"]) for r in rows])),
        "failure_dice_threshold": float(args.failure_dice),
        "n_failures": len(failures),
        "n_high_quality": int(sum(int(r["is_high_quality"]) for r in rows)),
        "failure_class_counts": dict(sorted(class_counts.items())),
        "failure_size_group_counts": dict(sorted(size_counts.items())),
        "failure_qc_bucket_counts": dict(sorted(qc_bucket_counts.items())),
        "gt_volume_q1_mm3": q1,
        "gt_volume_q3_mm3": q3,
        "boundary_band_mm": float(args.boundary_mm),
        "failure_case_ids": [r["case_id"] for r in failures],
        "warning": (
            "Failure classes and all spatial/error metrics are GT-derived retrospective diagnostics. "
            "Do not use them as deployment-time features or policy gates."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external31_ema_failure_analysis_all.csv", rows)
    write_csv(output_dir / "external31_ema_failure_analysis_failures.csv", failures)
    (output_dir / "external31_ema_failure_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "external31_ema_failure_analysis_report.md").write_text(
        markdown_report(rows, failures, summary), encoding="utf-8"
    )

    print("\n" + "=" * 124)
    print("FAILURE SUMMARY")
    print("=" * 124)
    print(f"Mean Dice:          {summary['mean_dice']:.4f}")
    print(f"Failures <.70:      {len(failures)}/{len(rows)}")
    print(f"High-quality >=.80: {summary['n_high_quality']}/{len(rows)}")
    print("Failure classes:")
    for name, count in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name:<24} {count}")
    print("Failure target sizes:")
    for name in ["SMALL", "MID", "LARGE"]:
        print(f"  {name:<24} {size_counts.get(name, 0)}")
    print("\nFAILED CASES (worst first)")
    print(
        f"{'case':<14} {'Dice':>6} {'class':<22} {'size':<6} {'Prec':>6} {'Rec':>6} "
        f"{'RVE':>8} {'CC':>4} {'LCF':>6} {'C/diag':>7} {'BndErr':>7} {'QC bucket':<26}"
    )
    for r in failures:
        print(
            f"{r['case_id'][:12]:<14} {r['dice']:>6.3f} {r['failure_class']:<22} "
            f"{r['gt_size_group']:<6} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['signed_rve_pct']:>+7.1f}% {r['pred_components']:>4d} "
            f"{r['pred_largest_component_fraction']:>6.2f} "
            f"{r['centroid_distance_over_gt_diag']:>7.2f} "
            f"{r['boundary_error_fraction']:>7.2f} {r['qc_bucket']:<26}"
        )

    print(f"\nOutputs: {output_dir}")
    print("Next decision should be based on which failure class dominates, not on another blind model change.")


if __name__ == "__main__":
    main()
