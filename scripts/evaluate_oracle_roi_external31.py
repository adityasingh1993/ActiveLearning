#!/usr/bin/env python3
"""Evaluate oracle-ROI Final62 EMA on frozen external31.

Ground-truth masks define the external crop, so this is an upper-bound feasibility experiment,
not deployable performance. The ROI prediction is mapped back into the original full native
volume before metrics are computed, then compared case-by-case with the current full-volume
Final62 EMA benchmark.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("evaluate_oracle_roi_external31.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.oracle_roi_utils import make_oracle_roi, paste_crop_into_full
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

DEFAULT_CHECKPOINT = Path("experiments/oracle_roi_final62_v1/checkpoints/final_checkpoint.pth")
DEFAULT_TRAINING_META = Path("experiments/oracle_roi_final62_v1/oracle_roi_training_metadata.json")
DEFAULT_POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
DEFAULT_ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_BASELINE_CASES = Path(
    "experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_oracle_roi_final62_v1")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
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
        writer.writerows(rows)


def summarize(rows, failure_dice=0.70, high_quality_dice=0.80):
    dice = np.asarray([float(r["dice"]) for r in rows], dtype=float)
    precision = np.asarray([float(r["precision"]) for r in rows], dtype=float)
    recall = np.asarray([float(r["recall"]) for r in rows], dtype=float)
    signed = np.asarray([float(r["signed_rve_pct"]) for r in rows], dtype=float)
    return {
        "n": len(rows),
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "std_dice": float(np.std(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "median_signed_rve_pct": float(np.median(signed)),
        "overseg_gt_20pct": int(np.sum(signed > 20.0)),
        "overseg_gt_50pct": int(np.sum(signed > 50.0)),
        "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
        "failures_dice_lt_070": int(np.sum(dice < failure_dice)),
        "high_quality_dice_gte_080": int(np.sum(dice >= high_quality_dice)),
        "dice_gte_090": int(np.sum(dice >= 0.90)),
    }


def baseline_ema_rows(path: Path, expected_ids):
    rows = read_csv(path)
    ema = [r for r in rows if str(r.get("mode", "")).upper() == "EMA"]
    by_id = {str(r["case_id"]): r for r in ema}
    if len(by_id) != len(ema):
        raise RuntimeError("Duplicate EMA case IDs in baseline inference-mode CSV")
    if set(by_id) != set(expected_ids):
        raise RuntimeError(
            "Baseline EMA case IDs do not exactly match the frozen external31 oracle evaluation set"
        )
    return by_id


def main():
    p = argparse.ArgumentParser(description="Evaluate GT-oracle ROI Final62 EMA on frozen external31")
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--training-metadata", default=str(DEFAULT_TRAINING_META))
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--baseline-ema-csv", default=str(DEFAULT_BASELINE_CASES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Oracle A/B is frozen at threshold 0.50")

    checkpoint = Path(args.checkpoint)
    training_meta_path = Path(args.training_metadata)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    baseline_path = Path(args.baseline_ema_csv)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, training_meta_path, pool_manifest, audit_path, baseline_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    training_meta = read_json(training_meta_path)
    margin_fraction = float(training_meta.get("oracle_margin_fraction_each_side", 0.40))
    audit = read_json(audit_path)
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Oracle ROI evaluation requires prototype checkpoint with EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError("EXTERNAL/TRAINING LEAKAGE: " + ", ".join(overlap))

    baseline_by_id = baseline_ema_rows(baseline_path, cases_by_id)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    crop_dir = output_dir / "oracle_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    crop_meta = {}
    loader_items = []
    for case in cases:
        crop_image, _, meta = make_oracle_roi(
            case["image"], case["gt_path"], margin_fraction=margin_fraction
        )
        crop_path = crop_dir / f"{case['id']}{config.image_suffix}"
        sitk.WriteImage(crop_image, str(crop_path), useCompression=True)
        crop_meta[case["id"]] = meta
        loader_items.append({"image": str(crop_path), "id": case["id"]})

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    loader = DataLoader(Dataset(loader_items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Oracle ROI checkpoint has no EMA teacher")
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 124)
    print("ORACLE ROI FINAL62 EMA — FROZEN EXTERNAL31 UPPER-BOUND EXPERIMENT")
    print(f"Cases:                {len(cases)}")
    print(f"Margin each side:     {margin_fraction:.2f} x GT bbox extent")
    print(f"Threshold:            {args.threshold:.2f}")
    print("External GT defines the crop. This is NOT deployable performance.")
    print("=" * 124)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            meta = crop_meta[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, teacher))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
            crop_ref_path = crop_dir / f"{case_id}{config.image_suffix}"
            _, crop_prob_zyx = normalize_native_probability(native_prob, crop_ref_path)
            crop_pred = crop_prob_zyx > float(args.threshold)

            full_source = sitk.ReadImage(str(case["image"]))
            gt = read_gt_binary(case["gt_path"], full_source)
            start_xyz = (meta["roi_start_x"], meta["roi_start_y"], meta["roi_start_z"])
            size_xyz = (meta["roi_size_x"], meta["roi_size_y"], meta["roi_size_z"])
            full_pred = paste_crop_into_full(crop_pred, gt.shape, start_xyz, size_xyz).astype(bool)
            metrics = binary_metrics(full_pred, gt)

            baseline = baseline_by_id[case_id]
            row = {
                "case_id": case_id,
                "baseline_full_volume_ema_dice": float(baseline["dice"]),
                "oracle_roi_ema_dice": float(metrics["dice"]),
                "delta_dice": float(metrics["dice"]) - float(baseline["dice"]),
                "baseline_full_volume_ema_precision": float(baseline["precision"]),
                "oracle_roi_ema_precision": float(metrics["precision"]),
                "baseline_full_volume_ema_recall": float(baseline["recall"]),
                "oracle_roi_ema_recall": float(metrics["recall"]),
                "baseline_full_volume_signed_rve_pct": float(baseline["signed_rve_pct"]),
                "oracle_roi_signed_rve_pct": float(metrics["signed_rve_pct"]),
                **meta,
                **metrics,
            }
            rows.append(row)
            print(
                f"[{index:2d}/{len(cases)}] {case_id} | "
                f"full EMA={float(baseline['dice']):.4f} -> oracle ROI={metrics['dice']:.4f} "
                f"({row['delta_dice']:+.4f})"
            )

    rows.sort(key=lambda r: float(r["baseline_full_volume_ema_dice"]))
    oracle_summary = summarize(rows, args.failure_dice, args.high_quality_dice)

    baseline_rows_for_summary = []
    for case_id in sorted(cases_by_id):
        r = baseline_by_id[case_id]
        baseline_rows_for_summary.append({
            "dice": float(r["dice"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "signed_rve_pct": float(r["signed_rve_pct"]),
        })
    baseline_summary = summarize(baseline_rows_for_summary, args.failure_dice, args.high_quality_dice)

    output_dir.mkdir(parents=True, exist_ok=True)
    case_csv = output_dir / "external31_oracle_roi_case_comparison.csv"
    write_csv(case_csv, rows)
    payload = {
        "version": "external31_oracle_roi_final62_v1",
        "experiment_type": "GT-defined oracle ROI upper bound",
        "checkpoint": str(checkpoint),
        "baseline_ema_csv": str(baseline_path),
        "n_cases": len(rows),
        "margin_fraction_each_side": margin_fraction,
        "threshold": float(args.threshold),
        "baseline_full_volume_ema": baseline_summary,
        "oracle_roi_ema": oracle_summary,
        "delta_mean_dice": oracle_summary["mean_dice"] - baseline_summary["mean_dice"],
        "delta_median_dice": oracle_summary["median_dice"] - baseline_summary["median_dice"],
        "warning": (
            "External ground truth defines each ROI. These results quantify the potential value of perfect "
            "offline localization/high-resolution refinement and are not deployable inference performance."
        ),
    }
    json_path = output_dir / "external31_oracle_roi_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("ORACLE ROI EXTERNAL31 RESULT")
    print(f"Full-volume EMA mean Dice:   {baseline_summary['mean_dice']:.4f}")
    print(f"Oracle-ROI EMA mean Dice:    {oracle_summary['mean_dice']:.4f}")
    print(f"Mean Dice change:            {oracle_summary['mean_dice'] - baseline_summary['mean_dice']:+.4f}")
    print(f"Full-volume / ROI median:    {baseline_summary['median_dice']:.4f} / {oracle_summary['median_dice']:.4f}")
    print(f"Dice<.70:                    {baseline_summary['failures_dice_lt_070']} -> {oracle_summary['failures_dice_lt_070']}")
    print(f"Dice>=.80:                   {baseline_summary['high_quality_dice_gte_080']} -> {oracle_summary['high_quality_dice_gte_080']}")
    print(f"Dice>=.90:                   {baseline_summary['dice_gte_090']} -> {oracle_summary['dice_gte_090']}")
    print(f"Median |RVE|:                {baseline_summary['median_abs_rve_pct']:.2f}% -> {oracle_summary['median_abs_rve_pct']:.2f}%")
    print("\nPreviously low-Dice cases:")
    for row in rows:
        if float(row["baseline_full_volume_ema_dice"]) < args.failure_dice:
            print(
                f"  {row['case_id']} | {float(row['baseline_full_volume_ema_dice']):.4f} "
                f"-> {float(row['oracle_roi_ema_dice']):.4f} ({float(row['delta_dice']):+.4f})"
            )
    print(f"\nCases:   {case_csv}")
    print(f"Summary: {json_path}")
    print("Interpretation: use only to decide whether ROI refinement is worth adding to OFFLINE AL/auto-labeling.")
    print("=" * 124)


if __name__ == "__main__":
    main()
