#!/usr/bin/env python3
"""Evaluate largest-connected-component post-processing on the frozen 47-case OOF baseline.

This script does not retrain the segmentation model and does not alter QC calibration.
It replays the exact frozen 5-fold held-out inference, then compares:

    RAW: ensemble probability > threshold
    LCC: RAW mask with only the single largest 6-connected component retained

The purpose is to decide, using held-out evidence, whether LCC should become part of the
official segmentation post-processing before rebuilding QC and rerunning the unlabeled pool.
If LCC is adopted, the current 10-case annotation batch must be treated as provisional and
reselected after QC recalibration and rerunning the 56-case pool.

Outputs:
  experiments/lcc_oof_evaluation/lcc_oof_case_comparison.csv
  experiments/lcc_oof_evaluation/lcc_oof_summary.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError("evaluate_lcc_oof.py requires scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import load_models


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/lcc_oof_evaluation")


def keep_largest_component(mask: np.ndarray):
    """Return single largest 6-connected foreground component and component statistics."""
    mask = np.asarray(mask).astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask.astype(np.uint8), 0, 0, 0.0

    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    largest_idx = int(np.argmax(sizes)) + 1
    largest_vox = int(sizes[largest_idx - 1])
    total_vox = int(mask.sum())
    lcc = labels == largest_idx
    retained_fraction = float(largest_vox / max(total_vox, 1))
    return lcc.astype(np.uint8), int(count), largest_vox, retained_fraction


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, prefix):
    def arr(name):
        return np.asarray([float(row[f"{prefix}_{name}"]) for row in rows], dtype=float)

    dice = arr("dice")
    precision = arr("precision")
    recall = arr("recall")
    rve = arr("rve")
    hd95 = arr("hd95")

    finite_hd95 = hd95[np.isfinite(hd95)]
    return {
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "mean_rve_pct": float(np.mean(rve)),
        "median_rve_pct": float(np.median(rve)),
        "mean_abs_rve_pct": float(np.mean(np.abs(rve))),
        "median_abs_rve_pct": float(np.median(np.abs(rve))),
        "mean_hd95_mm": float(np.mean(finite_hd95)) if finite_hd95.size else float("nan"),
        "median_hd95_mm": float(np.median(finite_hd95)) if finite_hd95.size else float("nan"),
        "dice_lt_0p20": int(np.sum(dice < 0.20)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p60": int(np.sum(dice < 0.60)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare raw vs keep-largest-component predictions on frozen OOF folds"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0,1)")

    experiment_dir = Path(args.experiment_dir)
    split_manifest = (
        Path(args.split_manifest)
        if args.split_manifest
        else experiment_dir / "cv_splits.json"
    )
    output_dir = Path(args.output_dir)
    if not split_manifest.exists():
        raise FileNotFoundError(split_manifest)

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, args.resize_size, epochs=100)
    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, args.folds, split_manifest)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    rows = []
    print("=" * 110)
    print("OOF LARGEST-CONNECTED-COMPONENT EVALUATION")
    print(f"Experiment: {experiment_dir}")
    print(f"Cases:      {len(cases)}")
    print(f"Threshold:  {args.threshold:.2f}")
    print("LCC rule:   keep one largest 6-connected foreground component")
    print("No training or QC calibration is changed by this script.")
    print("=" * 110)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        fold_cases = [by_id[case_id] for case_id in fold_spec["val_ids"]]
        loader = DataLoader(
            Dataset(fold_cases, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint, device)

        for batch in loader:
            image_t = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                if teacher is not None:
                    teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)
                else:
                    ensemble_prob_t = student_prob_t

            raw_np = (
                ensemble_prob_t[0, 0].detach().float().cpu().numpy() > float(args.threshold)
            ).astype(np.uint8)
            lcc_np, component_count, largest_vox, retained_fraction = keep_largest_component(raw_np)

            raw_t = torch.from_numpy(raw_np[None, None].astype(np.float32)).to(device)
            lcc_t = torch.from_numpy(lcc_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            raw = cv.case_metrics(raw_t, target_t, spacing)
            lcc = cv.case_metrics(lcc_t, target_t, spacing)

            row = {
                "fold": fold_idx,
                "case_id": case_id,
                "component_count_raw": int(component_count),
                "raw_pred_vox": int(raw_np.sum()),
                "lcc_pred_vox": int(lcc_np.sum()),
                "largest_component_vox": int(largest_vox),
                "largest_component_retained_fraction": float(retained_fraction),
                "removed_vox": int(raw_np.sum() - lcc_np.sum()),
                "removed_fraction": float(1.0 - retained_fraction) if raw_np.sum() else 0.0,
                "raw_dice": float(raw["dice"]),
                "lcc_dice": float(lcc["dice"]),
                "delta_dice": float(lcc["dice"] - raw["dice"]),
                "raw_precision": float(raw["precision"]),
                "lcc_precision": float(lcc["precision"]),
                "delta_precision": float(lcc["precision"] - raw["precision"]),
                "raw_recall": float(raw["recall"]),
                "lcc_recall": float(lcc["recall"]),
                "delta_recall": float(lcc["recall"] - raw["recall"]),
                "raw_rve": float(raw["rve"]),
                "lcc_rve": float(lcc["rve"]),
                "delta_abs_rve": float(abs(lcc["rve"]) - abs(raw["rve"])),
                "raw_hd95": float(raw["hd95"]),
                "lcc_hd95": float(lcc["hd95"]),
            }
            rows.append(row)

            print(
                f"[fold {fold_idx}] {case_id} | comp={component_count:4d} | "
                f"keep={retained_fraction:6.1%} | Dice {raw['dice']:.4f}->{lcc['dice']:.4f} "
                f"({row['delta_dice']:+.4f}) | Prec {raw['precision']:.4f}->{lcc['precision']:.4f} | "
                f"Rec {raw['recall']:.4f}->{lcc['recall']:.4f}"
            )

        del student
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    if len(rows) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} OOF rows, generated {len(rows)}")

    raw_summary = summarize(rows, "raw")
    lcc_summary = summarize(rows, "lcc")
    dice_delta = np.asarray([float(row["delta_dice"]) for row in rows], dtype=float)
    precision_delta = np.asarray([float(row["delta_precision"]) for row in rows], dtype=float)
    recall_delta = np.asarray([float(row["delta_recall"]) for row in rows], dtype=float)
    abs_rve_delta = np.asarray([float(row["delta_abs_rve"]) for row in rows], dtype=float)
    components = np.asarray([int(row["component_count_raw"]) for row in rows], dtype=int)

    multi_mask = components > 1
    changed = np.abs(dice_delta) > 1e-8
    materially_improved = dice_delta >= 0.01
    materially_worsened = dice_delta <= -0.01

    comparison = {}
    for key in raw_summary:
        if key in lcc_summary and isinstance(raw_summary[key], (int, float)):
            comparison[key] = float(lcc_summary[key] - raw_summary[key])

    summary = {
        "version": "lcc_oof_v1",
        "experiment_dir": str(experiment_dir),
        "split_manifest": str(split_manifest),
        "n_cases": len(rows),
        "threshold": float(args.threshold),
        "connectivity": "6-connected (scipy connectivity=1 in 3D)",
        "postprocess": "keep exactly one largest foreground component",
        "raw": raw_summary,
        "lcc": lcc_summary,
        "lcc_minus_raw": comparison,
        "case_effects": {
            "cases_with_multiple_components": int(np.sum(multi_mask)),
            "cases_with_single_or_zero_component": int(np.sum(~multi_mask)),
            "dice_improved_any": int(np.sum(dice_delta > 1e-8)),
            "dice_worsened_any": int(np.sum(dice_delta < -1e-8)),
            "dice_unchanged": int(np.sum(~changed)),
            "dice_improved_by_at_least_0p01": int(np.sum(materially_improved)),
            "dice_worsened_by_at_least_0p01": int(np.sum(materially_worsened)),
            "mean_delta_dice": float(np.mean(dice_delta)),
            "median_delta_dice": float(np.median(dice_delta)),
            "mean_delta_precision": float(np.mean(precision_delta)),
            "mean_delta_recall": float(np.mean(recall_delta)),
            "mean_delta_abs_rve_pct": float(np.mean(abs_rve_delta)),
            "multi_component_mean_delta_dice": (
                float(np.mean(dice_delta[multi_mask])) if np.any(multi_mask) else float("nan")
            ),
        },
        "interpretation_rule": (
            "Adopt LCC only if held-out results show a useful precision/RVE/failure improvement "
            "without a clinically unacceptable recall or Dice penalty. If adopted, rebuild QC and "
            "rerun unlabeled triage because the deployed mask distribution has changed."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lcc_oof_case_comparison.csv"
    json_path = output_dir / "lcc_oof_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best = sorted(rows, key=lambda row: float(row["delta_dice"]), reverse=True)[:5]
    worst = sorted(rows, key=lambda row: float(row["delta_dice"]))[:5]

    print("\n" + "=" * 110)
    print("RAW vs LCC OOF SUMMARY")
    print(f"Cases with >1 component: {int(np.sum(multi_mask))}/{len(rows)}")
    print(
        f"Mean Dice:      {raw_summary['mean_dice']:.4f} -> {lcc_summary['mean_dice']:.4f} "
        f"({lcc_summary['mean_dice'] - raw_summary['mean_dice']:+.4f})"
    )
    print(
        f"Mean Precision: {raw_summary['mean_precision']:.4f} -> {lcc_summary['mean_precision']:.4f} "
        f"({lcc_summary['mean_precision'] - raw_summary['mean_precision']:+.4f})"
    )
    print(
        f"Mean Recall:    {raw_summary['mean_recall']:.4f} -> {lcc_summary['mean_recall']:.4f} "
        f"({lcc_summary['mean_recall'] - raw_summary['mean_recall']:+.4f})"
    )
    print(
        f"Median |RVE|:   {raw_summary['median_abs_rve_pct']:.1f}% -> {lcc_summary['median_abs_rve_pct']:.1f}%"
    )
    print(
        f"Median HD95:    {raw_summary['median_hd95_mm']:.2f} -> {lcc_summary['median_hd95_mm']:.2f} mm"
    )
    print(
        f"Dice <0.70:     {raw_summary['dice_lt_0p70']} -> {lcc_summary['dice_lt_0p70']}"
    )
    print(
        f"Dice >=0.80:    {raw_summary['dice_ge_0p80']} -> {lcc_summary['dice_ge_0p80']}"
    )
    print(
        f"Case Dice effects: improved={int(np.sum(dice_delta > 1e-8))} | "
        f"worsened={int(np.sum(dice_delta < -1e-8))} | unchanged={int(np.sum(~changed))}"
    )
    print("\nLargest Dice improvements:")
    for row in best:
        print(
            f"  {row['case_id']} | components={row['component_count_raw']} | "
            f"keep={float(row['largest_component_retained_fraction']):.1%} | "
            f"Dice {float(row['raw_dice']):.4f}->{float(row['lcc_dice']):.4f} "
            f"({float(row['delta_dice']):+.4f})"
        )
    print("\nLargest Dice degradations:")
    for row in worst:
        print(
            f"  {row['case_id']} | components={row['component_count_raw']} | "
            f"keep={float(row['largest_component_retained_fraction']):.1%} | "
            f"Dice {float(row['raw_dice']):.4f}->{float(row['lcc_dice']):.4f} "
            f"({float(row['delta_dice']):+.4f})"
        )
    print(f"\nCase comparison: {csv_path}")
    print(f"Summary:         {json_path}")
    print("\nIf LCC wins clearly, next step is to make it the official mask postprocess, rebuild QC, and rerun the 56-case pool before annotation.")
    print("=" * 110)


if __name__ == "__main__":
    main()
