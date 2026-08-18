#!/usr/bin/env python3
"""Compare final Round-1 vs final Round-2 models on the frozen external 31-case benchmark.

This is a model-update validation, not threshold tuning.

Frozen conditions
-----------------
- exact external GT set resolved by intersection with the original pool manifest,
- expected matched count defaults to 31,
- original QC bucket assignments remain frozen from the Round-1 pool manifest,
- both models use the same raw student + EMA-teacher 50/50 probability ensemble,
- both models use the same fixed segmentation threshold 0.50,
- native-grid probability inversion is strict; no resized-space fallback is allowed.

The external labels are evaluation targets only. They are never used to choose a threshold,
retrain either segmentation model, or reassign QC buckets.
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.validate_external_threshold_31 import (
    BUCKET_ORDER,
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

DEFAULT_ROUND1_CHECKPOINT = Path(
    "experiments/final_supervised_round1_55_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_ROUND2_CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_POOL_MANIFEST = Path(
    "experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_final55_vs_final62")

MODEL_ORDER = {"FINAL55": 0, "FINAL62": 1}


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


def mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def median(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def evaluate_checkpoint(
    model_name,
    checkpoint,
    loader,
    inverse_transform,
    cases_by_id,
    config,
    device,
    threshold,
):
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError(f"{model_name}: checkpoint has no EMA teacher: {checkpoint}")
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            case_raw = batch.get("id")
            case_id = case_raw[0] if isinstance(case_raw, (list, tuple)) else str(case_raw)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                s_logits = cv.main_prediction(inferer(image_t, student))
                t_logits = cv.main_prediction(inferer(image_t, teacher))
                ensemble_prob_t = 0.5 * (torch.sigmoid(s_logits) + torch.sigmoid(t_logits))

            native_prob = invert_probability_exact(
                ensemble_prob_t,
                batch,
                inverse_transform,
                index=0,
            )
            source_image, native_prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)
            pred = native_prob_zyx > float(threshold)
            metrics = binary_metrics(pred, gt)

            row = {
                "case_id": case_id,
                "qc_bucket": case["qc_bucket"],
                "model": model_name,
                "checkpoint": str(checkpoint),
                "threshold": float(threshold),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{model_name} {index:2d}/{len(cases_by_id)}] {case_id} | "
                f"{case['qc_bucket']} | Dice={metrics['dice']:.4f} | "
                f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
                f"signedRVE={metrics['signed_rve_pct']:+.1f}%"
            )

    del student
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows, failure_dice, high_quality_dice):
    buckets = ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY", "ALL"]
    models = ["FINAL55", "FINAL62"]
    out = []
    for bucket in buckets:
        for model in models:
            subset = [
                row for row in rows
                if row["model"] == model and (bucket == "ALL" or row["qc_bucket"] == bucket)
            ]
            if not subset:
                continue
            dice = np.asarray([float(x["dice"]) for x in subset])
            precision = np.asarray([float(x["precision"]) for x in subset])
            recall = np.asarray([float(x["recall"]) for x in subset])
            signed = np.asarray([float(x["signed_rve_pct"]) for x in subset])
            fp = np.asarray([float(x["fp_vox"]) for x in subset])
            fn = np.asarray([float(x["fn_vox"]) for x in subset])
            out.append({
                "qc_bucket": bucket,
                "model": model,
                "n": len(subset),
                "mean_dice": mean(dice),
                "median_dice": median(dice),
                "mean_precision": mean(precision),
                "mean_recall": mean(recall),
                "mean_signed_rve_pct": mean(signed),
                "median_signed_rve_pct": median(signed),
                "median_abs_rve_pct": median(np.abs(signed)),
                "mean_fp_vox": mean(fp),
                "mean_fn_vox": mean(fn),
                "overseg_gt_20pct": int(np.sum(signed > 20.0)),
                "overseg_gt_50pct": int(np.sum(signed > 50.0)),
                "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
                "failures_dice_lt_070": int(np.sum(dice < failure_dice)),
                "high_quality_dice_gte_080": int(np.sum(dice >= high_quality_dice)),
            })
    out.sort(key=lambda x: (BUCKET_ORDER[x["qc_bucket"]], MODEL_ORDER[x["model"]]))
    return out


def paired_deltas(summary_rows):
    by_key = {(r["qc_bucket"], r["model"]): r for r in summary_rows}
    out = []
    for bucket in ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY", "ALL"]:
        old = by_key.get((bucket, "FINAL55"))
        new = by_key.get((bucket, "FINAL62"))
        if old is None or new is None:
            continue
        out.append({
            "qc_bucket": bucket,
            "n": int(old["n"]),
            "delta_mean_dice": float(new["mean_dice"]) - float(old["mean_dice"]),
            "delta_mean_precision": float(new["mean_precision"]) - float(old["mean_precision"]),
            "delta_mean_recall": float(new["mean_recall"]) - float(old["mean_recall"]),
            "delta_median_signed_rve_pct": float(new["median_signed_rve_pct"]) - float(old["median_signed_rve_pct"]),
            "delta_median_abs_rve_pct": float(new["median_abs_rve_pct"]) - float(old["median_abs_rve_pct"]),
            "delta_mean_fp_vox": float(new["mean_fp_vox"]) - float(old["mean_fp_vox"]),
            "delta_mean_fn_vox": float(new["mean_fn_vox"]) - float(old["mean_fn_vox"]),
            "final55_overseg_gt_20pct": int(old["overseg_gt_20pct"]),
            "final62_overseg_gt_20pct": int(new["overseg_gt_20pct"]),
            "final55_overseg_gt_50pct": int(old["overseg_gt_50pct"]),
            "final62_overseg_gt_50pct": int(new["overseg_gt_50pct"]),
            "final55_failures": int(old["failures_dice_lt_070"]),
            "final62_failures": int(new["failures_dice_lt_070"]),
            "final55_high_quality": int(old["high_quality_dice_gte_080"]),
            "final62_high_quality": int(new["high_quality_dice_gte_080"]),
        })
    return out


def paired_case_deltas(rows):
    by_key = {(r["case_id"], r["model"]): r for r in rows}
    case_ids = sorted({r["case_id"] for r in rows})
    out = []
    for case_id in case_ids:
        old = by_key[(case_id, "FINAL55")]
        new = by_key[(case_id, "FINAL62")]
        out.append({
            "case_id": case_id,
            "qc_bucket": old["qc_bucket"],
            "final55_dice": old["dice"],
            "final62_dice": new["dice"],
            "delta_dice": float(new["dice"]) - float(old["dice"]),
            "final55_precision": old["precision"],
            "final62_precision": new["precision"],
            "delta_precision": float(new["precision"]) - float(old["precision"]),
            "final55_recall": old["recall"],
            "final62_recall": new["recall"],
            "delta_recall": float(new["recall"]) - float(old["recall"]),
            "final55_signed_rve_pct": old["signed_rve_pct"],
            "final62_signed_rve_pct": new["signed_rve_pct"],
            "delta_signed_rve_pct": float(new["signed_rve_pct"]) - float(old["signed_rve_pct"]),
            "final55_fp_vox": old["fp_vox"],
            "final62_fp_vox": new["fp_vox"],
            "delta_fp_vox": int(new["fp_vox"]) - int(old["fp_vox"]),
            "final55_fn_vox": old["fn_vox"],
            "final62_fn_vox": new["fn_vox"],
            "delta_fn_vox": int(new["fn_vox"]) - int(old["fn_vox"]),
        })
    return out


def print_summary(summary_rows, deltas):
    print("\n" + "=" * 126)
    print("FROZEN EXTERNAL-31 MODEL UPDATE VALIDATION — FINAL55 vs FINAL62 @ THRESHOLD 0.50")
    print("=" * 126)
    print(
        "bucket                         model     n   meanDice  precision  recall   medSignedRVE  >+20%  >+50%  Dice<.70  Dice>=.80"
    )
    for row in summary_rows:
        print(
            f"{row['qc_bucket']:<30} {row['model']:<8} {int(row['n']):>3d}   "
            f"{float(row['mean_dice']):.4f}    {float(row['mean_precision']):.4f}    "
            f"{float(row['mean_recall']):.4f}    {float(row['median_signed_rve_pct']):+8.2f}%    "
            f"{int(row['overseg_gt_20pct']):>3d}     {int(row['overseg_gt_50pct']):>3d}      "
            f"{int(row['failures_dice_lt_070']):>3d}        {int(row['high_quality_dice_gte_080']):>3d}"
        )

    print("\nPAIRED MODEL CHANGE (FINAL62 - FINAL55)")
    print(
        "bucket                           Dice      Precision   Recall    medSignedRVE   >+20%   >+50%   failures   high-quality"
    )
    for row in deltas:
        print(
            f"{row['qc_bucket']:<30} {float(row['delta_mean_dice']):+8.4f}   "
            f"{float(row['delta_mean_precision']):+8.4f}  {float(row['delta_mean_recall']):+8.4f}   "
            f"{float(row['delta_median_signed_rve_pct']):+9.2f} pts     "
            f"{row['final55_overseg_gt_20pct']}->{row['final62_overseg_gt_20pct']}    "
            f"{row['final55_overseg_gt_50pct']}->{row['final62_overseg_gt_50pct']}     "
            f"{row['final55_failures']}->{row['final62_failures']}        "
            f"{row['final55_high_quality']}->{row['final62_high_quality']}"
        )

    hc = next((r for r in deltas if r["qc_bucket"] == "HIGH_CONFIDENCE_PSEUDO_LABEL"), None)
    active = next((r for r in deltas if r["qc_bucket"] == "ACTIVE_LEARN_PRIORITY"), None)
    print("\nDECISION CHECKS")
    if hc:
        print(
            f"  HIGH_CONF safety leakage: Dice<.70 {hc['final55_failures']} -> {hc['final62_failures']}."
        )
    if active:
        print(
            f"  ACTIVE failure tail: Dice {active['delta_mean_dice']:+.4f}, "
            f"precision {active['delta_mean_precision']:+.4f}, "
            f"median signed-RVE {active['delta_median_signed_rve_pct']:+.2f} pts."
        )
    print("  External labels remain evaluation-only; this script does not retune threshold or QC policy.")


def main():
    p = argparse.ArgumentParser(
        description="Compare final55 vs final62 on the frozen external 31-case benchmark"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External GT root; repeat for multiple roots")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round1-checkpoint", default=str(DEFAULT_ROUND1_CHECKPOINT))
    p.add_argument("--round2-checkpoint", default=str(DEFAULT_ROUND2_CHECKPOINT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    args = p.parse_args()

    if not 0.0 < args.threshold < 1.0:
        p.error("--threshold must be in (0,1)")
    if args.expected_count < 1:
        p.error("--expected-count must be >=1")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        p.error("Require 0 <= failure-dice < high-quality-dice <= 1")

    round1_checkpoint = Path(args.round1_checkpoint)
    round2_checkpoint = Path(args.round2_checkpoint)
    pool_manifest = Path(args.pool_manifest)
    output_dir = Path(args.output_dir)
    for path in [round1_checkpoint, round2_checkpoint, pool_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Frozen model comparison expects prototype student + EMA-teacher inference")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {case["id"]: case for case in cases}

    bucket_counts = {
        bucket: sum(case["qc_bucket"] == bucket for case in cases)
        for bucket in ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY"]
    }

    print("=" * 116)
    print("FROZEN EXTERNAL-31 FINAL MODEL COMPARISON")
    print(f"Cases:               {len(cases)}")
    print(f"Original QC buckets: {bucket_counts}")
    print(f"Threshold:           {args.threshold:.2f} (fixed for both models)")
    print(f"Final55 checkpoint:  {round1_checkpoint}")
    print(f"Final62 checkpoint:  {round2_checkpoint}")
    print("External GT is evaluation-only; original QC buckets remain frozen.")
    print("=" * 116)

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    loader_items = [{"image": case["image"], "id": case["id"]} for case in cases]
    loader = DataLoader(Dataset(loader_items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    all_rows = []
    all_rows.extend(evaluate_checkpoint(
        "FINAL55", round1_checkpoint, loader, inverse_transform, cases_by_id,
        config, device, args.threshold,
    ))
    all_rows.extend(evaluate_checkpoint(
        "FINAL62", round2_checkpoint, loader, inverse_transform, cases_by_id,
        config, device, args.threshold,
    ))

    all_rows.sort(key=lambda r: (str(r["case_id"]), MODEL_ORDER[r["model"]]))
    summary_rows = summarize(all_rows, args.failure_dice, args.high_quality_dice)
    deltas = paired_deltas(summary_rows)
    case_deltas = paired_case_deltas(all_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external31_model_case_metrics.csv", all_rows)
    write_csv(output_dir / "external31_model_bucket_summary.csv", summary_rows)
    write_csv(output_dir / "external31_model_paired_deltas.csv", deltas)
    write_csv(output_dir / "external31_model_case_paired_deltas.csv", case_deltas)

    metadata = {
        "version": "external31_final55_vs_final62_v1",
        "n_cases": len(cases),
        "expected_count": int(args.expected_count),
        "original_qc_bucket_counts": bucket_counts,
        "pool_manifest": str(pool_manifest),
        "round1_checkpoint": str(round1_checkpoint),
        "round2_checkpoint": str(round2_checkpoint),
        "threshold": float(args.threshold),
        "prediction_source": "raw_student_teacher_50_50_ensemble",
        "external_gt_dirs": list(args.gt_dir),
        "external_gt_usage": "evaluation_only",
        "qc_bucket_rule": "frozen_from_original_round1_pool_manifest",
        "threshold_rule": "fixed_before_external_model_update_comparison",
    }
    (output_dir / "external31_model_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print_summary(summary_rows, deltas)
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
