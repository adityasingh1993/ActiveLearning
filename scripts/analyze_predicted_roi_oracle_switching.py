#!/usr/bin/env python3
"""Measure complementarity between full-volume EMA and jittered predicted-ROI refinement.

This is an ORACLE switching upper-bound analysis only. For each frozen external31 case it uses
GT Dice to choose whichever existing prediction is better:

    oracle_switch_dice = max(full_volume_ema_dice, predicted_roi_refined_dice)

No model is trained and no deployable selector is implied. The purpose is to answer whether the
full-volume model and ROI refiner are complementary enough to justify building a leakage-safe,
guided selector/refiner next.

The script also compares the switch upper bound with the GT-oracle ROI reference and reports
which cases would benefit from ROI refinement.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_BASELINE = Path(
    "experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv"
)
DEFAULT_PREDICTED = Path(
    "experiments/external31_predicted_roi_jittered_v1/external31_predicted_roi_case_comparison.csv"
)
DEFAULT_ORACLE = Path(
    "experiments/external31_oracle_roi_final62_v1/external31_oracle_roi_case_comparison.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_roi_oracle_switching_v1")


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
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
        writer.writerows(rows)


def unique_by_case(rows, label):
    by_id = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise RuntimeError(f"{label} contains an empty case_id")
        if case_id in by_id:
            raise RuntimeError(f"Duplicate case_id in {label}: {case_id}")
        by_id[case_id] = row
    return by_id


def summarize(dice_values, failure=0.70, high_quality=0.80):
    arr = np.asarray(dice_values, dtype=float)
    return {
        "n": int(arr.size),
        "mean_dice": float(np.mean(arr)),
        "median_dice": float(np.median(arr)),
        "std_dice": float(np.std(arr)),
        "min_dice": float(np.min(arr)),
        "max_dice": float(np.max(arr)),
        "failures_dice_lt_070": int(np.sum(arr < failure)),
        "high_quality_dice_gte_080": int(np.sum(arr >= high_quality)),
        "dice_gte_090": int(np.sum(arr >= 0.90)),
    }


def main():
    p = argparse.ArgumentParser(
        description="Oracle switch upper bound: full-volume EMA vs jittered predicted-ROI refinement"
    )
    p.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE))
    p.add_argument("--predicted-roi-csv", default=str(DEFAULT_PREDICTED))
    p.add_argument("--oracle-roi-csv", default=str(DEFAULT_ORACLE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    args = p.parse_args()

    baseline_path = Path(args.baseline_csv)
    predicted_path = Path(args.predicted_roi_csv)
    oracle_path = Path(args.oracle_roi_csv)
    output_dir = Path(args.output_dir)

    baseline_rows_all = read_csv(baseline_path)
    baseline_rows = [r for r in baseline_rows_all if str(r.get("mode", "")).upper() == "EMA"]
    baseline = unique_by_case(baseline_rows, "baseline EMA CSV")
    predicted = unique_by_case(read_csv(predicted_path), "predicted ROI CSV")
    oracle = unique_by_case(read_csv(oracle_path), "oracle ROI CSV")

    ids = sorted(set(baseline))
    if set(predicted) != set(ids) or set(oracle) != set(ids):
        raise RuntimeError(
            "Case IDs do not match exactly across baseline, predicted-ROI, and oracle-ROI CSVs"
        )
    if args.expected_count > 0 and len(ids) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} cases, found {len(ids)}")

    rows = []
    for case_id in ids:
        full_dice = float(baseline[case_id]["dice"])
        pred_dice = float(predicted[case_id]["predicted_roi_refined_ema_dice"])
        oracle_dice = float(oracle[case_id]["oracle_roi_ema_dice"])
        coverage = float(predicted[case_id].get("gt_coverage_by_predicted_roi", "nan"))

        if pred_dice > full_dice:
            chosen = "PREDICTED_ROI"
            switch_dice = pred_dice
        else:
            chosen = "FULL_VOLUME"
            switch_dice = full_dice

        oracle_full_switch = max(full_dice, oracle_dice)
        rows.append({
            "case_id": case_id,
            "full_volume_ema_dice": full_dice,
            "predicted_roi_refined_dice": pred_dice,
            "oracle_roi_dice": oracle_dice,
            "predicted_minus_full": pred_dice - full_dice,
            "oracle_minus_full": oracle_dice - full_dice,
            "oracle_switch_choice": chosen,
            "oracle_switch_dice": switch_dice,
            "oracle_switch_gain_vs_full": switch_dice - full_dice,
            "gt_oracle_roi_or_full_dice": oracle_full_switch,
            "gt_oracle_roi_or_full_gain_vs_full": oracle_full_switch - full_dice,
            "gt_coverage_by_predicted_roi": coverage,
        })

    rows.sort(key=lambda r: float(r["full_volume_ema_dice"]))

    full_summary = summarize(
        [r["full_volume_ema_dice"] for r in rows], args.failure_dice, args.high_quality_dice
    )
    pred_summary = summarize(
        [r["predicted_roi_refined_dice"] for r in rows], args.failure_dice, args.high_quality_dice
    )
    switch_summary = summarize(
        [r["oracle_switch_dice"] for r in rows], args.failure_dice, args.high_quality_dice
    )
    oracle_summary = summarize(
        [r["oracle_roi_dice"] for r in rows], args.failure_dice, args.high_quality_dice
    )
    oracle_full_switch_summary = summarize(
        [r["gt_oracle_roi_or_full_dice"] for r in rows], args.failure_dice, args.high_quality_dice
    )

    roi_wins = [r for r in rows if r["oracle_switch_choice"] == "PREDICTED_ROI"]
    meaningful_roi_wins = [r for r in roi_wins if float(r["predicted_minus_full"]) >= 0.02]
    strong_roi_wins = [r for r in roi_wins if float(r["predicted_minus_full"]) >= 0.05]

    switch_gain = switch_summary["mean_dice"] - full_summary["mean_dice"]
    oracle_total_gain = oracle_summary["mean_dice"] - full_summary["mean_dice"]
    complement_fraction = switch_gain / oracle_total_gain if oracle_total_gain > 1e-12 else float("nan")

    if switch_gain >= 0.03:
        recommendation = "PROCEED_GUIDED_REFINER_STRONG_HEADROOM"
    elif switch_gain >= 0.015:
        recommendation = "PROCEED_GUIDED_REFINER_MODERATE_HEADROOM"
    else:
        recommendation = "LIMITED_COMPLEMENTARITY_RECONSIDER_GUIDED_REFINER"

    payload = {
        "version": "external31_roi_oracle_switching_v1",
        "experiment_type": "GT-oracle per-case switch upper bound; not deployable",
        "n_cases": len(rows),
        "baseline_full_volume_ema": full_summary,
        "predicted_roi_refiner": pred_summary,
        "oracle_switch_full_vs_predicted_roi": switch_summary,
        "oracle_roi_reference": oracle_summary,
        "oracle_switch_full_vs_gt_oracle_roi": oracle_full_switch_summary,
        "oracle_switch_gain_vs_full_mean_dice": switch_gain,
        "oracle_roi_total_gain_vs_full_mean_dice": oracle_total_gain,
        "fraction_of_oracle_roi_mean_gain_available_from_existing_refiner_complementarity": complement_fraction,
        "predicted_roi_wins_n": len(roi_wins),
        "predicted_roi_wins_ge_002_n": len(meaningful_roi_wins),
        "predicted_roi_wins_ge_005_n": len(strong_roi_wins),
        "recommendation": recommendation,
        "warning": (
            "The switch uses external GT Dice to select the better output per case. It is an upper bound only. "
            "Do not report it as deployable or prospective performance."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "external31_roi_oracle_switch_case_comparison.csv"
    summary_path = output_dir / "external31_roi_oracle_switch_summary.json"
    write_csv(case_path, rows)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 116)
    print("ORACLE SWITCHING UPPER BOUND — FULL EMA vs JITTERED PREDICTED ROI")
    print(f"Cases:                         {len(rows)}")
    print(f"Full-volume EMA mean Dice:     {full_summary['mean_dice']:.4f}")
    print(f"Predicted-ROI mean Dice:       {pred_summary['mean_dice']:.4f}")
    print(f"Oracle-switch mean Dice:       {switch_summary['mean_dice']:.4f}")
    print(f"Oracle ROI mean Dice:          {oracle_summary['mean_dice']:.4f}")
    print(f"Full-or-oracleROI upper bound: {oracle_full_switch_summary['mean_dice']:.4f}")
    print(f"Switch gain vs full:           {switch_gain:+.4f}")
    print(f"ROI wins cases:                {len(roi_wins)}/{len(rows)}")
    print(f"ROI wins by >=.02 Dice:        {len(meaningful_roi_wins)}/{len(rows)}")
    print(f"ROI wins by >=.05 Dice:        {len(strong_roi_wins)}/{len(rows)}")
    print(f"Dice<.70 full -> switch:       {full_summary['failures_dice_lt_070']} -> {switch_summary['failures_dice_lt_070']}")
    print(f"Dice>=.80 full -> switch:      {full_summary['high_quality_dice_gte_080']} -> {switch_summary['high_quality_dice_gte_080']}")
    print(f"Dice>=.90 full -> switch:      {full_summary['dice_gte_090']} -> {switch_summary['dice_gte_090']}")
    print(f"Complement/oracle fraction:    {complement_fraction:.3f}")
    print(f"Decision:                      {recommendation}")

    print("\nCases where predicted ROI beats full-volume EMA:")
    if not roi_wins:
        print("  none")
    else:
        for row in sorted(roi_wins, key=lambda r: float(r["predicted_minus_full"]), reverse=True):
            print(
                f"  {row['case_id']} | full={float(row['full_volume_ema_dice']):.4f} "
                f"predROI={float(row['predicted_roi_refined_dice']):.4f} "
                f"gain={float(row['predicted_minus_full']):+.4f} "
                f"coverage={float(row['gt_coverage_by_predicted_roi']):.3f}"
            )

    print(f"\nCases:   {case_path}")
    print(f"Summary: {summary_path}")
    print("WARNING: GT chooses the winner; this is not deployable performance.")
    print("=" * 116)


if __name__ == "__main__":
    main()
