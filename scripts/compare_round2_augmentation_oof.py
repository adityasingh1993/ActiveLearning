#!/usr/bin/env python3
"""Paired OOF comparison of Final62 Round-2 augmentation experiments.

The comparison is intentionally development-only and never reads external31. All compared
runs must contain the exact same 47 original held-out cases in the exact same folds:

  A0  experiments/round2_cv_62_translation12
      frozen translation-only reference
  A1  experiments/round2_cv_62_translation12_aug_appearance_v1
      translation + conservative appearance augmentation
  A2  experiments/round2_cv_62_translation12_aug_mildaffine_v1
      translation + +/-3deg rotation + +/-5% scale

The 15 post-source HUMAN_GOLD cases remain train-only in all three experiments. This makes the
case-by-case deltas paired and avoids repeatedly using external31 for augmentation selection.

In addition to global metrics, target-size strata are defined ONCE from baseline GT voxel counts
using tertiles. Because all held-out labels are transformed to the same 128^3 model space, this
provides a stable development diagnostic for whether an augmentation specifically helps small
foreground targets.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_RUNS = {
    "A0_translation": Path("experiments/round2_cv_62_translation12"),
    "A1_appearance": Path("experiments/round2_cv_62_translation12_aug_appearance_v1"),
    "A2_mild_affine": Path("experiments/round2_cv_62_translation12_aug_mildaffine_v1"),
}
DEFAULT_OUTPUT = Path("experiments/round2_augmentation_comparison")
EXPECTED_CASES = 47


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


def f(row, key):
    return float(row[key])


def enrich(row):
    out = dict(row)
    pred = f(out, "pred_vox")
    gt = f(out, "gt_vox")
    dice = f(out, "dice")
    eps = 1e-5
    tp = (dice * (pred + gt + eps) - eps) / 2.0
    tp = min(max(tp, 0.0), min(pred, gt))
    fp = max(pred - tp, 0.0)
    fn = max(gt - tp, 0.0)
    signed_rve = 100.0 * (pred - gt) / (gt + 1e-8)
    out.update(
        signed_rve=float(signed_rve),
        abs_rve=float(abs(signed_rve)),
        tp_vox=float(tp),
        fp_vox=float(fp),
        fn_vox=float(fn),
    )
    return out


def load_run(name, directory):
    rows = [enrich(row) for row in read_csv(directory / "cv_results.csv")]
    by_id = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in by_id:
            raise RuntimeError(f"{name}: duplicate case_id {case_id}")
        by_id[case_id] = row
    if len(by_id) != EXPECTED_CASES:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_CASES} held-out cases, found {len(by_id)}. "
            "Complete all five folds before comparing."
        )
    return by_id


def validate_pairing(runs):
    names = list(runs)
    ref_name = names[0]
    ref = runs[ref_name]
    ref_ids = set(ref)
    for name in names[1:]:
        current = runs[name]
        if set(current) != ref_ids:
            missing = sorted(ref_ids - set(current))
            extra = sorted(set(current) - ref_ids)
            raise RuntimeError(
                f"{name}: held-out case IDs differ from {ref_name}; missing={missing[:10]}, extra={extra[:10]}"
            )
        for case_id in sorted(ref_ids):
            if int(current[case_id]["fold"]) != int(ref[case_id]["fold"]):
                raise RuntimeError(
                    f"{case_id}: fold changed {ref_name}={ref[case_id]['fold']} vs "
                    f"{name}={current[case_id]['fold']}"
                )
            if abs(f(current[case_id], "gt_vox") - f(ref[case_id], "gt_vox")) > 1e-6:
                raise RuntimeError(f"{case_id}: GT voxel count changed between paired runs")


def size_thresholds(reference):
    gt = np.asarray([f(row, "gt_vox") for row in reference.values()], dtype=float)
    return float(np.quantile(gt, 1.0 / 3.0)), float(np.quantile(gt, 2.0 / 3.0))


def size_group(gt_vox, q1, q2):
    if gt_vox <= q1:
        return "SMALL"
    if gt_vox <= q2:
        return "MID"
    return "LARGE"


def finite_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_median(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def summarize(rows):
    dice = np.asarray([f(r, "dice") for r in rows], dtype=float)
    precision = np.asarray([f(r, "precision") for r in rows], dtype=float)
    recall = np.asarray([f(r, "recall") for r in rows], dtype=float)
    signed = np.asarray([float(r["signed_rve"]) for r in rows], dtype=float)
    hd95 = [f(r, "hd95") for r in rows]
    return {
        "n": len(rows),
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "overseg_gt_20pct": int(np.sum(signed > 20.0)),
        "overseg_gt_50pct": int(np.sum(signed > 50.0)),
        "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
        "mean_hd95_mm": finite_mean(hd95),
        "median_hd95_mm": finite_median(hd95),
    }


def paired_effects(reference, candidate):
    ids = sorted(reference)
    delta = np.asarray(
        [f(candidate[x], "dice") - f(reference[x], "dice") for x in ids], dtype=float
    )
    return {
        "mean_case_dice_delta": float(np.mean(delta)),
        "median_case_dice_delta": float(np.median(delta)),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "unchanged": int(np.sum(np.abs(delta) <= 1e-6)),
        "improved_ge_0p05": int(np.sum(delta >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
    }


def print_global_table(summaries):
    print("\n" + "=" * 132)
    print("FINAL62 ROUND-2 AUGMENTATION — PAIRED ORIGINAL47 OOF")
    print("=" * 132)
    print(
        f"{'run':<20} {'n':>3} {'meanDice':>9} {'median':>8} {'precision':>10} {'recall':>8} "
        f"{'med|RVE|':>10} {'>20':>5} {'>50':>5} {'<-20':>5} {'Dice<.70':>9} {'Dice>=.80':>10}"
    )
    for name, s in summaries.items():
        print(
            f"{name:<20} {s['n']:>3d} {s['mean_dice']:>9.4f} {s['median_dice']:>8.4f} "
            f"{s['mean_precision']:>10.4f} {s['mean_recall']:>8.4f} {s['median_abs_rve_pct']:>9.2f}% "
            f"{s['overseg_gt_20pct']:>5d} {s['overseg_gt_50pct']:>5d} "
            f"{s['underseg_lt_minus20pct']:>5d} {s['dice_lt_0p70']:>9d} {s['dice_ge_0p80']:>10d}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare controlled Final62 augmentation OOF runs")
    parser.add_argument("--baseline-dir", default=str(DEFAULT_RUNS["A0_translation"]))
    parser.add_argument("--appearance-dir", default=str(DEFAULT_RUNS["A1_appearance"]))
    parser.add_argument("--mild-affine-dir", default=str(DEFAULT_RUNS["A2_mild_affine"]))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    dirs = {
        "A0_translation": Path(args.baseline_dir),
        "A1_appearance": Path(args.appearance_dir),
        "A2_mild_affine": Path(args.mild_affine_dir),
    }
    runs = {name: load_run(name, path) for name, path in dirs.items()}
    validate_pairing(runs)

    baseline = runs["A0_translation"]
    q1, q2 = size_thresholds(baseline)
    summaries = {
        name: summarize([rows[x] for x in sorted(rows)]) for name, rows in runs.items()
    }
    effects = {
        name: paired_effects(baseline, rows)
        for name, rows in runs.items()
        if name != "A0_translation"
    }

    print_global_table(summaries)
    print("\nPAIRED DICE EFFECTS vs A0")
    for name, e in effects.items():
        print(
            f"  {name:<18} mean delta={e['mean_case_dice_delta']:+.4f} | "
            f"improved={e['improved']:2d} worsened={e['worsened']:2d} | "
            f"delta>=+.05={e['improved_ge_0p05']:2d} delta<=-.05={e['worsened_le_minus_0p05']:2d}"
        )

    size_summaries = {}
    print("\nTARGET-SIZE DIAGNOSTIC (tertiles defined from A0 GT voxels)")
    print(f"  SMALL <= {q1:.1f} vox | MID <= {q2:.1f} vox | LARGE > {q2:.1f} vox")
    print(f"{'run':<20} {'size':<7} {'n':>3} {'meanDice':>9} {'median':>8} {'Dice<.70':>9} {'Dice>=.80':>10}")
    for name, rows in runs.items():
        size_summaries[name] = {}
        for group in ("SMALL", "MID", "LARGE"):
            subset = [
                row
                for row in rows.values()
                if size_group(f(row, "gt_vox"), q1, q2) == group
            ]
            s = summarize(subset)
            size_summaries[name][group] = s
            print(
                f"{name:<20} {group:<7} {s['n']:>3d} {s['mean_dice']:>9.4f} {s['median_dice']:>8.4f} "
                f"{s['dice_lt_0p70']:>9d} {s['dice_ge_0p80']:>10d}"
            )

    case_rows = []
    for case_id in sorted(baseline):
        base = baseline[case_id]
        group = size_group(f(base, "gt_vox"), q1, q2)
        row = {
            "case_id": case_id,
            "fold": int(base["fold"]),
            "size_group": group,
            "gt_vox": f(base, "gt_vox"),
        }
        base_dice = f(base, "dice")
        for name, run in runs.items():
            r = run[case_id]
            row[f"{name}_dice"] = f(r, "dice")
            row[f"{name}_precision"] = f(r, "precision")
            row[f"{name}_recall"] = f(r, "recall")
            row[f"{name}_signed_rve"] = float(r["signed_rve"])
            if name != "A0_translation":
                row[f"{name}_delta_dice"] = f(r, "dice") - base_dice
        case_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "augmentation_case_comparison.csv", case_rows)

    payload = {
        "version": "final62_round2_augmentation_paired_oof_v1",
        "comparison_cohort": "exact original47 held-out folds; 15 added HUMAN_GOLD train-only",
        "external31_used": False,
        "size_thresholds_gt_vox": {"small_max": q1, "mid_max": q2},
        "global": summaries,
        "paired_effects_vs_A0": effects,
        "by_target_size": size_summaries,
        "selection_guidance": (
            "Prefer an augmentation only if it improves paired OOF mean/median Dice without increasing "
            "Dice<.70 failures; give extra weight to reducing SMALL-target failures. Do not inspect "
            "external31 until a development winner is frozen."
        ),
    }
    (output_dir / "augmentation_comparison_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nDecision rule:")
    print("  1) no increase in Dice<.70 failures")
    print("  2) paired mean/median Dice improves")
    print("  3) prefer fewer SMALL-target failures if global metrics are close")
    print("  4) external31 remains untouched until one profile is frozen")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
