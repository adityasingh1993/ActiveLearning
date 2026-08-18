#!/usr/bin/env python3
"""Controlled Round-1 CV after adding new active-learning human labels.

Goal
----
Measure the effect of the newly annotated active-learning cases without changing the
segmentation recipe or the original held-out evaluation cases.

For each original frozen fold:

    Round-0 train IDs + ALL audited new human labels -> training
    exact same Round-0 val IDs                    -> held-out evaluation

The new active-learning labels are never placed in a validation fold for this comparison.
The held-out predictions therefore cover the same original 47 cases as Round 0.

To match the frozen Round-0 CV experiment, each fold uses:
- 100 epochs by default
- resize 128^3
- DiceCE / AdamW / LR 1e-4 / dropout 0 / lambda_unsup 0
- translation-only +/-12 voxels at p=0.8
- ensemble evaluation at threshold 0.50
- best_checkpoint.pth evaluation, exactly as the frozen Round-0 CV runner

The 81-epoch duration used for final_supervised_47_translation12 was the all-label deployment
fit and is deliberately NOT substituted for the frozen 100-epoch CV protocol here.

Prerequisite:
  python scripts/audit_round1_labels.py --config <config>

Default outputs:
  experiments/round1_cv_56_translation12/
    round1_cv_plan.json
    checkpoints/fold_*/...
    cv_results.csv
    round1_vs_round0_case_comparison.csv
    round1_vs_round0_summary.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
from scripts.audit_round1_labels import discover_round1_cases
from scripts.train_supervised_cv_47_translation import translation_only_transform


DEFAULT_SOURCE_CV_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_AUDIT_METADATA = Path("experiments/round1_supervised_56_translation12/round1_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/round1_cv_56_translation12")
EXPECTED_SOURCE_CASES = 47
EXPECTED_NEW_LABELS = 9
EXPECTED_TOTAL_LABELS = EXPECTED_SOURCE_CASES + EXPECTED_NEW_LABELS


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize(rows):
    dice = np.asarray([float(r["dice"]) for r in rows], dtype=float)
    precision = np.asarray([float(r["precision"]) for r in rows], dtype=float)
    recall = np.asarray([float(r["recall"]) for r in rows], dtype=float)
    rve = np.asarray([float(r["rve"]) for r in rows], dtype=float)
    hd95 = finite([float(r["hd95"]) for r in rows])
    return {
        "n_cases": int(len(rows)),
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "mean_rve_pct": float(np.mean(rve)),
        "median_rve_pct": float(np.median(rve)),
        "mean_hd95_mm": float(np.mean(hd95)) if hd95.size else float("nan"),
        "median_hd95_mm": float(np.median(hd95)) if hd95.size else float("nan"),
        "dice_lt_0p20": int(np.sum(dice < 0.20)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p60": int(np.sum(dice < 0.60)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def compare_to_round0(round0_rows, round1_rows, output_dir: Path):
    r0 = {str(row["case_id"]): row for row in round0_rows}
    r1 = {str(row["case_id"]): row for row in round1_rows}
    if len(r0) != EXPECTED_SOURCE_CASES or len(r1) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_CASES} unique held-out cases in both rounds; "
            f"Round0={len(r0)} Round1={len(r1)}"
        )
    if set(r0) != set(r1):
        raise RuntimeError(
            "Round-1 held-out case IDs differ from Round 0. Refusing a non-comparable summary."
        )

    comparison = []
    for case_id in sorted(r0):
        old = r0[case_id]
        new = r1[case_id]
        row = {
            "case_id": case_id,
            "fold_round0": int(old["fold"]),
            "fold_round1": int(new["fold"]),
            "round0_dice": float(old["dice"]),
            "round1_dice": float(new["dice"]),
            "delta_dice": float(new["dice"]) - float(old["dice"]),
            "round0_precision": float(old["precision"]),
            "round1_precision": float(new["precision"]),
            "delta_precision": float(new["precision"]) - float(old["precision"]),
            "round0_recall": float(old["recall"]),
            "round1_recall": float(new["recall"]),
            "delta_recall": float(new["recall"]) - float(old["recall"]),
            "round0_rve": float(old["rve"]),
            "round1_rve": float(new["rve"]),
            "delta_rve": float(new["rve"]) - float(old["rve"]),
            "round0_hd95": float(old["hd95"]),
            "round1_hd95": float(new["hd95"]),
            "delta_hd95": float(new["hd95"]) - float(old["hd95"]),
        }
        if row["fold_round0"] != row["fold_round1"]:
            raise RuntimeError(f"Case {case_id} moved folds between Round 0 and Round 1")
        comparison.append(row)

    r0_summary = summarize(list(r0.values()))
    r1_summary = summarize(list(r1.values()))
    deltas = np.asarray([row["delta_dice"] for row in comparison], dtype=float)
    summary = {
        "version": "round1_vs_round0_fixed_fold_v1",
        "round0": r0_summary,
        "round1": r1_summary,
        "delta": {
            "mean_dice": float(r1_summary["mean_dice"] - r0_summary["mean_dice"]),
            "mean_precision": float(r1_summary["mean_precision"] - r0_summary["mean_precision"]),
            "mean_recall": float(r1_summary["mean_recall"] - r0_summary["mean_recall"]),
            "median_rve_pct": float(r1_summary["median_rve_pct"] - r0_summary["median_rve_pct"]),
            "median_hd95_mm": float(r1_summary["median_hd95_mm"] - r0_summary["median_hd95_mm"]),
            "dice_lt_0p70": int(r1_summary["dice_lt_0p70"] - r0_summary["dice_lt_0p70"]),
            "dice_lt_0p50": int(r1_summary["dice_lt_0p50"] - r0_summary["dice_lt_0p50"]),
            "dice_ge_0p80": int(r1_summary["dice_ge_0p80"] - r0_summary["dice_ge_0p80"]),
        },
        "case_effects": {
            "improved": int(np.sum(deltas > 1e-6)),
            "worsened": int(np.sum(deltas < -1e-6)),
            "unchanged": int(np.sum(np.abs(deltas) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(deltas >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(deltas <= -0.05)),
        },
        "interpretation": (
            "Paired development comparison on the exact same original 47 held-out cases. "
            "The intended training change is addition of the audited Round-1 human labels only."
        ),
    }

    comparison_path = output_dir / "round1_vs_round0_case_comparison.csv"
    summary_path = output_dir / "round1_vs_round0_summary.json"
    write_csv(comparison_path, comparison)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print("ROUND-0 vs ROUND-1 FIXED-FOLD COMPARISON")
    print(f"Mean Dice:      {r0_summary['mean_dice']:.4f} -> {r1_summary['mean_dice']:.4f} "
          f"({summary['delta']['mean_dice']:+.4f})")
    print(f"Precision:      {r0_summary['mean_precision']:.4f} -> {r1_summary['mean_precision']:.4f} "
          f"({summary['delta']['mean_precision']:+.4f})")
    print(f"Recall:         {r0_summary['mean_recall']:.4f} -> {r1_summary['mean_recall']:.4f} "
          f"({summary['delta']['mean_recall']:+.4f})")
    print(f"Median RVE:     {r0_summary['median_rve_pct']:.1f}% -> {r1_summary['median_rve_pct']:.1f}%")
    print(f"Median HD95:    {r0_summary['median_hd95_mm']:.2f} -> {r1_summary['median_hd95_mm']:.2f} mm")
    print(f"Dice <0.70:     {r0_summary['dice_lt_0p70']} -> {r1_summary['dice_lt_0p70']}")
    print(f"Dice <0.50:     {r0_summary['dice_lt_0p50']} -> {r1_summary['dice_lt_0p50']}")
    print(f"Dice >=0.80:    {r0_summary['dice_ge_0p80']} -> {r1_summary['dice_ge_0p80']}")
    print(
        f"Case effects:   improved={summary['case_effects']['improved']} | "
        f"worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | "
        f"<=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print(f"Case CSV:       {comparison_path}")
    print(f"Summary JSON:   {summary_path}")
    print("=" * 104)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Round-1 CV with original fixed folds plus audited active-learning labels"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default="all", help="0..4 or all")
    parser.add_argument("--source-cv-dir", default=str(DEFAULT_SOURCE_CV_DIR))
    parser.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    parser.add_argument("--eval-threshold", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >=1")
    if not 0.0 < args.eval_threshold < 1.0:
        parser.error("--eval-threshold must be in (0,1)")

    source_cv_dir = Path(args.source_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    baseline_results_path = source_cv_dir / "cv_results.csv"
    audit_metadata_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)

    source_manifest = read_json(source_manifest_path)
    baseline_rows = cv.read_results(baseline_results_path)
    if len(source_manifest.get("all_case_ids", [])) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Source CV manifest is not the frozen {EXPECTED_SOURCE_CASES}-case manifest")
    if len({str(row["case_id"]) for row in baseline_rows}) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Round-0 cv_results.csv does not contain {EXPECTED_SOURCE_CASES} unique cases")

    audit = read_json(audit_metadata_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-1 audit metadata does not record a passing label audit")
    new_ids = sorted(str(x) for x in audit.get("new_human_label_ids", []))
    if len(new_ids) != EXPECTED_NEW_LABELS:
        raise RuntimeError(
            f"This controlled Round-1 profile expects exactly {EXPECTED_NEW_LABELS} audited new labels; "
            f"audit contains {len(new_ids)}: {new_ids}"
        )

    base_config = HASSLConfig.from_yaml(args.config)
    if base_config.compute_mode != "prototype":
        raise RuntimeError("Frozen Round-0 experiment used prototype student + EMA teacher")
    _, source_ids, by_id, current_new_ids = discover_round1_cases(base_config, source_manifest_path)
    if sorted(current_new_ids) != new_ids:
        raise RuntimeError(
            "Current label folder no longer matches the audited Round-1 label set. "
            f"Audited={new_ids} Current={sorted(current_new_ids)}. Re-run the audit before training."
        )
    if len(by_id) != EXPECTED_TOTAL_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_LABELS} total valid labels (47+9), found {len(by_id)}"
        )

    try:
        selected_folds = cv.parse_fold(args.fold, 5)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    # Reuse shared runner but replace its generic spatial transform with the frozen
    # translation-only transform. The Namespace contains exactly the fields run_fold/loader need.
    cv.spatial_aug_transform = translation_only_transform
    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=int(args.seed),
        resize_size=int(args.resize_size),
        epochs=int(args.epochs),
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source=args.eval_source,
        eval_threshold=float(args.eval_threshold),
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=12.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(baseline_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    fold_specs = []
    for original in source_manifest["folds"]:
        fold_idx = int(original["fold"])
        original_train = [str(x) for x in original["train_ids"]]
        original_val = [str(x) for x in original["val_ids"]]
        train_ids = sorted(set(original_train) | set(new_ids))
        val_ids = sorted(original_val)
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: train/held-out overlap detected")
        if set(new_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: a new AL label leaked into held-out evaluation")
        if not set(original_train).issubset(train_ids):
            raise RuntimeError(f"Fold {fold_idx}: original training cases were lost")
        fold_specs.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "original_train_ids": sorted(original_train),
            "new_round1_train_ids": new_ids,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "round1_cv_56_translation12_v1",
        "source_cv_dir": str(source_cv_dir),
        "source_manifest": str(source_manifest_path),
        "round0_results": str(baseline_results_path),
        "audit_metadata": str(audit_metadata_path),
        "n_frozen_source_labels": len(source_ids),
        "n_new_human_labels": len(new_ids),
        "n_total_human_labels": len(by_id),
        "new_human_label_ids": new_ids,
        "not_annotatable_ids": list(audit.get("selected_unlabeled_ids", [])),
        "folds": fold_specs,
        "recipe": {
            "epochs": int(args.epochs),
            "resize_size": [int(args.resize_size)] * 3,
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "rotation": 0.0,
            "scale": 0.0,
            "flips": False,
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": args.eval_source,
            "eval_threshold": float(args.eval_threshold),
            "checkpoint_for_evaluation": "best_checkpoint.pth",
        },
        "comparison_rule": (
            "Exact original val_ids are reused. Every audited new human label is appended only to each fold's training IDs."
        ),
    }
    plan_path = output_dir / "round1_cv_plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise RuntimeError(
                f"Existing Round-1 plan differs from current provenance: {plan_path}. "
                "Use a fresh output directory rather than mixing experiments."
            )
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(fold["fold"]): fold for fold in fold_specs}
    print("=" * 104)
    print("ROUND-1 ACTIVE-LEARNING CONTROLLED CV")
    print(f"Frozen source labels: {len(source_ids)}")
    print(f"New human labels:     {len(new_ids)}")
    print(f"Total human labels:   {len(by_id)}")
    print(f"Running folds:        {selected_folds}")
    print(f"New IDs in TRAIN ONLY: {', '.join(new_ids)}")
    print("Held-out IDs: exact original Round-0 folds; no new AL label is evaluated as held-out here.")
    print("Recipe: frozen Round-0 CV | 100 epochs default | translation +/-12 p=0.8 | ensemble @0.50")
    print(f"Plan: {plan_path}")
    print("=" * 104)

    new_rows = []
    for fold_idx in selected_folds:
        rows = cv.run_fold(runtime_args, fold_map[fold_idx], output_dir)
        for row in rows:
            row["experiment"] = "round1_56labels_translation12"
        new_rows.extend(rows)

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    selected_set = set(selected_folds)
    existing = [row for row in existing if int(row["fold"]) not in selected_set]
    merged = existing + new_rows
    merged.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    cv.write_results(results_path, merged)
    print(f"\nRound-1 results saved: {results_path}")

    completed_folds = {int(row["fold"]) for row in merged}
    if completed_folds == set(range(5)):
        if len({str(row["case_id"]) for row in merged}) != EXPECTED_SOURCE_CASES:
            raise RuntimeError("Completed Round-1 CV does not cover the exact original 47 held-out cases")
        cv.print_complete_summary(merged, 5)
        compare_to_round0(baseline_rows, merged, output_dir)
    else:
        print(f"Completed folds so far: {sorted(completed_folds)} / [0, 1, 2, 3, 4]")
        print("Run the remaining folds; the paired Round-0 comparison is emitted only after all five complete.")


if __name__ == "__main__":
    main()
