#!/usr/bin/env python3
"""Controlled Round-2 CV from a passing provenance-safe Round-2 audit.

Scientific comparison
---------------------
The exact original 47 held-out validation IDs and fold assignments are reused. Every audited
human label added after the original 47 (Round-1 + Round-2) is appended to TRAIN ONLY in each
fold. No pseudo-label or external validation label is allowed into this controlled run.

The segmentation recipe is intentionally unchanged from the successful Round-1 experiment:
- DynUNet / prototype student + EMA teacher
- resize 128^3
- DiceCE
- random initialization
- dropout 0
- LR 1e-4
- lambda_unsup 0
- translation-only +/-12 voxels, p=0.8
- raw student/teacher 50/50 ensemble
- threshold 0.50

In addition to the historical metrics, this runner derives signed relative volume error and
FP/FN voxel counts from each held-out prediction. Positive signed RVE means over-segmentation;
negative signed RVE means under-segmentation.
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
DEFAULT_ROUND1_CV_DIR = Path("experiments/round1_cv_55_translation12")
EXPECTED_SOURCE_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def enrich_volume_metrics(row):
    """Add signed RVE and FP/FN counts without rerunning inference.

    cv.case_metrics already stores Dice, pred_vox and gt_vox. Recover TP from the exact Dice
    equation used there, then derive FP/FN. Existing CSV rows can therefore be enriched too.
    """
    out = dict(row)
    gt = float(out["gt_vox"])
    pred = float(out["pred_vox"])
    dice = float(out["dice"])
    eps = 1e-5

    signed_rve = 100.0 * (pred - gt) / (gt + 1e-8)
    tp = (dice * (pred + gt + eps) - eps) / 2.0
    tp = min(max(tp, 0.0), min(pred, gt))
    fp = max(pred - tp, 0.0)
    fn = max(gt - tp, 0.0)

    out["signed_rve"] = float(signed_rve)
    out["volume_ratio"] = float(pred / (gt + 1e-8))
    out["tp_vox"] = float(tp)
    out["fp_vox"] = float(fp)
    out["fn_vox"] = float(fn)
    out["overseg_gt_20pct"] = int(signed_rve > 20.0)
    out["overseg_gt_50pct"] = int(signed_rve > 50.0)
    out["underseg_lt_minus20pct"] = int(signed_rve < -20.0)
    return out


def finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize(rows):
    rows = [enrich_volume_metrics(row) for row in rows]
    dice = np.asarray([float(r["dice"]) for r in rows], dtype=float)
    precision = np.asarray([float(r["precision"]) for r in rows], dtype=float)
    recall = np.asarray([float(r["recall"]) for r in rows], dtype=float)
    signed = np.asarray([float(r["signed_rve"]) for r in rows], dtype=float)
    abs_rve = np.abs(signed)
    hd95 = finite([float(r["hd95"]) for r in rows])
    fp = np.asarray([float(r["fp_vox"]) for r in rows], dtype=float)
    fn = np.asarray([float(r["fn_vox"]) for r in rows], dtype=float)

    return {
        "n_cases": len(rows),
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "mean_signed_rve_pct": float(np.mean(signed)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(abs_rve)),
        "overseg_gt_20pct": int(np.sum(signed > 20.0)),
        "overseg_gt_50pct": int(np.sum(signed > 50.0)),
        "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
        "mean_fp_vox": float(np.mean(fp)),
        "mean_fn_vox": float(np.mean(fn)),
        "mean_hd95_mm": float(np.mean(hd95)) if hd95.size else float("nan"),
        "median_hd95_mm": float(np.median(hd95)) if hd95.size else float("nan"),
        "dice_lt_0p20": int(np.sum(dice < 0.20)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p60": int(np.sum(dice < 0.60)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def compare_round1_round2(round1_rows, round2_rows, output_dir: Path):
    r1 = {str(row["case_id"]): enrich_volume_metrics(row) for row in round1_rows}
    r2 = {str(row["case_id"]): enrich_volume_metrics(row) for row in round2_rows}
    if len(r1) != EXPECTED_SOURCE_CASES or len(r2) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_CASES} unique held-out cases; Round1={len(r1)} Round2={len(r2)}"
        )
    if set(r1) != set(r2):
        raise RuntimeError("Round-2 held-out IDs differ from Round-1; comparison is not controlled")

    comparison = []
    for case_id in sorted(r1):
        old = r1[case_id]
        new = r2[case_id]
        if int(old["fold"]) != int(new["fold"]):
            raise RuntimeError(f"Case {case_id} moved folds between Round 1 and Round 2")
        comparison.append({
            "case_id": case_id,
            "fold": int(new["fold"]),
            "round1_dice": float(old["dice"]),
            "round2_dice": float(new["dice"]),
            "delta_dice": float(new["dice"]) - float(old["dice"]),
            "round1_precision": float(old["precision"]),
            "round2_precision": float(new["precision"]),
            "delta_precision": float(new["precision"]) - float(old["precision"]),
            "round1_recall": float(old["recall"]),
            "round2_recall": float(new["recall"]),
            "delta_recall": float(new["recall"]) - float(old["recall"]),
            "round1_signed_rve": float(old["signed_rve"]),
            "round2_signed_rve": float(new["signed_rve"]),
            "delta_signed_rve": float(new["signed_rve"]) - float(old["signed_rve"]),
            "round1_fp_vox": float(old["fp_vox"]),
            "round2_fp_vox": float(new["fp_vox"]),
            "delta_fp_vox": float(new["fp_vox"]) - float(old["fp_vox"]),
            "round1_fn_vox": float(old["fn_vox"]),
            "round2_fn_vox": float(new["fn_vox"]),
            "delta_fn_vox": float(new["fn_vox"]) - float(old["fn_vox"]),
            "round1_hd95": float(old["hd95"]),
            "round2_hd95": float(new["hd95"]),
            "delta_hd95": float(new["hd95"]) - float(old["hd95"]),
        })

    s1 = summarize(list(r1.values()))
    s2 = summarize(list(r2.values()))
    dice_delta = np.asarray([x["delta_dice"] for x in comparison], dtype=float)
    summary = {
        "version": "round2_vs_round1_fixed_original47_v1",
        "round1": s1,
        "round2": s2,
        "delta": {
            "mean_dice": s2["mean_dice"] - s1["mean_dice"],
            "mean_precision": s2["mean_precision"] - s1["mean_precision"],
            "mean_recall": s2["mean_recall"] - s1["mean_recall"],
            "median_signed_rve_pct": s2["median_signed_rve_pct"] - s1["median_signed_rve_pct"],
            "median_abs_rve_pct": s2["median_abs_rve_pct"] - s1["median_abs_rve_pct"],
            "overseg_gt_20pct": s2["overseg_gt_20pct"] - s1["overseg_gt_20pct"],
            "overseg_gt_50pct": s2["overseg_gt_50pct"] - s1["overseg_gt_50pct"],
            "underseg_lt_minus20pct": s2["underseg_lt_minus20pct"] - s1["underseg_lt_minus20pct"],
            "mean_fp_vox": s2["mean_fp_vox"] - s1["mean_fp_vox"],
            "mean_fn_vox": s2["mean_fn_vox"] - s1["mean_fn_vox"],
            "median_hd95_mm": s2["median_hd95_mm"] - s1["median_hd95_mm"],
            "dice_lt_0p70": s2["dice_lt_0p70"] - s1["dice_lt_0p70"],
            "dice_lt_0p50": s2["dice_lt_0p50"] - s1["dice_lt_0p50"],
            "dice_ge_0p80": s2["dice_ge_0p80"] - s1["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(dice_delta > 1e-6)),
            "worsened": int(np.sum(dice_delta < -1e-6)),
            "unchanged": int(np.sum(np.abs(dice_delta) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(dice_delta >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(dice_delta <= -0.05)),
        },
        "interpretation": (
            "Paired development comparison on the exact same original 47 held-out cases. "
            "The intended experimental change is addition of audited Round-2 human labels only."
        ),
    }

    case_path = output_dir / "round2_vs_round1_case_comparison.csv"
    summary_path = output_dir / "round2_vs_round1_summary.json"
    write_csv(case_path, comparison)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("ROUND-1 vs ROUND-2 — FIXED ORIGINAL-47 HELD-OUT COMPARISON")
    print(f"Mean Dice:          {s1['mean_dice']:.4f} -> {s2['mean_dice']:.4f} ({summary['delta']['mean_dice']:+.4f})")
    print(f"Precision:          {s1['mean_precision']:.4f} -> {s2['mean_precision']:.4f} ({summary['delta']['mean_precision']:+.4f})")
    print(f"Recall:             {s1['mean_recall']:.4f} -> {s2['mean_recall']:.4f} ({summary['delta']['mean_recall']:+.4f})")
    print(f"Median signed RVE:  {s1['median_signed_rve_pct']:+.2f}% -> {s2['median_signed_rve_pct']:+.2f}%")
    print(f"Median |RVE|:       {s1['median_abs_rve_pct']:.2f}% -> {s2['median_abs_rve_pct']:.2f}%")
    print(f"Overseg >+20%:      {s1['overseg_gt_20pct']} -> {s2['overseg_gt_20pct']}")
    print(f"Overseg >+50%:      {s1['overseg_gt_50pct']} -> {s2['overseg_gt_50pct']}")
    print(f"Underseg <-20%:     {s1['underseg_lt_minus20pct']} -> {s2['underseg_lt_minus20pct']}")
    print(f"Mean FP vox:        {s1['mean_fp_vox']:.1f} -> {s2['mean_fp_vox']:.1f}")
    print(f"Mean FN vox:        {s1['mean_fn_vox']:.1f} -> {s2['mean_fn_vox']:.1f}")
    print(f"Median HD95:        {s1['median_hd95_mm']:.2f} -> {s2['median_hd95_mm']:.2f} mm")
    print(f"Dice <0.70:         {s1['dice_lt_0p70']} -> {s2['dice_lt_0p70']}")
    print(f"Dice <0.50:         {s1['dice_lt_0p50']} -> {s2['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {s1['dice_ge_0p80']} -> {s2['dice_ge_0p80']}")
    print(
        "Case effects:       "
        f"improved={summary['case_effects']['improved']} | worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | <=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print(f"Case comparison:    {case_path}")
    print(f"Summary:            {summary_path}")
    print("=" * 112)
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Controlled Round-2 CV from a passing Round-2 label audit"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", required=True)
    p.add_argument("--fold", default="all", help="0..4 or all")
    p.add_argument("--source-cv-dir", default=str(DEFAULT_SOURCE_CV_DIR))
    p.add_argument("--round1-cv-dir", default=str(DEFAULT_ROUND1_CV_DIR))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    p.add_argument("--eval-threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs < 1:
        p.error("--epochs must be >=1")
    if abs(args.eval_threshold - 0.50) > 1e-8:
        p.error("Controlled Round-2 comparison is frozen at threshold 0.50")

    source_cv_dir = Path(args.source_cv_dir)
    round1_cv_dir = Path(args.round1_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    round1_results_path = round1_cv_dir / "cv_results.csv"
    audit_path = Path(args.audit_metadata)

    source_manifest = read_json(source_manifest_path)
    if len(set(str(x) for x in source_manifest.get("all_case_ids", []))) != EXPECTED_SOURCE_CASES:
        raise RuntimeError("Source manifest is not the frozen original 47-case CV manifest")

    round1_rows = read_csv(round1_results_path)
    if len({str(row["case_id"]) for row in round1_rows}) != EXPECTED_SOURCE_CASES:
        raise RuntimeError("Round-1 results do not contain exactly the original 47 held-out cases")

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata does not record a passing audit")
    if not audit.get("selection_provenance_enforced", False):
        raise RuntimeError(
            "Round-2 audit bypassed the selection-provenance guard. Refusing controlled CV."
        )
    unselected = list(audit.get("unselected_new_label_ids", []))
    if unselected:
        raise RuntimeError(f"Audit contains unselected new labels: {unselected}")

    round1_ids = sorted(str(x) for x in audit.get("round1_human_label_ids", []))
    round2_ids = sorted(str(x) for x in audit.get("round2_new_human_label_ids", []))
    if not round1_ids:
        raise RuntimeError("Audit contains no Round-1 human labels")
    if not round2_ids:
        raise RuntimeError("Audit contains no newly added Round-2 labels")
    all_extra_ids = sorted(set(round1_ids) | set(round2_ids))

    expected_total = EXPECTED_SOURCE_CASES + len(all_extra_ids)
    reported_total = int(audit.get("n_current_valid_human_labels", expected_total))
    if reported_total != expected_total:
        raise RuntimeError(
            f"Audit total mismatch: reported={reported_total}, expected source+human extras={expected_total}"
        )

    base_config = HASSLConfig.from_yaml(args.config)
    if base_config.compute_mode != "prototype":
        raise RuntimeError("Frozen controlled experiments use prototype student + EMA teacher")

    _, source_ids, by_id, current_non_source = discover_round1_cases(base_config, source_manifest_path)
    if sorted(current_non_source) != all_extra_ids:
        raise RuntimeError(
            "Current training label folder differs from the passing Round-2 audit. Re-run audit.\n"
            f"Audited extras={all_extra_ids}\nCurrent extras={sorted(current_non_source)}"
        )
    if len(by_id) != expected_total:
        raise RuntimeError(f"Expected {expected_total} current valid labels, found {len(by_id)}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"experiments/round2_cv_{expected_total}_translation12")
    )

    try:
        selected_folds = cv.parse_fold(args.fold, 5)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))

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
        eval_threshold=0.50,
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=12.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(round1_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    fold_specs = []
    source_id_set = set(str(x) for x in source_ids)
    for original in source_manifest["folds"]:
        fold_idx = int(original["fold"])
        original_train = [str(x) for x in original["train_ids"]]
        original_val = [str(x) for x in original["val_ids"]]
        train_ids = sorted(set(original_train) | set(all_extra_ids))
        val_ids = sorted(original_val)

        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: train/held-out overlap")
        if set(all_extra_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: AL human label leaked into held-out evaluation")
        if set(val_ids) - source_id_set:
            raise RuntimeError(f"Fold {fold_idx}: held-out set contains non-source IDs")

        fold_specs.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "original_train_ids": sorted(original_train),
            "round1_train_only_ids": round1_ids,
            "round2_train_only_ids": round2_ids,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "round2_controlled_cv_from_audit_v1",
        "source_cv_dir": str(source_cv_dir),
        "round1_cv_dir": str(round1_cv_dir),
        "source_manifest": str(source_manifest_path),
        "round1_results": str(round1_results_path),
        "audit_metadata": str(audit_path),
        "n_frozen_source_labels": EXPECTED_SOURCE_CASES,
        "n_round1_human_labels": len(round1_ids),
        "n_round2_new_human_labels": len(round2_ids),
        "n_total_human_labels": expected_total,
        "round1_human_label_ids": round1_ids,
        "round2_new_human_label_ids": round2_ids,
        "folds": fold_specs,
        "recipe": {
            "epochs": int(args.epochs),
            "resize_size": [int(args.resize_size)] * 3,
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": args.eval_source,
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "comparison_rule": (
            "Exact original 47 val_ids are reused. Round-1 and Round-2 human labels are TRAIN ONLY. "
            "No external-validation or pseudo labels are permitted."
        ),
    }
    plan_path = output_dir / "round2_cv_plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise RuntimeError(
                f"Existing plan differs from current provenance: {plan_path}. Use a fresh output directory."
            )
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(x["fold"]): x for x in fold_specs}
    print("=" * 112)
    print("ROUND-2 ACTIVE-LEARNING CONTROLLED CV")
    print(f"Frozen source labels:     {EXPECTED_SOURCE_CASES}")
    print(f"Round-1 human labels:     {len(round1_ids)}")
    print(f"Round-2 new labels:       {len(round2_ids)}")
    print(f"Total human labels:       {expected_total}")
    print(f"Running folds:            {selected_folds}")
    print("Held-out evaluation:      exact original 47 cases / exact original folds")
    print("Recipe:                   unchanged Round-1 recipe | RAW ensemble @0.50")
    print("New reporting:            signed RVE + FP/FN + over/under-segmentation counts")
    print(f"Plan:                      {plan_path}")
    print("=" * 112)

    new_rows = []
    for fold_idx in selected_folds:
        rows = cv.run_fold(runtime_args, fold_map[fold_idx], output_dir)
        for row in rows:
            row["experiment"] = f"round2_{expected_total}labels_translation12"
            row = enrich_volume_metrics(row)
            new_rows.append(row)

    results_path = output_dir / "cv_results.csv"
    existing_rows = []
    if results_path.exists():
        existing_rows = read_csv(results_path)
    selected_set = set(selected_folds)
    existing_rows = [row for row in existing_rows if int(row["fold"]) not in selected_set]
    existing_rows = [enrich_volume_metrics(row) for row in existing_rows]
    merged = existing_rows + new_rows
    merged.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    write_csv(results_path, merged)
    print(f"\nRound-2 results saved: {results_path}")

    completed_folds = {int(row["fold"]) for row in merged}
    if completed_folds == set(range(5)):
        unique_ids = {str(row["case_id"]) for row in merged}
        if len(unique_ids) != EXPECTED_SOURCE_CASES:
            raise RuntimeError(
                f"Completed Round-2 CV must cover exact original 47 cases; found {len(unique_ids)}"
            )
        cv.print_complete_summary(merged, 5)
        compare_round1_round2(round1_rows, merged, output_dir)
    else:
        print(f"Completed folds so far: {sorted(completed_folds)} / [0, 1, 2, 3, 4]")
        print("Run remaining folds; paired Round-1 comparison is emitted after all five complete.")


if __name__ == "__main__":
    main()
