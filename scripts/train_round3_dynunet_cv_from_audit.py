#!/usr/bin/env python3
"""Controlled Final72 DynUNet CV after Round-3 HUMAN_GOLD acquisition.

Evaluation remains the exact original 47 held-out cases in their frozen folds. All human labels
added after the original 47 (Rounds 1, 2, and 3) are TRAIN ONLY in every fold. This preserves a
paired comparison against the frozen Final62 DynUNet A0 CV result while changing only the
training-label set from 62 -> 72.

Frozen recipe: DynUNet, resize128, DiceCE, AdamW 1e-4, dropout0, lambda_unsup0, paired
translation +/-12 vox p=0.8, random init seed42, 100 epochs, raw Student+EMA 50/50 ensemble at
threshold 0.50, no SSL, no LCC, no external31 access.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_active_learning_round2_cv_from_audit as r2
from scripts.audit_round1_labels import discover_round1_cases
from scripts.train_supervised_cv_47_translation import translation_only_transform

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
BASELINE_CV = Path("experiments/round2_cv_62_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
OUTPUT = Path("experiments/round3_cv_72_translation12")
EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 62
EXPECTED_ROUND3 = 10
EXPECTED_TOTAL = 72


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compare_final62_final72(baseline_rows, final72_rows, output_dir: Path):
    old = {str(x["case_id"]): r2.enrich_volume_metrics(x) for x in baseline_rows}
    new = {str(x["case_id"]): r2.enrich_volume_metrics(x) for x in final72_rows}
    if len(old) != EXPECTED_SOURCE or len(new) != EXPECTED_SOURCE or set(old) != set(new):
        raise RuntimeError("Final62/Final72 paired comparison must contain the same exact original47 IDs")

    paired = []
    for case_id in sorted(old):
        a = old[case_id]
        b = new[case_id]
        if int(a["fold"]) != int(b["fold"]):
            raise RuntimeError(f"Fold changed for {case_id}")
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "final62_dice": float(a["dice"]),
            "final72_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "final62_precision": float(a["precision"]),
            "final72_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "final62_recall": float(a["recall"]),
            "final72_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "final62_signed_rve": float(a["signed_rve"]),
            "final72_signed_rve": float(b["signed_rve"]),
            "delta_signed_rve": float(b["signed_rve"]) - float(a["signed_rve"]),
            "final62_hd95": float(a["hd95"]),
            "final72_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
        })

    s0 = r2.summarize(list(old.values()))
    s1 = r2.summarize(list(new.values()))
    delta = np.asarray([x["delta_dice"] for x in paired], dtype=float)
    summary = {
        "version": "final72_vs_final62_fixed_original47_v1",
        "final62": s0,
        "final72": s1,
        "delta": {
            "mean_dice": s1["mean_dice"] - s0["mean_dice"],
            "mean_precision": s1["mean_precision"] - s0["mean_precision"],
            "mean_recall": s1["mean_recall"] - s0["mean_recall"],
            "median_signed_rve_pct": s1["median_signed_rve_pct"] - s0["median_signed_rve_pct"],
            "median_abs_rve_pct": s1["median_abs_rve_pct"] - s0["median_abs_rve_pct"],
            "dice_lt_0p70": s1["dice_lt_0p70"] - s0["dice_lt_0p70"],
            "dice_lt_0p50": s1["dice_lt_0p50"] - s0["dice_lt_0p50"],
            "dice_ge_0p80": s1["dice_ge_0p80"] - s0["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(delta > 1e-6)),
            "worsened": int(np.sum(delta < -1e-6)),
            "unchanged": int(np.sum(np.abs(delta) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(delta >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
        },
        "interpretation": (
            "Paired development comparison on the exact original47 held-out cases. The intended change is only addition of 10 audited Round-3 HUMAN_GOLD labels."
        ),
    }
    r2.write_csv(output_dir / "final72_vs_final62_case_comparison.csv", paired)
    (output_dir / "final72_vs_final62_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("FINAL62 vs FINAL72 — FIXED ORIGINAL47 HELD-OUT COMPARISON")
    print(f"Mean Dice:          {s0['mean_dice']:.4f} -> {s1['mean_dice']:.4f} ({summary['delta']['mean_dice']:+.4f})")
    print(f"Median Dice:        {s0['median_dice']:.4f} -> {s1['median_dice']:.4f}")
    print(f"Precision:          {s0['mean_precision']:.4f} -> {s1['mean_precision']:.4f}")
    print(f"Recall:             {s0['mean_recall']:.4f} -> {s1['mean_recall']:.4f}")
    print(f"Median signed RVE:  {s0['median_signed_rve_pct']:+.2f}% -> {s1['median_signed_rve_pct']:+.2f}%")
    print(f"Median |RVE|:       {s0['median_abs_rve_pct']:.2f}% -> {s1['median_abs_rve_pct']:.2f}%")
    print(f"Dice <0.70:         {s0['dice_lt_0p70']} -> {s1['dice_lt_0p70']}")
    print(f"Dice <0.50:         {s0['dice_lt_0p50']} -> {s1['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {s0['dice_ge_0p80']} -> {s1['dice_ge_0p80']}")
    print(
        "Case effects:       "
        f"improved={summary['case_effects']['improved']} | worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | <=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print("=" * 118)


def main():
    p = argparse.ArgumentParser(description="Controlled Final72 DynUNet CV from passing Round-3 audit")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--baseline-cv-dir", default=str(BASELINE_CV))
    p.add_argument("--output-dir", default=str(OUTPUT))
    p.add_argument("--fold", default="all", help="0..4 or all")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    p.add_argument("--eval-threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs != 100 or args.seed != 42 or args.resize_size != 128:
        p.error("Controlled Final72 locks epochs=100, seed=42, resize-size=128")
    if args.eval_source != "ensemble" or abs(args.eval_threshold - 0.50) > 1e-8:
        p.error("Controlled Final72 locks raw Student+EMA ensemble at threshold 0.50")

    source_cv_dir = Path(args.source_cv_dir)
    baseline_cv_dir = Path(args.baseline_cv_dir)
    output_dir = Path(args.output_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    baseline_results_path = baseline_cv_dir / "cv_results.csv"

    source_manifest = read_json(source_manifest_path)
    if len(set(str(x) for x in source_manifest.get("all_case_ids", []))) != EXPECTED_SOURCE:
        raise RuntimeError("Source CV manifest is not the frozen original47 manifest")
    baseline_rows = r2.read_csv(baseline_results_path)
    if len({str(x["case_id"]) for x in baseline_rows}) != EXPECTED_SOURCE:
        raise RuntimeError("Final62 baseline results do not contain exact original47 held-out cases")

    audit = read_json(Path(args.audit_metadata))
    if not audit.get("all_visible_labels_passed_audit", False) or not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Round-3 training audit must be passing and provenance-enforced")
    prior_ids = sorted(str(x) for x in audit.get("prior_final62_human_label_ids", []))
    round3_ids = sorted(str(x) for x in audit.get("round3_new_human_label_ids", []))
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR or len(round3_ids) != EXPECTED_ROUND3 or len(audited_ids) != EXPECTED_TOTAL:
        raise RuntimeError("Round-3 audit counts are not 62 prior + 10 new = 72")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Frozen controlled recipe requires prototype Student + EMA")
    _, source_ids, by_id, current_non_source = discover_round1_cases(config, source_manifest_path)
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        raise RuntimeError("Current training label folder changed after the passing Round-3 audit")
    if len(by_id) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} valid labels, found {len(by_id)}")

    source_set = set(str(x) for x in source_ids)
    all_extra_ids = sorted(set(current_ids) - source_set)
    if len(all_extra_ids) != EXPECTED_TOTAL - EXPECTED_SOURCE:
        raise RuntimeError("Expected 25 train-only human labels beyond original47")
    if set(round3_ids) - set(all_extra_ids):
        raise RuntimeError("A Round-3 label is not in the train-only extra set")

    selected_folds = cv.parse_fold(args.fold, 5)
    cv.spatial_aug_transform = translation_only_transform
    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=42,
        resize_size=128,
        epochs=100,
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source="ensemble",
        eval_threshold=0.50,
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
        val_ids = sorted(str(x) for x in original["val_ids"])
        train_ids = sorted(set(str(x) for x in original["train_ids"]) | set(all_extra_ids))
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: train/validation overlap")
        if set(all_extra_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: added HUMAN_GOLD leaked into held-out original47")
        fold_specs.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "original_train_ids": sorted(str(x) for x in original["train_ids"]),
            "prior_final62_train_only_ids": sorted(set(prior_ids) - source_set),
            "round3_train_only_ids": round3_ids,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "round3_final72_controlled_cv_v1",
        "source_manifest": str(source_manifest_path),
        "baseline_final62_cv": str(baseline_cv_dir),
        "audit_metadata": str(args.audit_metadata),
        "n_frozen_source": EXPECTED_SOURCE,
        "n_prior_final62": EXPECTED_PRIOR,
        "n_round3_new": EXPECTED_ROUND3,
        "n_total_human_gold": EXPECTED_TOTAL,
        "all_train_only_extra_ids": all_extra_ids,
        "round3_new_human_label_ids": round3_ids,
        "folds": fold_specs,
        "recipe": {
            "architecture": "DynUNet",
            "epochs": 100,
            "resize_size": [128, 128, 128],
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "external31_access": False,
    }
    plan_path = output_dir / "round3_cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing plan differs: {plan_path}; use a fresh output directory")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(x["fold"]): x for x in fold_specs}
    print("=" * 116)
    print("ROUND-3 / FINAL72 DYNUNET CONTROLLED CV")
    print(f"Prior HUMAN_GOLD:       {EXPECTED_PRIOR}")
    print(f"Round-3 HUMAN_GOLD:     {EXPECTED_ROUND3}")
    print(f"Total HUMAN_GOLD:       {EXPECTED_TOTAL}")
    print(f"Train-only extras:      {len(all_extra_ids)}")
    print(f"Running folds:          {selected_folds}")
    print("Held-out evaluation:    exact original47 / exact frozen folds")
    print("Recipe:                 Final62 A0 unchanged; only +10 Round3 HUMAN_GOLD")
    print("External31 access:      NONE")
    print("=" * 116)

    new_rows = []
    for fold_idx in selected_folds:
        rows = cv.run_fold(runtime_args, fold_map[fold_idx], output_dir)
        for row in rows:
            row["experiment"] = "round3_final72_dynunet_translation12"
            new_rows.append(r2.enrich_volume_metrics(row))

    results_path = output_dir / "cv_results.csv"
    existing = r2.read_csv(results_path) if results_path.exists() else []
    selected_set = set(selected_folds)
    existing = [r2.enrich_volume_metrics(x) for x in existing if int(x["fold"]) not in selected_set]
    merged = existing + new_rows
    merged.sort(key=lambda x: (int(x["fold"]), str(x["case_id"])))
    r2.write_csv(results_path, merged)
    print(f"\nFinal72 CV results saved: {results_path}")

    completed = {int(x["fold"]) for x in merged}
    if completed == set(range(5)):
        if len({str(x["case_id"]) for x in merged}) != EXPECTED_SOURCE:
            raise RuntimeError("Completed Final72 CV does not cover exact original47")
        cv.print_complete_summary(merged, 5)
        compare_final62_final72(baseline_rows, merged, output_dir)
    else:
        print(f"Completed folds: {sorted(completed)} / [0, 1, 2, 3, 4]")


if __name__ == "__main__":
    main()
