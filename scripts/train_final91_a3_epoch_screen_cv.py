#!/usr/bin/env python3
"""Screen a longer Final91 A3 training horizon using internal original47 CV only.

This preserves the exact Final91 91-label dataset, frozen folds, A3 augmentation,
constant AdamW LR=1e-4, Student+EMA evaluation, threshold 0.50, and raw/no-LCC
post-processing. The only intended change from the completed Final91 baseline is the
maximum epoch count. External31 is never read by this script.
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
import scripts.train_final72_screen_spatial_folds12 as spatial
import scripts.train_round5_final91_a3_cv as final91
from scripts.audit_round1_labels import discover_round1_cases

BASELINE_CV = Path("experiments/round5_cv_91_a3")
SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
EXPECTED_SOURCE = 47
EXPECTED_TOTAL = 91


def compare_epoch_screen(baseline_rows, candidate_rows, output_dir: Path, epochs: int):
    old = {str(row["case_id"]): r2.enrich_volume_metrics(row) for row in baseline_rows}
    new = {str(row["case_id"]): r2.enrich_volume_metrics(row) for row in candidate_rows}
    if len(old) != EXPECTED_SOURCE or set(old) != set(new):
        raise RuntimeError("Epoch comparison requires identical frozen original47 cases")
    paired = []
    for case_id in sorted(old):
        before, after = old[case_id], new[case_id]
        if int(before["fold"]) != int(after["fold"]):
            raise RuntimeError(f"Frozen fold changed for {case_id}")
        paired.append({
            "case_id": case_id,
            "fold": int(after["fold"]),
            "epoch100_dice": float(before["dice"]),
            f"epoch{epochs}_dice": float(after["dice"]),
            "delta_dice": float(after["dice"]) - float(before["dice"]),
            "epoch100_precision": float(before["precision"]),
            f"epoch{epochs}_precision": float(after["precision"]),
            "delta_precision": float(after["precision"]) - float(before["precision"]),
            "epoch100_recall": float(before["recall"]),
            f"epoch{epochs}_recall": float(after["recall"]),
            "delta_recall": float(after["recall"]) - float(before["recall"]),
            "epoch100_signed_rve": float(before["signed_rve"]),
            f"epoch{epochs}_signed_rve": float(after["signed_rve"]),
            "epoch100_hd95": float(before["hd95"]),
            f"epoch{epochs}_hd95": float(after["hd95"]),
        })
    s0 = r2.summarize(list(old.values()))
    s1 = r2.summarize(list(new.values()))
    delta = np.asarray([row["delta_dice"] for row in paired], dtype=float)
    folds = []
    for fold in range(5):
        subset = [row for row in paired if int(row["fold"]) == fold]
        folds.append({
            "fold": fold,
            "n": len(subset),
            "epoch100_mean_dice": float(np.mean([row["epoch100_dice"] for row in subset])),
            f"epoch{epochs}_mean_dice": float(np.mean([row[f"epoch{epochs}_dice"] for row in subset])),
            "delta_mean_dice": float(np.mean([row["delta_dice"] for row in subset])),
        })
    summary = {
        "version": "final91_a3_epoch_screen_v1",
        "baseline_epochs": 100,
        "candidate_max_epochs": epochs,
        "epoch100": s0,
        f"epoch{epochs}": s1,
        "delta": {
            "mean_dice": s1["mean_dice"] - s0["mean_dice"],
            "mean_precision": s1["mean_precision"] - s0["mean_precision"],
            "mean_recall": s1["mean_recall"] - s0["mean_recall"],
            "median_abs_rve_pct": s1["median_abs_rve_pct"] - s0["median_abs_rve_pct"],
            "dice_lt_0p70": s1["dice_lt_0p70"] - s0["dice_lt_0p70"],
        },
        "case_effects": {
            "improved": int(np.sum(delta > 1e-6)),
            "worsened": int(np.sum(delta < -1e-6)),
            "improved_ge_0p05": int(np.sum(delta >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
        },
        "folds": folds,
    }
    r2.write_csv(output_dir / f"final91_a3_e100_vs_e{epochs}_case_comparison.csv", paired)
    r2.write_csv(output_dir / f"final91_a3_e100_vs_e{epochs}_fold_summary.csv", folds)
    (output_dir / f"final91_a3_e100_vs_e{epochs}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 118)
    print(f"FINAL91 A3 — 100 vs {epochs} MAX EPOCHS — FIXED ORIGINAL47")
    print(f"Mean Dice:  {s0['mean_dice']:.4f} -> {s1['mean_dice']:.4f} ({summary['delta']['mean_dice']:+.4f})")
    print(f"Precision:  {s0['mean_precision']:.4f} -> {s1['mean_precision']:.4f} ({summary['delta']['mean_precision']:+.4f})")
    print(f"Recall:     {s0['mean_recall']:.4f} -> {s1['mean_recall']:.4f} ({summary['delta']['mean_recall']:+.4f})")
    print(f"Dice <.70:  {s0['dice_lt_0p70']} -> {s1['dice_lt_0p70']}")
    for row in folds:
        print(
            f"Fold {row['fold']}:     {row['epoch100_mean_dice']:.4f} -> "
            f"{row[f'epoch{epochs}_mean_dice']:.4f} ({row['delta_mean_dice']:+.4f})"
        )
    print("External31 was not accessed.")
    print("=" * 118)


def main():
    parser = argparse.ArgumentParser(description="Final91 A3 internal-CV longer-epoch screen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fold", default="all", help="0..4 or all")
    parser.add_argument("--baseline-cv-dir", default=str(BASELINE_CV))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 100 or args.epochs > 300:
        parser.error("Controlled screen requires 101 <= --epochs <= 300")

    baseline_dir = Path(args.baseline_cv_dir)
    source_manifest_path = Path(args.source_cv_dir) / "cv_splits.json"
    baseline_rows = r2.read_csv(baseline_dir / "cv_results.csv")
    if len({str(row["case_id"]) for row in baseline_rows}) != EXPECTED_SOURCE:
        raise RuntimeError("Final91 100-epoch baseline must contain exact original47 held-out cases")
    if {int(row["fold"]) for row in baseline_rows} != set(range(5)):
        raise RuntimeError("Final91 baseline must contain all five folds")

    source_manifest = final91.read_json(source_manifest_path)
    source_ids = sorted(str(case_id) for case_id in source_manifest.get("all_case_ids", []))
    if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
        raise RuntimeError("Source manifest is not frozen original47")
    audit = final91.read_json(Path(args.audit_metadata))
    if not audit.get("all_visible_labels_passed_audit", False) or not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Final91 audit must be passing and provenance-enforced")
    audited_ids = sorted(str(case_id) for case_id in audit.get("all_current_human_label_ids", []))
    if len(audited_ids) != EXPECTED_TOTAL:
        raise RuntimeError("Final91 audit must contain exactly 91 HUMAN_GOLD IDs")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Final91 A3 requires prototype DynUNet Student+EMA")
    _, frozen_source_ids, by_id, _ = discover_round1_cases(config, source_manifest_path)
    if sorted(str(case_id) for case_id in by_id) != audited_ids:
        raise RuntimeError("Live label directory changed after Final91 audit")
    extra_ids = sorted(set(audited_ids) - set(str(case_id) for case_id in frozen_source_ids))
    if len(extra_ids) != EXPECTED_TOTAL - EXPECTED_SOURCE:
        raise RuntimeError("Expected exactly 44 train-only cases beyond original47")

    selected_folds = cv.parse_fold(args.fold, 5)
    fold_specs = []
    for original in source_manifest["folds"]:
        fold = int(original["fold"])
        val_ids = sorted(str(case_id) for case_id in original["val_ids"])
        train_ids = sorted(set(str(case_id) for case_id in original["train_ids"]) | set(extra_ids))
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold}: train/validation overlap")
        fold_specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/final91_a3_epoch_screen_e{args.epochs}")
    spatial.ACTIVE_VARIANT = dict(final91.A3_VARIANT)
    cv.spatial_aug_transform = spatial.screening_spatial_transform
    runtime_args = SimpleNamespace(
        config=args.config, fold=args.fold, folds=5, seed=42, resize_size=128, epochs=args.epochs,
        output_dir=str(output_dir), split_manifest=str(source_manifest_path), eval_source="ensemble",
        eval_threshold=0.50, overwrite=bool(args.overwrite), regenerate_splits=False, spatial_aug=True,
        translate_voxels=4.0, rotate_degrees=0.0, scale_fraction=0.0,
        baseline_results=str(baseline_dir / "cv_results.csv"),
    )
    cv.install_cv_loader_hook(runtime_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "final91_a3_epoch_screen_plan_v1",
        "baseline_final91_cv": str(baseline_dir),
        "audit_metadata": str(args.audit_metadata),
        "source_manifest": str(source_manifest_path),
        "n_total_human_gold": EXPECTED_TOTAL,
        "n_frozen_source": EXPECTED_SOURCE,
        "n_train_only_extra": len(extra_ids),
        "max_epochs": args.epochs,
        "early_stopping": {
            "enabled_from_config": bool(getattr(config, "use_early_stopping", True)),
            "patience": int(getattr(config, "early_stopping_patience", 30)),
            "min_delta": float(getattr(config, "early_stopping_min_delta", 1e-4)),
        },
        "recipe": {
            "architecture": "DynUNet", "resize_size": [128, 128, 128], "loss": "dice_ce",
            "learning_rate": 1e-4, "lr_scheduler": "none", "dropout": 0.0, "lambda_unsup": 0.0,
            "translation_voxels": 4.0, "translation_probability": 0.5,
            "lr_flip": True, "lr_flip_probability": 0.5,
            "eval_source": "ensemble", "eval_threshold": 0.50, "postprocessing": "raw_no_lcc",
        },
        "only_intended_change_from_baseline": "maximum_epochs_100_to_requested_value",
        "external31_access": False,
    }
    plan_path = output_dir / "final91_a3_epoch_screen_plan.json"
    if plan_path.exists() and final91.read_json(plan_path) != plan:
        raise RuntimeError(f"Existing plan differs: {plan_path}; use a fresh output directory")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(spec["fold"]): spec for spec in fold_specs}
    print("=" * 118)
    print(f"FINAL91 A3 EPOCH SCREEN — 100 vs MAX {args.epochs}")
    print(f"Running folds:       {selected_folds}")
    print(f"HUMAN_GOLD:          {EXPECTED_TOTAL} (unchanged)")
    print("Held-out validation: frozen original47")
    print("Only change:         maximum training epochs")
    print("External31:          NOT ACCESSED")
    print("=" * 118)
    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    kept = [row for row in existing if int(row["fold"]) not in selected_folds]
    combined = kept + new_rows
    combined.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    cv.write_results(results_path, combined)
    completed = {int(row["fold"]) for row in combined}
    if completed == set(range(5)) and len({str(row["case_id"]) for row in combined}) == EXPECTED_SOURCE:
        compare_epoch_screen(baseline_rows, combined, output_dir, args.epochs)
    else:
        print(f"Partial screen complete: folds {sorted(completed)}; comparison waits for all five folds")
    print(f"Results: {results_path}")
    print(f"Plan:    {plan_path}")


if __name__ == "__main__":
    main()
