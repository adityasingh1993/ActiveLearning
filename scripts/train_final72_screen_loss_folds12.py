#!/usr/bin/env python3
"""Fast Final72 loss screening on frozen difficult folds 1 and 2.

The spatial recipe is FIXED to the current screening winner A2:
    paired translation +/-4 voxels, p=0.5, no flips.

Loss variants:
    B1 : Tversky alpha=0.60 (FP), beta=0.40 (FN)
    B2 : Tversky alpha=0.70 (FP), beta=0.30 (FN)
    B3 : Generalized Dice + Focal, lambda_gdl=1.0, lambda_focal=0.25, gamma=2.0

Everything else remains the controlled Final72 DynUNet recipe: audited 72 HUMAN_GOLD,
exact frozen original47 validation folds, resize128, AdamW 1e-4, dropout0,
lambda_unsup0, seed42, 100 epochs, random initialization, raw Student+EMA 50/50
ensemble at threshold 0.50, no LCC, and no external31 access.

This is a SCREENING stage. Candidates are compared to the already-completed A2 DiceCE
screening run on folds 1 and 2. A selected recipe still requires full frozen 5-fold CV.

Examples:
    python scripts/train_final72_screen_loss_folds12.py \
        --config config_resize128.yaml --variant B1 --fold screen

    python scripts/train_final72_screen_loss_folds12.py \
        --config config_resize128.yaml --variant B2 --fold screen

    python scripts/train_final72_screen_loss_folds12.py \
        --config config_resize128.yaml --variant B3 --fold screen
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from monai.transforms import Compose, RandAffined

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_final72_screen_spatial_folds12 as spatial_screen

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
A2_CV = Path("experiments/final72_screen_a2_translation4_p05")

LOSS_VARIANTS = {
    "B1": {
        "name": "tversky_60_40",
        "loss_type": "tversky_60_40",
        "alpha_fp": 0.60,
        "beta_fn": 0.40,
        "lambda_gdl": None,
        "lambda_focal": None,
        "gamma": None,
        "output": Path("experiments/final72_screen_b1_tversky60_40"),
    },
    "B2": {
        "name": "tversky_70_30",
        "loss_type": "tversky_70_30",
        "alpha_fp": 0.70,
        "beta_fn": 0.30,
        "lambda_gdl": None,
        "lambda_focal": None,
        "gamma": None,
        "output": Path("experiments/final72_screen_b2_tversky70_30"),
    },
    "B3": {
        "name": "generalized_dice_focal",
        "loss_type": "generalized_dice_focal",
        "alpha_fp": None,
        "beta_fn": None,
        "lambda_gdl": 1.0,
        "lambda_focal": 0.25,
        "gamma": 2.0,
        "output": Path("experiments/final72_screen_b3_gendice_focal"),
    },
}


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
            writer.writerow({k: row.get(k, "") for k in fields})


def translation4_p05_transform(
    base_transform,
    translate_voxels=4.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """A2 spatial recipe: paired +/-4 voxel translation, p=.5, no flip."""
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        )
    ])


def merge_fold_rows(existing_rows, new_rows, selected_folds):
    """Replace selected folds while preserving any previously completed other folds."""
    keep = [r for r in existing_rows if int(r["fold"]) not in set(selected_folds)]
    merged = keep + list(new_rows)
    return sorted(merged, key=lambda r: (int(r["fold"]), str(r["case_id"])))


def compare_to_a2(candidate_rows, a2_rows, selected_folds, output_dir: Path, variant_id: str):
    candidate = {
        str(r["case_id"]): r
        for r in candidate_rows
        if int(r["fold"]) in selected_folds
    }
    baseline = {
        str(r["case_id"]): r
        for r in a2_rows
        if int(r["fold"]) in selected_folds
    }

    if set(candidate) != set(baseline):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_a2 = sorted(set(candidate) - set(baseline))
        raise RuntimeError(
            "Loss screening and A2 must contain the exact same held-out IDs.\n"
            f"Missing candidate: {missing_candidate}\n"
            f"Missing A2: {missing_a2}"
        )

    paired = []
    for case_id in sorted(candidate):
        a = baseline[case_id]
        b = candidate[case_id]
        if int(a["fold"]) != int(b["fold"]):
            raise RuntimeError(f"Fold mismatch for {case_id}")
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "a2_dice": float(a["dice"]),
            "candidate_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a2_precision": float(a["precision"]),
            "candidate_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a2_recall": float(a["recall"]),
            "candidate_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a2_rve": float(a["rve"]),
            "candidate_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
            "a2_hd95": float(a["hd95"]),
            "candidate_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
        })

    write_csv(output_dir / "screening_vs_a2_case_comparison.csv", paired)

    fold_summaries = []
    for fold in selected_folds:
        subset = [x for x in paired if int(x["fold"]) == fold]
        if not subset:
            continue
        a2_d = np.asarray([x["a2_dice"] for x in subset], dtype=float)
        cand_d = np.asarray([x["candidate_dice"] for x in subset], dtype=float)
        fold_summaries.append({
            "fold": fold,
            "n": len(subset),
            "a2_mean_dice": float(np.mean(a2_d)),
            "candidate_mean_dice": float(np.mean(cand_d)),
            "delta_mean_dice": float(np.mean(cand_d) - np.mean(a2_d)),
            "a2_mean_precision": float(np.mean([x["a2_precision"] for x in subset])),
            "candidate_mean_precision": float(np.mean([x["candidate_precision"] for x in subset])),
            "delta_mean_precision": float(np.mean([x["delta_precision"] for x in subset])),
            "a2_mean_recall": float(np.mean([x["a2_recall"] for x in subset])),
            "candidate_mean_recall": float(np.mean([x["candidate_recall"] for x in subset])),
            "delta_mean_recall": float(np.mean([x["delta_recall"] for x in subset])),
            "improved_cases": int(sum(x["delta_dice"] > 1e-6 for x in subset)),
            "worsened_cases": int(sum(x["delta_dice"] < -1e-6 for x in subset)),
            "improved_ge_0p05": int(sum(x["delta_dice"] >= 0.05 for x in subset)),
            "worsened_le_minus_0p05": int(sum(x["delta_dice"] <= -0.05 for x in subset)),
        })

    a2_d = np.asarray([x["a2_dice"] for x in paired], dtype=float)
    cand_d = np.asarray([x["candidate_dice"] for x in paired], dtype=float)
    overall = {
        "version": "final72_loss_screen_folds12_v1",
        "variant": variant_id,
        "reference": "A2 = translation +/-4 vox p=.5 + DiceCE",
        "screening_folds": selected_folds,
        "n_cases": len(paired),
        "a2_mean_dice": float(np.mean(a2_d)),
        "candidate_mean_dice": float(np.mean(cand_d)),
        "delta_mean_dice": float(np.mean(cand_d) - np.mean(a2_d)),
        "a2_median_dice": float(np.median(a2_d)),
        "candidate_median_dice": float(np.median(cand_d)),
        "a2_mean_precision": float(np.mean([x["a2_precision"] for x in paired])),
        "candidate_mean_precision": float(np.mean([x["candidate_precision"] for x in paired])),
        "a2_mean_recall": float(np.mean([x["a2_recall"] for x in paired])),
        "candidate_mean_recall": float(np.mean([x["candidate_recall"] for x in paired])),
        "improved_cases": int(np.sum(cand_d > a2_d + 1e-6)),
        "worsened_cases": int(np.sum(cand_d < a2_d - 1e-6)),
        "improved_ge_0p05": int(np.sum((cand_d - a2_d) >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum((cand_d - a2_d) <= -0.05)),
        "folds": fold_summaries,
        "screening_only": True,
        "note": "A winner still requires confirmation on all five frozen folds.",
    }
    (output_dir / "screening_vs_a2_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print(f"{variant_id} vs A2 DICECE — LOSS SCREENING")
    for s in fold_summaries:
        print(
            f"Fold {s['fold']}: A2={s['a2_mean_dice']:.4f} -> candidate={s['candidate_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | PrecΔ={s['delta_mean_precision']:+.4f} | "
            f"RecΔ={s['delta_mean_recall']:+.4f} | improved={s['improved_cases']} | "
            f"worsened={s['worsened_cases']}"
        )
    print(
        f"Combined: A2={overall['a2_mean_dice']:.4f} -> candidate={overall['candidate_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f})"
    )
    print(
        f"Precision: {overall['a2_mean_precision']:.4f} -> {overall['candidate_mean_precision']:.4f} | "
        f"Recall: {overall['a2_mean_recall']:.4f} -> {overall['candidate_mean_recall']:.4f}"
    )
    print(
        f"Case effects: improved={overall['improved_cases']} | worsened={overall['worsened_cases']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: do not promote without full frozen 5-fold confirmation.")
    print("=" * 120)
    return overall


def main():
    p = argparse.ArgumentParser(description="Final72 loss screening on frozen folds 1 and 2")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True, choices=sorted(LOSS_VARIANTS))
    p.add_argument("--fold", default="screen", help="screen (= folds 1+2), all, or 0..4")
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--a2-cv-dir", default=str(A2_CV))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    p.add_argument("--eval-threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs != 100 or args.seed != 42 or args.resize_size != 128:
        p.error("Controlled screening locks epochs=100, seed=42, resize-size=128")
    if args.eval_source != "ensemble" or abs(args.eval_threshold - 0.50) > 1e-8:
        p.error("Controlled screening locks raw Student+EMA ensemble @ threshold 0.50")

    variant = dict(LOSS_VARIANTS[args.variant])
    selected_folds = spatial_screen.parse_screen_fold(args.fold)
    output_dir = Path(args.output_dir) if args.output_dir else Path(variant["output"])
    source_cv_dir = Path(args.source_cv_dir)
    a2_cv_dir = Path(args.a2_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    a2_results_path = a2_cv_dir / "cv_results.csv"

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Controlled Final72 screening requires prototype Student+EMA mode")
    if int(config.num_classes) != 1:
        raise RuntimeError("This loss screen is locked to the binary bladder task (num_classes=1)")

    _, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    a2_rows = read_csv(a2_results_path)
    a2_selected = [r for r in a2_rows if int(r["fold"]) in selected_folds]
    expected_val_ids = sorted(
        str(case_id)
        for fold in selected_folds
        for case_id in fold_map[fold]["val_ids"]
    )
    actual_a2_ids = sorted(str(r["case_id"]) for r in a2_selected)
    if actual_a2_ids != expected_val_ids:
        raise RuntimeError(
            "A2 reference does not contain the exact selected frozen validation cases. "
            "Complete A2 folds 1 and 2 before running loss screening."
        )

    # Patch the shared CV baseline setter so cv.run_fold keeps every baseline setting but
    # changes ONLY the supervised loss. Restore it in finally to avoid process-wide leakage.
    original_apply_baseline = cv.apply_baseline

    def apply_loss_screen_baseline(cfg, resize_size, epochs):
        original_apply_baseline(cfg, resize_size, epochs)
        cfg.loss_type = variant["loss_type"]
        cfg.include_boundary = False
        if variant["loss_type"] == "generalized_dice_focal":
            cfg.loss_lambda_gdl = float(variant["lambda_gdl"])
            cfg.loss_lambda_focal = float(variant["lambda_focal"])
            cfg.loss_focal_gamma = float(variant["gamma"])

    cv.apply_baseline = apply_loss_screen_baseline
    cv.spatial_aug_transform = translation4_p05_transform

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
        translate_voxels=4.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(a2_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    plan = {
        "version": "final72_loss_screen_folds12_v1",
        "variant": args.variant,
        "variant_name": variant["name"],
        "screening_folds": selected_folds,
        "dataset": {
            "human_gold": 72,
            "frozen_source_validation_cohort": 47,
            "train_only_extra_human_gold": len(extra_ids),
            "source_manifest": str(source_manifest_path),
            "audit_metadata": str(args.audit_metadata),
        },
        "reference": {
            "variant": "A2",
            "results": str(a2_results_path),
            "spatial": "translation +/-4 vox p=.5, no flip",
            "loss": "dice_ce",
        },
        "candidate_recipe": {
            "architecture": "DynUNet",
            "resize_size": [128, 128, 128],
            "epochs": 100,
            "seed": 42,
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "flip": False,
            "loss_type": variant["loss_type"],
            "tversky_alpha_fp": variant["alpha_fp"],
            "tversky_beta_fn": variant["beta_fn"],
            "lambda_gdl": variant["lambda_gdl"],
            "lambda_focal": variant["lambda_focal"],
            "focal_gamma": variant["gamma"],
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "external31_access": False,
        "screening_only": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "screening_plan.json"
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise RuntimeError(f"Existing plan differs at {plan_path}; use a fresh output directory")
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 120)
    print("FINAL72 LOSS SCREENING — FROZEN FOLDS 1/2")
    print(f"Variant:                 {args.variant} / {variant['name']}")
    print(f"Running folds:           {selected_folds}")
    print("Spatial recipe:          A2 = translation +/-4 vox p=.5, no flip")
    if variant["loss_type"].startswith("tversky"):
        print(
            f"Loss:                    Tversky alpha(FP)={variant['alpha_fp']:.2f}, "
            f"beta(FN)={variant['beta_fn']:.2f}"
        )
    else:
        print(
            f"Loss:                    GeneralizedDiceFocal gdl={variant['lambda_gdl']:.2f}, "
            f"focal={variant['lambda_focal']:.2f}, gamma={variant['gamma']:.1f}"
        )
    print("Validation:              exact frozen original47 held-out IDs")
    print("Prediction:              raw Student+EMA 50/50 @ .50")
    print("External31 access:       NO")
    print("=" * 120)

    new_rows = []
    try:
        for fold in selected_folds:
            rows = cv.run_fold(runtime_args, fold_map[fold], output_dir)
            for row in rows:
                row["experiment"] = args.variant
                row["loss_type"] = variant["loss_type"]
            new_rows.extend(rows)
    finally:
        cv.apply_baseline = original_apply_baseline

    results_path = output_dir / "cv_results.csv"
    existing_rows = read_csv(results_path) if results_path.exists() else []
    merged_rows = merge_fold_rows(existing_rows, new_rows, selected_folds)
    write_csv(results_path, merged_rows)

    compare_to_a2(merged_rows, a2_rows, selected_folds, output_dir, args.variant)

    print(f"\nResults:    {results_path}")
    print(f"Plan:       {plan_path}")
    print(f"Comparison: {output_dir / 'screening_vs_a2_case_comparison.csv'}")
    print(f"Summary:    {output_dir / 'screening_vs_a2_summary.json'}")


if __name__ == "__main__":
    main()
