#!/usr/bin/env python3
"""Fast Final72 spatial-augmentation screening on the difficult frozen folds.

Purpose
-------
Screen candidate spatial recipes on folds 1 and 2 before spending GPU time on full 5-fold CV.
The training set remains the audited Final72 HUMAN_GOLD set and each fold keeps the exact frozen
original47 held-out validation IDs. Added HUMAN_GOLD cases remain TRAIN ONLY.

Variants
--------
A1 : no spatial augmentation
A2 : paired translation +/-4 voxels, p=0.5
A3 : paired translation +/-4 voxels, p=0.5 + left/right flip, p=0.5

Everything else stays locked to the Final72 DynUNet recipe: resize128, DiceCE, AdamW 1e-4,
dropout0, lambda_unsup0, seed42, 100 epochs, random initialization, raw Student+EMA 50/50
ensemble at threshold 0.50, no LCC, no external31 access.

The default --fold screen runs folds 1 and 2. This is a hypothesis-screening stage only; the
winning recipe must still be confirmed on all five frozen folds before making a final claim.

Examples
--------
  python scripts/train_final72_screen_spatial_folds12.py \
    --config config_resize128.yaml --variant A1 --fold screen

  python scripts/train_final72_screen_spatial_folds12.py \
    --config config_resize128.yaml --variant A2 --fold screen

  python scripts/train_final72_screen_spatial_folds12.py \
    --config config_resize128.yaml --variant A3 --fold screen
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from monai.transforms import Compose, RandAffined, RandFlipd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_active_learning_round2_cv_from_audit as r2
from scripts.audit_round1_labels import discover_round1_cases

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
BASELINE_CV = Path("experiments/round3_cv_72_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")

EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 62
EXPECTED_ROUND3 = 10
EXPECTED_TOTAL = 72

VARIANTS = {
    "A1": {
        "name": "no_translation",
        "spatial_aug": False,
        "translate_voxels": 0.0,
        "translate_prob": 0.0,
        "lr_flip": False,
        "lr_flip_prob": 0.0,
        "output": Path("experiments/final72_screen_a1_no_translation"),
    },
    "A2": {
        "name": "translation4_p05",
        "spatial_aug": True,
        "translate_voxels": 4.0,
        "translate_prob": 0.5,
        "lr_flip": False,
        "lr_flip_prob": 0.0,
        "output": Path("experiments/final72_screen_a2_translation4_p05"),
    },
    "A3": {
        "name": "translation4_p05_lrflip_p05",
        "spatial_aug": True,
        "translate_voxels": 4.0,
        "translate_prob": 0.5,
        "lr_flip": True,
        "lr_flip_prob": 0.5,
        "output": Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05"),
    },
}

ACTIVE_VARIANT = None


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_screen_fold(value: str):
    value = str(value).strip().lower()
    if value in {"screen", "1,2", "1+2"}:
        return [1, 2]
    if value == "all":
        return list(range(5))
    idx = int(value)
    if not 0 <= idx <= 4:
        raise ValueError("--fold must be screen, all, or 0..4")
    return [idx]


def screening_spatial_transform(
    base_transform,
    translate_voxels=0.0,
    rotate_degrees=0.0,
    scale_fraction=0.0,
):
    """Append only the spatial operations defined by ACTIVE_VARIANT.

    The base transform already performs deterministic channel/orientation/spacing/intensity/
    resize preprocessing. Therefore spatial_axis=0 here is the first RAS spatial axis (LR).
    """
    if ACTIVE_VARIANT is None:
        raise RuntimeError("ACTIVE_VARIANT was not initialized")

    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    steps = list(base_steps)

    if ACTIVE_VARIANT["lr_flip"]:
        steps.append(
            RandFlipd(
                keys=["image", "label"],
                prob=float(ACTIVE_VARIANT["lr_flip_prob"]),
                spatial_axis=0,
            )
        )

    translate = float(ACTIVE_VARIANT["translate_voxels"])
    if translate > 0:
        steps.append(
            RandAffined(
                keys=["image", "label"],
                prob=float(ACTIVE_VARIANT["translate_prob"]),
                rotate_range=(0.0, 0.0, 0.0),
                translate_range=(translate, translate, translate),
                scale_range=(0.0, 0.0, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    return Compose(steps)


def build_final72_fold_specs(config, source_manifest_path: Path, audit_path: Path):
    source_manifest = read_json(source_manifest_path)
    source_ids_manifest = sorted(str(x) for x in source_manifest.get("all_case_ids", []))
    if len(set(source_ids_manifest)) != EXPECTED_SOURCE:
        raise RuntimeError("Source CV manifest is not the frozen original47 manifest")

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-3 audit is not passing")
    if not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Round-3 audit does not enforce selection provenance")

    prior_ids = sorted(str(x) for x in audit.get("prior_final62_human_label_ids", []))
    round3_ids = sorted(str(x) for x in audit.get("round3_new_human_label_ids", []))
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR:
        raise RuntimeError(f"Expected {EXPECTED_PRIOR} prior Final62 labels, found {len(prior_ids)}")
    if len(round3_ids) != EXPECTED_ROUND3:
        raise RuntimeError(f"Expected {EXPECTED_ROUND3} Round3 labels, found {len(round3_ids)}")
    if len(audited_ids) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} Final72 labels, found {len(audited_ids)}")

    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest_path)
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        raise RuntimeError(
            "Current training-label discovery differs from the passing Final72 audit. "
            "Refusing to run a screening experiment on a changed dataset."
        )

    source_set = set(str(x) for x in source_ids)
    extra_ids = sorted(set(current_ids) - source_set)
    if len(extra_ids) != EXPECTED_TOTAL - EXPECTED_SOURCE:
        raise RuntimeError("Expected exactly 25 train-only HUMAN_GOLD cases beyond original47")
    if set(round3_ids) - set(extra_ids):
        raise RuntimeError("A Round3 label is not in the train-only extra set")

    fold_specs = []
    for original in source_manifest.get("folds", []):
        fold = int(original["fold"])
        val_ids = sorted(str(x) for x in original["val_ids"])
        train_ids = sorted(set(str(x) for x in original["train_ids"]) | set(extra_ids))
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold}: train/validation overlap")
        if set(extra_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold}: train-only HUMAN_GOLD leaked into validation")
        fold_specs.append({
            "fold": fold,
            "train_ids": train_ids,
            "val_ids": val_ids,
        })

    if len(fold_specs) != 5:
        raise RuntimeError(f"Expected 5 frozen folds, found {len(fold_specs)}")
    return source_manifest, extra_ids, fold_specs


def compare_to_a0(rows, baseline_rows, selected_folds, output_dir: Path, variant_id: str):
    candidate = {
        str(r["case_id"]): r
        for r in rows
        if int(r["fold"]) in selected_folds
    }
    baseline = {
        str(r["case_id"]): r
        for r in baseline_rows
        if int(r["fold"]) in selected_folds
    }
    common = sorted(set(candidate) & set(baseline))
    if not common:
        raise RuntimeError("No common held-out cases between screening result and Final72 A0 baseline")

    paired = []
    for case_id in common:
        a = baseline[case_id]
        b = candidate[case_id]
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "a0_dice": float(a["dice"]),
            "candidate_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a0_precision": float(a["precision"]),
            "candidate_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a0_recall": float(a["recall"]),
            "candidate_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a0_rve": float(a["rve"]),
            "candidate_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
            "a0_hd95": float(a["hd95"]),
            "candidate_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
        })

    write_csv(output_dir / "screening_vs_a0_case_comparison.csv", paired)

    summaries = []
    for fold in selected_folds:
        fold_rows = [x for x in paired if int(x["fold"]) == fold]
        if not fold_rows:
            continue
        base_d = np.asarray([x["a0_dice"] for x in fold_rows], dtype=float)
        cand_d = np.asarray([x["candidate_dice"] for x in fold_rows], dtype=float)
        summaries.append({
            "fold": fold,
            "n": len(fold_rows),
            "a0_mean_dice": float(np.mean(base_d)),
            "candidate_mean_dice": float(np.mean(cand_d)),
            "delta_mean_dice": float(np.mean(cand_d) - np.mean(base_d)),
            "a0_mean_precision": float(np.mean([x["a0_precision"] for x in fold_rows])),
            "candidate_mean_precision": float(np.mean([x["candidate_precision"] for x in fold_rows])),
            "a0_mean_recall": float(np.mean([x["a0_recall"] for x in fold_rows])),
            "candidate_mean_recall": float(np.mean([x["candidate_recall"] for x in fold_rows])),
            "improved_cases": int(sum(x["delta_dice"] > 1e-6 for x in fold_rows)),
            "worsened_cases": int(sum(x["delta_dice"] < -1e-6 for x in fold_rows)),
        })

    all_base = np.asarray([x["a0_dice"] for x in paired], dtype=float)
    all_cand = np.asarray([x["candidate_dice"] for x in paired], dtype=float)
    overall = {
        "variant": variant_id,
        "screening_folds": selected_folds,
        "n_cases": len(paired),
        "a0_mean_dice": float(np.mean(all_base)),
        "candidate_mean_dice": float(np.mean(all_cand)),
        "delta_mean_dice": float(np.mean(all_cand) - np.mean(all_base)),
        "a0_median_dice": float(np.median(all_base)),
        "candidate_median_dice": float(np.median(all_cand)),
        "improved_cases": int(np.sum(all_cand > all_base + 1e-6)),
        "worsened_cases": int(np.sum(all_cand < all_base - 1e-6)),
        "improved_ge_0p05": int(np.sum((all_cand - all_base) >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum((all_cand - all_base) <= -0.05)),
        "folds": summaries,
        "screening_only": True,
        "note": "Winner must be confirmed on all five frozen folds before final model selection.",
    }
    (output_dir / "screening_vs_a0_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 116)
    print(f"{variant_id} vs FINAL72 A0 — SCREENING COMPARISON")
    for s in summaries:
        print(
            f"Fold {s['fold']}: A0={s['a0_mean_dice']:.4f} -> candidate={s['candidate_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | improved={s['improved_cases']} | worsened={s['worsened_cases']}"
        )
    print(
        f"Combined: A0={overall['a0_mean_dice']:.4f} -> candidate={overall['candidate_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f})"
    )
    print(
        f"Case effects: improved={overall['improved_cases']} | worsened={overall['worsened_cases']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: do not promote a winner without full frozen 5-fold confirmation.")
    print("=" * 116)


def main():
    global ACTIVE_VARIANT

    p = argparse.ArgumentParser(description="Final72 fast spatial screening on frozen folds 1 and 2")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--fold", default="screen", help="screen (=1+2), all, or 0..4")
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--baseline-cv-dir", default=str(BASELINE_CV))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    variant = dict(VARIANTS[args.variant])
    ACTIVE_VARIANT = variant
    selected_folds = parse_screen_fold(args.fold)

    source_cv_dir = Path(args.source_cv_dir)
    baseline_cv_dir = Path(args.baseline_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    baseline_results_path = baseline_cv_dir / "cv_results.csv"
    output_dir = Path(args.output_dir) if args.output_dir else variant["output"]

    if not baseline_results_path.exists():
        raise FileNotFoundError(f"Missing Final72 A0 baseline results: {baseline_results_path}")
    baseline_rows = r2.read_csv(baseline_results_path)
    if len({str(x["case_id"]) for x in baseline_rows}) != EXPECTED_SOURCE:
        raise RuntimeError("Final72 A0 baseline must contain the exact 47 held-out source cases")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Screening requires prototype Student + EMA mode")

    source_manifest, extra_ids, fold_specs = build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    # Reuse the exact CV data-loader machinery; only the candidate training transform changes.
    cv.spatial_aug_transform = screening_spatial_transform
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
        spatial_aug=bool(variant["spatial_aug"]),
        translate_voxels=float(variant["translate_voxels"]),
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(baseline_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "final72_spatial_screen_folds12_v1",
        "variant": args.variant,
        "variant_name": variant["name"],
        "purpose": "fast hypothesis screening before full frozen 5-fold confirmation",
        "default_screening_folds": [1, 2],
        "source_manifest": str(source_manifest_path),
        "baseline_final72_a0": str(baseline_cv_dir),
        "audit_metadata": str(args.audit_metadata),
        "n_frozen_source": EXPECTED_SOURCE,
        "n_total_human_gold": EXPECTED_TOTAL,
        "n_train_only_extra": len(extra_ids),
        "recipe": {
            "architecture": "DynUNet",
            "resize_size": [128, 128, 128],
            "epochs": 100,
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "translation_voxels": variant["translate_voxels"],
            "translation_probability": variant["translate_prob"],
            "lr_flip": variant["lr_flip"],
            "lr_flip_probability": variant["lr_flip_prob"],
            "lr_flip_axis_after_ras": 0,
            "rotation": "off",
            "scale": "off",
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "external31_access": False,
    }
    plan_path = output_dir / "screening_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing screening plan differs: {plan_path}; use a fresh output directory")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 116)
    print("FINAL72 SPATIAL SCREEN — FROZEN OOF FOLDS")
    print(f"Variant:                {args.variant} / {variant['name']}")
    print(f"Running folds:          {selected_folds}")
    print(f"Final72 HUMAN_GOLD:     {EXPECTED_TOTAL}")
    print(f"Train-only extras:      {len(extra_ids)}")
    print("Validation:             exact frozen original47 held-out IDs")
    print("Loss/model:             DiceCE / DynUNet unchanged")
    print(
        f"Translation:            +/-{variant['translate_voxels']:g} vox, p={variant['translate_prob']:.2f}"
        if variant["translate_voxels"] > 0 else
        "Translation:            OFF"
    )
    print(
        f"LR flip:                p={variant['lr_flip_prob']:.2f} (RAS spatial axis 0)"
        if variant["lr_flip"] else
        "LR flip:                OFF"
    )
    print("External31:             NOT ACCESSED")
    print("=" * 116)

    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    kept = [r for r in existing if int(r["fold"]) not in selected_folds]
    combined = kept + new_rows
    combined.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    cv.write_results(results_path, combined)

    compare_to_a0(combined, baseline_rows, selected_folds, output_dir, args.variant)

    print(f"\nResults:    {results_path}")
    print(f"Plan:       {plan_path}")
    print(f"Comparison: {output_dir / 'screening_vs_a0_case_comparison.csv'}")
    print(f"Summary:    {output_dir / 'screening_vs_a0_summary.json'}")


if __name__ == "__main__":
    main()
