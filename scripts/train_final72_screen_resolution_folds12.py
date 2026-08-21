#!/usr/bin/env python3
"""Fast Final72 resolution screening on frozen folds 1 and 2.

Reference: A3 at 128^3 = DiceCE + LR flip p=.5 + translation +/-4 vox p=.5.
Candidates: 144^3 and 160^3.

To isolate resolution, translation is scaled by resolution/128 so the normalized displacement
remains 3.125% of an axis:
    128 -> +/-4.0 vox
    144 -> +/-4.5 vox
    160 -> +/-5.0 vox

Everything else stays fixed: audited Final72, exact original47 held-out folds, DynUNet,
DiceCE, AdamW 1e-4, dropout0, lambda_unsup0, seed42, 100 epochs, random initialization,
raw Student+EMA 50/50 ensemble at threshold .50, no LCC, no external31 access.
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
import scripts.train_final72_screen_spatial_folds12 as spatial_screen

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
A3_128 = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")

ACTIVE_RESOLUTION = None


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


def matched_a3_transform(base_transform, translate_voxels=0.0, rotate_degrees=0.0, scale_fraction=0.0):
    """A3 spatial recipe with translation scaled to preserve normalized displacement."""
    if ACTIVE_RESOLUTION is None:
        raise RuntimeError("ACTIVE_RESOLUTION not initialized")
    resolution = int(ACTIVE_RESOLUTION)
    translate = 4.0 * resolution / 128.0
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(translate, translate, translate),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


def merge_rows(existing, new_rows, selected_folds):
    selected = set(int(x) for x in selected_folds)
    keep = [r for r in existing if int(r["fold"]) not in selected]
    return sorted(keep + list(new_rows), key=lambda r: (int(r["fold"]), str(r["case_id"])))


def compare_to_128(candidate_rows, baseline_rows, selected_folds, output_dir, resolution):
    selected = set(selected_folds)
    cand = {str(r["case_id"]): r for r in candidate_rows if int(r["fold"]) in selected}
    base = {str(r["case_id"]): r for r in baseline_rows if int(r["fold"]) in selected}
    if set(cand) != set(base):
        raise RuntimeError("Candidate resolution and A3-128 must contain exact same held-out IDs")

    paired = []
    for case_id in sorted(base):
        a, b = base[case_id], cand[case_id]
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "resolution": int(resolution),
            "a3_128_dice": float(a["dice"]),
            "candidate_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a3_128_precision": float(a["precision"]),
            "candidate_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a3_128_recall": float(a["recall"]),
            "candidate_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a3_128_rve": float(a["rve"]),
            "candidate_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
            "a3_128_hd95": float(a["hd95"]),
            "candidate_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
        })
    write_csv(output_dir / "screening_vs_a3_128_case_comparison.csv", paired)

    folds = []
    for fold in sorted(selected):
        subset = [x for x in paired if int(x["fold"]) == fold]
        bd = np.asarray([x["a3_128_dice"] for x in subset], dtype=float)
        cd = np.asarray([x["candidate_dice"] for x in subset], dtype=float)
        folds.append({
            "fold": fold,
            "n": len(subset),
            "a3_128_mean_dice": float(np.mean(bd)),
            "candidate_mean_dice": float(np.mean(cd)),
            "delta_mean_dice": float(np.mean(cd) - np.mean(bd)),
            "delta_mean_precision": float(np.mean([x["delta_precision"] for x in subset])),
            "delta_mean_recall": float(np.mean([x["delta_recall"] for x in subset])),
            "improved": int(np.sum(cd > bd + 1e-6)),
            "worsened": int(np.sum(cd < bd - 1e-6)),
            "improved_ge_0p05": int(np.sum(cd - bd >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(cd - bd <= -0.05)),
        })

    bd = np.asarray([x["a3_128_dice"] for x in paired], dtype=float)
    cd = np.asarray([x["candidate_dice"] for x in paired], dtype=float)
    summary = {
        "version": "final72_resolution_screen_folds12_v1",
        "reference": "A3 128^3",
        "candidate_resolution": int(resolution),
        "translation_voxels": 4.0 * int(resolution) / 128.0,
        "translation_fraction_of_axis": 4.0 / 128.0,
        "screening_folds": sorted(selected),
        "n_cases": len(paired),
        "a3_128_mean_dice": float(np.mean(bd)),
        "candidate_mean_dice": float(np.mean(cd)),
        "delta_mean_dice": float(np.mean(cd) - np.mean(bd)),
        "a3_128_mean_precision": float(np.mean([x["a3_128_precision"] for x in paired])),
        "candidate_mean_precision": float(np.mean([x["candidate_precision"] for x in paired])),
        "a3_128_mean_recall": float(np.mean([x["a3_128_recall"] for x in paired])),
        "candidate_mean_recall": float(np.mean([x["candidate_recall"] for x in paired])),
        "improved": int(np.sum(cd > bd + 1e-6)),
        "worsened": int(np.sum(cd < bd - 1e-6)),
        "improved_ge_0p05": int(np.sum(cd - bd >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum(cd - bd <= -0.05)),
        "folds": folds,
        "screening_only": True,
    }
    (output_dir / "screening_vs_a3_128_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print(f"R{resolution} vs A3-128 — RESOLUTION SCREENING")
    for f in folds:
        print(
            f"Fold {f['fold']}: 128={f['a3_128_mean_dice']:.4f} -> {resolution}={f['candidate_mean_dice']:.4f} "
            f"({f['delta_mean_dice']:+.4f}) | PrecΔ={f['delta_mean_precision']:+.4f} | "
            f"RecΔ={f['delta_mean_recall']:+.4f} | improved={f['improved']} | worsened={f['worsened']}"
        )
    print(
        f"Combined: 128={summary['a3_128_mean_dice']:.4f} -> {resolution}={summary['candidate_mean_dice']:.4f} "
        f"({summary['delta_mean_dice']:+.4f})"
    )
    print(
        f"Precision: {summary['a3_128_mean_precision']:.4f} -> {summary['candidate_mean_precision']:.4f} | "
        f"Recall: {summary['a3_128_mean_recall']:.4f} -> {summary['candidate_mean_recall']:.4f}"
    )
    print(
        f"Case effects: improved={summary['improved']} | worsened={summary['worsened']} | "
        f"+>=.05={summary['improved_ge_0p05']} | <=-.05={summary['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: resolution winner must still be confirmed on all frozen folds.")
    print("=" * 120)


def main():
    global ACTIVE_RESOLUTION
    p = argparse.ArgumentParser(description="Final72 A3 resolution screening on frozen folds 1+2")
    p.add_argument("--config", required=True)
    p.add_argument("--resolution", required=True, type=int, choices=[144, 160])
    p.add_argument("--fold", default="screen", help="screen (=1+2), all, or 0..4")
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--a3-128-dir", default=str(A3_128))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-source", choices=["ensemble"], default="ensemble")
    p.add_argument("--eval-threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs != 100 or args.seed != 42:
        p.error("Controlled screen locks epochs=100 and seed=42")
    if abs(args.eval_threshold - 0.50) > 1e-8:
        p.error("Controlled screen locks ensemble threshold=.50")

    ACTIVE_RESOLUTION = int(args.resolution)
    selected_folds = spatial_screen.parse_screen_fold(args.fold)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/final72_screen_a3_res{args.resolution}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or int(config.num_classes) != 1:
        raise RuntimeError("Resolution screen requires prototype binary bladder setup")

    source_manifest_path = Path(args.source_cv_dir) / "cv_splits.json"
    source_manifest, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    a3_rows = read_csv(Path(args.a3_128_dir) / "cv_results.csv")
    a3_selected = [r for r in a3_rows if int(r["fold"]) in set(selected_folds)]
    expected_ids = {str(x) for f in selected_folds for x in fold_map[f]["val_ids"]}
    if {str(r["case_id"]) for r in a3_selected} != expected_ids:
        raise RuntimeError("A3-128 reference does not contain exact selected held-out fold IDs")

    cv.spatial_aug_transform = matched_a3_transform
    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=42,
        resize_size=int(args.resolution),
        epochs=100,
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source="ensemble",
        eval_threshold=0.50,
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=4.0 * int(args.resolution) / 128.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(Path(args.a3_128_dir) / "cv_results.csv"),
    )
    cv.install_cv_loader_hook(runtime_args)

    plan = {
        "version": "final72_resolution_screen_folds12_v1",
        "resolution": [int(args.resolution)] * 3,
        "reference_resolution": [128, 128, 128],
        "spatial_recipe": "A3: LR flip p=.5 + matched normalized translation p=.5",
        "translation_voxels": 4.0 * int(args.resolution) / 128.0,
        "translation_axis_fraction": 4.0 / 128.0,
        "loss": "dice_ce",
        "selected_folds": selected_folds,
        "n_train_only_extras": len(extra_ids),
        "eval": "raw Student+EMA 50/50 ensemble @ .50",
        "external31_access": False,
    }
    plan_path = output_dir / "screening_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError(f"Existing plan differs at {plan_path}; use fresh output dir or --overwrite fold checkpoints only")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    existing = read_csv(output_dir / "cv_results.csv") if (output_dir / "cv_results.csv").exists() else []
    new_rows = []
    print("=" * 116)
    print(f"FINAL72 A3 RESOLUTION SCREEN: {args.resolution}^3")
    print(f"Folds: {selected_folds} | translation=+/-{4.0 * int(args.resolution) / 128.0:.2f} vox p=.5 | LR flip p=.5")
    print("DiceCE | Student+EMA @ .50 | RAW | external31 excluded")
    print("=" * 116)

    for fold in selected_folds:
        rows = cv.run_fold(runtime_args, fold_map[fold], output_dir)
        for row in rows:
            row["experiment"] = f"final72_a3_res{args.resolution}"
        new_rows.extend(rows)

    merged = merge_rows(existing, new_rows, selected_folds)
    cv.write_results(output_dir / "cv_results.csv", merged)
    compare_to_128(merged, a3_rows, selected_folds, output_dir, int(args.resolution))

    print(f"Results: {output_dir / 'cv_results.csv'}")
    print(f"Plan:    {plan_path}")


if __name__ == "__main__":
    main()
