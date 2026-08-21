#!/usr/bin/env python3
"""Screen fold-safe small-bladder oversampling on top of the Final72 A3 recipe.

Base recipe (fixed)
-------------------
- Final72 HUMAN_GOLD
- frozen original47 OOF folds
- DynUNet, resize128, DiceCE
- translation +/-4 vox, p=0.5
- left/right flip p=0.5
- AdamW 1e-4, dropout0, lambda_unsup0
- 100 epochs, seed42
- raw Student+EMA 50/50 ensemble @ 0.50

Only the TRAINING SAMPLER changes.
For each fold independently, native physical bladder volume is computed from that fold's
TRAINING labels only. The bottom training-volume tercile is tagged SMALL for sampling.
Validation labels are never used to determine sampling weights.

Variants
--------
S1 : SMALL training cases weight 1.5, all others weight 1.0
S2 : SMALL training cases weight 2.0, all others weight 1.0

WeightedRandomSampler uses exactly len(train_ids) samples per epoch, so the number of optimizer
steps per epoch remains matched to A3. Sampling is with replacement, which is part of the
oversampling intervention.

Default screening folds are 1 and 2. The candidate is compared directly against the completed
A3 results on the same held-out IDs. Fixed Final72 global SMALL/MEDIUM/LARGE groups are used only
for post-hoc reporting, never for training.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nrrd
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, RandAffined, RandFlipd
from torch.utils.data import WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
import scripts.train_supervised_cv as cv
import scripts.train_final72_screen_spatial_folds12 as spatial_screen
from scripts.analyze_final72_bladder_size_oof import seg_voxel_volume_mm3

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
A3_CV = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
SIZE_PROFILE = Path("experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv")

VARIANTS = {
    "S1": {
        "name": "small_weight_1p5",
        "small_weight": 1.5,
        "output": Path("experiments/final72_screen_s1_small_weight1p5"),
    },
    "S2": {
        "name": "small_weight_2p0",
        "small_weight": 2.0,
        "output": Path("experiments/final72_screen_s2_small_weight2p0"),
    },
}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def a3_transform(base_transform, translate_voxels=4.0, rotate_degrees=0.0, scale_fraction=0.0):
    """Exact A3 training transform: LR flip p=.5 + paired translation +/-4 p=.5."""
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


def native_bladder_ml(case):
    """Compute physical HUMAN_GOLD bladder volume from one training label."""
    label_path = str(case["label"])
    image_path = str(case["image"])

    import SimpleITK as sitk
    image = sitk.ReadImage(image_path)
    fallback_spacing = tuple(float(x) for x in image.GetSpacing())

    seg, header = nrrd.read(label_path)
    seg = np.squeeze(seg)
    if seg.ndim != 3:
        raise RuntimeError(f"Label is not 3D after squeeze: {label_path} shape={seg.shape}")
    fg = int(np.count_nonzero(seg > 0))
    if fg <= 0:
        raise RuntimeError(f"Empty HUMAN_GOLD label: {label_path}")
    voxel_mm3 = seg_voxel_volume_mm3(header, fallback_spacing)
    return float(fg * voxel_mm3 / 1000.0)


def build_fold_sampling(train_ids, cases_by_id, small_weight):
    """Derive SMALL threshold and weights from this fold's training labels only."""
    volumes = {case_id: native_bladder_ml(cases_by_id[case_id]) for case_id in train_ids}
    values = np.asarray([volumes[x] for x in train_ids], dtype=float)
    q1 = float(np.quantile(values, 1.0 / 3.0))
    small_ids = sorted([x for x in train_ids if volumes[x] <= q1])
    small_set = set(small_ids)
    weights = [float(small_weight if x in small_set else 1.0) for x in train_ids]
    return q1, small_ids, volumes, weights


def make_dataset(items, transform, use_cache):
    if use_cache and items:
        return CacheDataset(items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(items, transform=transform)


def install_oversampling_loader_hook(small_weight, sampling_records, fold_lookup):
    """Install A3 dataloaders with fold-local weighted sampling and fixed epoch length."""
    def build_cv_dataloaders(config):
        cases = {c["id"]: c for c in cv.collect_cases(config)}
        train_ids = sorted(list(config._cv_train_ids))
        val_ids = sorted(list(config._cv_val_ids))
        missing = sorted((set(train_ids) | set(val_ids)) - set(cases))
        if missing:
            raise RuntimeError(f"Missing labeled cases for fold: {missing}")

        key = frozenset(train_ids)
        if key not in fold_lookup:
            raise RuntimeError("Current training-ID set does not match any frozen Final72 fold")
        fold_idx = int(fold_lookup[key])

        q1, small_ids, volumes, weights = build_fold_sampling(
            train_ids, cases, float(small_weight)
        )
        sampling_records[fold_idx] = {
            "fold": fold_idx,
            "n_train": len(train_ids),
            "small_weight": float(small_weight),
            "small_threshold_ml_training_only": q1,
            "n_small_training": len(small_ids),
            "small_training_ids": small_ids,
            "training_case_volumes_ml": {k: float(volumes[k]) for k in train_ids},
            "sampler": "WeightedRandomSampler(replacement=True)",
            "samples_per_epoch": len(train_ids),
            "steps_per_epoch_matched_to_a3": True,
            "validation_labels_used_for_sampling": False,
        }

        train_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        train_t = a3_transform(train_t)
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )

        train_items = [cases[x] for x in train_ids]
        val_items = [cases[x] for x in val_ids]
        use_cache = bool(getattr(config, "use_cache_dataset", True))
        train_ds = make_dataset(train_items, train_t, use_cache)
        val_ds = make_dataset(val_items, val_t, use_cache)

        generator = torch.Generator()
        generator.manual_seed(int(config.seed) + 10000)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ids),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            sampler=sampler,
            shuffle=False,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_cv_dataloaders
    pipeline_module.build_dataloaders = build_cv_dataloaders


def merge_fold_rows(existing_rows, new_rows, selected_folds):
    selected = set(int(x) for x in selected_folds)
    kept = [r for r in existing_rows if int(r["fold"]) not in selected]
    merged = kept + list(new_rows)
    return sorted(merged, key=lambda r: (int(r["fold"]), str(r["case_id"])))


def summarize_metric(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def compare_to_a3(candidate_rows, a3_rows, selected_folds, size_profile, output_dir, variant_id):
    selected = set(int(x) for x in selected_folds)
    cand = {str(r["case_id"]): r for r in candidate_rows if int(r["fold"]) in selected}
    base = {str(r["case_id"]): r for r in a3_rows if int(r["fold"]) in selected}
    if set(cand) != set(base):
        raise RuntimeError("Candidate and A3 must contain exact same held-out IDs for selected folds")

    size_by_id = {str(r["case_id"]): str(r["size_group"]) for r in size_profile}
    paired = []
    for case_id in sorted(cand):
        if case_id not in size_by_id:
            raise RuntimeError(f"Missing fixed size group for {case_id}")
        a = base[case_id]
        b = cand[case_id]
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "size_group": size_by_id[case_id],
            "a3_dice": float(a["dice"]),
            "candidate_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a3_precision": float(a["precision"]),
            "candidate_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a3_recall": float(a["recall"]),
            "candidate_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a3_hd95": float(a["hd95"]),
            "candidate_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
            "a3_rve": float(a["rve"]),
            "candidate_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
        })
    write_csv(output_dir / "screening_vs_a3_case_comparison.csv", paired)

    def block(subset):
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "a3_mean_dice": summarize_metric(subset, "a3_dice"),
            "candidate_mean_dice": summarize_metric(subset, "candidate_dice"),
            "delta_mean_dice": summarize_metric(subset, "delta_dice"),
            "a3_mean_precision": summarize_metric(subset, "a3_precision"),
            "candidate_mean_precision": summarize_metric(subset, "candidate_precision"),
            "delta_mean_precision": summarize_metric(subset, "delta_precision"),
            "a3_mean_recall": summarize_metric(subset, "a3_recall"),
            "candidate_mean_recall": summarize_metric(subset, "candidate_recall"),
            "delta_mean_recall": summarize_metric(subset, "delta_recall"),
            "a3_mean_hd95": summarize_metric(subset, "a3_hd95"),
            "candidate_mean_hd95": summarize_metric(subset, "candidate_hd95"),
            "improved": int(sum(float(r["delta_dice"]) > 1e-6 for r in subset)),
            "worsened": int(sum(float(r["delta_dice"]) < -1e-6 for r in subset)),
            "improved_ge_0p05": int(sum(float(r["delta_dice"]) >= 0.05 for r in subset)),
            "worsened_le_minus_0p05": int(sum(float(r["delta_dice"]) <= -0.05 for r in subset)),
        }

    folds = {str(f): block([r for r in paired if int(r["fold"]) == f]) for f in selected_folds}
    groups = {g: block([r for r in paired if r["size_group"] == g]) for g in ("SMALL", "MEDIUM", "LARGE")}
    overall = block(paired)
    summary = {
        "version": "final72_small_bladder_oversampling_screen_v1",
        "variant": variant_id,
        "reference": "A3 = +/-4 vox p=.5 + LR flip p=.5 + DiceCE + 128^3",
        "screening_folds": list(selected_folds),
        "overall": overall,
        "folds": folds,
        "fixed_validation_size_groups": groups,
        "screening_only": True,
    }
    (output_dir / "screening_vs_a3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 122)
    print(f"{variant_id} vs A3 — SMALL-BLADDER OVERSAMPLING SCREEN")
    for f in selected_folds:
        s = folds[str(f)]
        print(
            f"Fold {f}: A3={s['a3_mean_dice']:.4f} -> candidate={s['candidate_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | PrecDelta={s['delta_mean_precision']:+.4f} | "
            f"RecDelta={s['delta_mean_recall']:+.4f}"
        )
    print(
        f"Combined: A3={overall['a3_mean_dice']:.4f} -> candidate={overall['candidate_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f}) | PrecDelta={overall['delta_mean_precision']:+.4f} | "
        f"RecDelta={overall['delta_mean_recall']:+.4f}"
    )
    print("\nFIXED VALIDATION SIZE GROUPS (diagnostic only)")
    for g in ("SMALL", "MEDIUM", "LARGE"):
        s = groups[g]
        if s["n"] == 0:
            print(f"  {g}: n=0")
            continue
        print(
            f"  {g}: n={s['n']} | Dice {s['a3_mean_dice']:.4f}->{s['candidate_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | Prec {s['a3_mean_precision']:.4f}->{s['candidate_mean_precision']:.4f} "
            f"({s['delta_mean_precision']:+.4f}) | Rec {s['a3_mean_recall']:.4f}->{s['candidate_mean_recall']:.4f} "
            f"({s['delta_mean_recall']:+.4f}) | HD95 {s['a3_mean_hd95']:.2f}->{s['candidate_mean_hd95']:.2f}mm"
        )
    print(
        f"Case effects: improved={overall['improved']} | worsened={overall['worsened']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: full frozen CV is required before promotion.")
    print("=" * 122)
    return summary


def main():
    p = argparse.ArgumentParser(description="Final72 fold-safe small-bladder oversampling screen")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--fold", default="screen", help="screen (=1+2), all, or 0..4")
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--a3-cv-dir", default=str(A3_CV))
    p.add_argument("--size-profile", default=str(SIZE_PROFILE))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    variant = dict(VARIANTS[args.variant])
    selected_folds = spatial_screen.parse_screen_fold(args.fold)
    source_cv_dir = Path(args.source_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    a3_cv_dir = Path(args.a3_cv_dir)
    a3_results_path = a3_cv_dir / "cv_results.csv"
    output_dir = Path(args.output_dir) if args.output_dir else Path(variant["output"])

    a3_rows = read_csv(a3_results_path)
    if len({str(r["case_id"]) for r in a3_rows}) != 47:
        raise RuntimeError("A3 reference must contain all 47 frozen OOF cases")
    size_profile = read_csv(Path(args.size_profile))

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or int(config.num_classes) != 1:
        raise RuntimeError("Screen is locked to prototype binary bladder training")

    _, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}
    fold_lookup = {
        frozenset(str(x) for x in spec["train_ids"]): int(spec["fold"])
        for spec in fold_specs
    }

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
        baseline_results=str(a3_results_path),
    )

    sampling_records = {}
    install_oversampling_loader_hook(variant["small_weight"], sampling_records, fold_lookup)

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 122)
    print("FINAL72 SMALL-BLADDER OVERSAMPLING SCREEN")
    print(f"Variant:                    {args.variant} / {variant['name']}")
    print(f"Running folds:              {selected_folds}")
    print(f"Small training weight:      {variant['small_weight']:.2f}x")
    print("Small definition:           bottom tercile of TRAINING physical bladder volume, per fold")
    print("Validation used for weights:NO")
    print("Samples per epoch:          unchanged vs A3")
    print("Base recipe:                A3 (128, DiceCE, +/-4 p=.5, LR flip p=.5)")
    print("External31:                 NOT ACCESSED")
    print("=" * 122)

    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    combined = merge_fold_rows(existing, new_rows, selected_folds)
    cv.write_results(results_path, combined)

    sampling_path = output_dir / "fold_training_sampling.json"
    sampling_path.write_text(
        json.dumps({str(k): v for k, v in sorted(sampling_records.items())}, indent=2),
        encoding="utf-8",
    )

    plan = {
        "version": "final72_small_bladder_oversampling_screen_v1",
        "variant": args.variant,
        "small_training_weight": float(variant["small_weight"]),
        "small_definition": "per-fold bottom tercile of native physical bladder volume from training labels only",
        "sampler": "WeightedRandomSampler with replacement",
        "samples_per_epoch": "exactly len(train_ids), matched to A3",
        "validation_size_groups": "fixed Final72 global groups, reporting only",
        "base_recipe": "A3: DynUNet 128^3 DiceCE translation +/-4 p=.5 LR flip p=.5",
        "screening_folds": selected_folds,
        "n_total_human_gold": 72,
        "n_train_only_extra": len(extra_ids),
        "external31_access": False,
    }
    plan_path = output_dir / "screening_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    compare_to_a3(combined, a3_rows, selected_folds, size_profile, output_dir, args.variant)

    print(f"\nResults:          {results_path}")
    print(f"Sampling audit:   {sampling_path}")
    print(f"Plan:             {plan_path}")
    print(f"Comparison:       {output_dir / 'screening_vs_a3_case_comparison.csv'}")
    print(f"Summary:          {output_dir / 'screening_vs_a3_summary.json'}")


if __name__ == "__main__":
    main()
