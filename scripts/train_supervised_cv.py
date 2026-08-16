#!/usr/bin/env python3
"""Run deterministic K-fold supervised HASSL experiments on all labeled cases.

Baseline defaults match the successful diagnostic:
  spacing from config -> whole-volume Resize(128^3)
  DynUNet, DiceCE, random init, no augmentation, dropout=0, LR=1e-4,
  lambda_unsup=0, ensemble evaluation at threshold 0.50.

Use --spatial-aug for the controlled A/B experiment. It changes ONLY the training
transform by adding paired image/label spatial augmentation after deterministic
preprocessing:
  - in-plane random flips
  - affine translation up to +/-12 voxels
  - rotation up to +/-7 degrees
  - scale up to +/-10%
Images use bilinear interpolation and labels use nearest-neighbor interpolation.
Validation/held-out preprocessing stays deterministic and unchanged.

The spatial-augmentation run reuses the baseline cv_splits.json by default and writes
checkpoints/results into a separate output directory, so the baseline remains untouched.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import Compose, RandAffined, RandFlipd
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
from hassl.data.data_engine import _strip_suffix
from hassl.training.trainer import build_network

ORIGINAL_GET_TRANSFORMS = data_engine.get_base_transforms
BASELINE_OUTPUT_DIR = Path("experiments/cv5_supervised_resize128")
SPATIAL_AUG_OUTPUT_DIR = Path("experiments/cv5_supervised_resize128_spatialaug")


def collect_cases(config):
    root = Path(config.data_dir)
    cases = []
    for image_path in sorted(glob.glob(str(root / f"**/*{config.image_suffix}"), recursive=True)):
        case_id = _strip_suffix(os.path.basename(image_path), config.image_suffix)
        label_path = root / "labels" / f"{case_id}{config.label_suffix}"
        if not label_path.exists():
            label_path = Path(image_path).parent / f"{case_id}{config.label_suffix}"
        if label_path.exists():
            cases.append({"image": image_path, "label": str(label_path), "id": case_id})
    return cases


def patient_id(case_id, regex=None):
    if regex:
        match = re.search(regex, case_id)
        return match.group(1) if match else case_id
    return case_id.split("_")[0] if "_" in case_id else case_id


def validate_manifest(manifest, cases, n_folds, path):
    all_ids = sorted(c["id"] for c in cases)
    held_out_once = sorted(x for fold in manifest.get("folds", []) for x in fold.get("val_ids", []))
    if manifest.get("n_folds") != n_folds:
        raise RuntimeError(
            f"{path} has n_folds={manifest.get('n_folds')}, requested {n_folds}."
        )
    if held_out_once != all_ids:
        raise RuntimeError(
            f"{path} does not match the current labeled dataset. "
            "Do not continue until the split provenance is resolved."
        )


def create_manifest(cases, n_folds, seed, regex):
    all_ids = sorted(c["id"] for c in cases)
    groups: Dict[str, List[str]] = {}
    for case in cases:
        groups.setdefault(patient_id(case["id"], regex), []).append(case["id"])

    patients = sorted(groups)
    if len(patients) < n_folds:
        raise ValueError(f"Need at least {n_folds} unique patients; found {len(patients)}")

    rng = random.Random(seed)
    rng.shuffle(patients)
    chunks = np.array_split(np.asarray(patients, dtype=object), n_folds)

    folds = []
    for idx, chunk in enumerate(chunks):
        val_patients = {str(x) for x in chunk.tolist()}
        val_ids = sorted(case for pid, ids in groups.items() if pid in val_patients for case in ids)
        val_set = set(val_ids)
        folds.append({
            "fold": idx,
            "train_ids": sorted(x for x in all_ids if x not in val_set),
            "val_ids": val_ids,
            "val_patients": sorted(val_patients),
        })

    return {
        "version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "patient_id_regex": regex,
        "all_case_ids": all_ids,
        "folds": folds,
    }


def load_or_create_manifest(cases, n_folds, seed, regex, source_path, regenerate=False):
    if source_path.exists() and not regenerate:
        manifest = json.loads(source_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, cases, n_folds, source_path)
        return manifest

    if regenerate:
        manifest = create_manifest(cases, n_folds, seed, regex)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    if source_path.exists():
        raise RuntimeError(f"Could not use split manifest: {source_path}")

    manifest = create_manifest(cases, n_folds, seed, regex)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def persist_manifest_copy(manifest, output_dir, source_path):
    destination = output_dir / "cv_splits.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"Existing {destination} differs from source split manifest {source_path}. "
                "Use a fresh output directory rather than mixing experiments."
            )
    else:
        destination.write_text(payload, encoding="utf-8")


def make_dataset(items, transform, use_cache):
    if use_cache and items:
        return CacheDataset(items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(items, transform=transform)


def spatial_aug_transform(base_transform, translate_voxels, rotate_degrees, scale_fraction):
    """Append paired spatial randomization after deterministic 128^3 preprocessing."""
    rotate_radians = math.radians(float(rotate_degrees))
    translate = float(translate_voxels)
    scale = float(scale_fraction)

    # Flatten the base Compose so CacheDataset can cache deterministic preprocessing up to
    # the first Randomizable transform, while still sampling fresh augmentation each epoch.
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.8,
            rotate_range=(rotate_radians, rotate_radians, rotate_radians),
            translate_range=(translate, translate, translate),
            scale_range=(scale, scale, scale),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


def install_cv_loader_hook(args):
    def build_cv_dataloaders(config):
        cases = {c["id"]: c for c in collect_cases(config)}
        train_ids = list(config._cv_train_ids)
        val_ids = list(config._cv_val_ids)
        missing = sorted((set(train_ids) | set(val_ids)) - set(cases))
        if missing:
            raise RuntimeError(f"Missing labeled cases for fold: {missing}")

        train_t = ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        if args.spatial_aug:
            train_t = spatial_aug_transform(
                train_t,
                translate_voxels=args.translate_voxels,
                rotate_degrees=args.rotate_degrees,
                scale_fraction=args.scale_fraction,
            )

        val_t = ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )

        use_cache = bool(getattr(config, "use_cache_dataset", True))
        train_ds = make_dataset([cases[x] for x in sorted(train_ids)], train_t, use_cache)
        val_ds = make_dataset([cases[x] for x in sorted(val_ids)], val_t, use_cache)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            shuffle=True,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_cv_dataloaders
    pipeline_module.build_dataloaders = build_cv_dataloaders


def apply_baseline(config, resize_size, epochs):
    config.preprocessing_mode = "resize"
    config.spatial_size = (int(resize_size),) * 3
    config.loss_type = "dice_ce"
    config.include_boundary = False
    config.lambda_unsup = 0.0
    config.dropout = 0.0
    config.train_lr = 1e-4
    config.lr_scheduler = "none"
    config.lr_warmup_epochs = 0
    config.prediction_threshold = 0.5
    config.train_epochs = int(epochs)


def namespace_fold(config, output_dir, fold_idx, spatial_aug=False):
    config.checkpoint_dir = str(output_dir / "checkpoints" / f"fold_{fold_idx}")
    config.cache_dir = str(output_dir / "cache" / f"fold_{fold_idx}")
    if hasattr(config, "log_dir"):
        config.log_dir = str(output_dir / "logs" / f"fold_{fold_idx}")
    tag = "spatialaug" if spatial_aug else "baseline"
    config.experiment_name = f"{config.experiment_name}-cv5-{tag}-fold{fold_idx}"


def main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def transformed_spacing(image, config):
    try:
        affine = image.affine[0] if image.affine.ndim == 3 else image.affine
        return tuple(float(torch.linalg.norm(affine[:3, i]).item()) for i in range(3))
    except Exception:
        return tuple(float(x) for x in config.spacing)


def case_metrics(pred, target, spacing):
    tp = float((pred * target).sum().item())
    pred_sum = float(pred.sum().item())
    gt_sum = float(target.sum().item())
    eps = 1e-5

    hd95 = float("nan")
    try:
        hd = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
        hd(y_pred=pred, y=target, spacing=spacing)
        value = float(hd.aggregate().item())
        if math.isfinite(value):
            hd95 = value
    except Exception:
        pass

    return {
        "dice": (2 * tp + eps) / (pred_sum + gt_sum + eps),
        "precision": (tp + eps) / (pred_sum + eps),
        "recall": (tp + eps) / (gt_sum + eps),
        "gt_fg": float(target.mean().item()),
        "pred_fg": float(pred.mean().item()),
        "rve": abs(pred_sum - gt_sum) / (gt_sum + 1e-8) * 100.0,
        "hd95": hd95,
        "gt_vox": gt_sum,
        "pred_vox": pred_sum,
    }


@torch.no_grad()
def evaluate_fold(config, val_ids, checkpoint, source, threshold):
    by_id = {c["id"]: c for c in collect_cases(config)}
    transform = ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    loader = DataLoader(
        Dataset([by_id[x] for x in sorted(val_ids)], transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()
    if source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"Checkpoint has no teacher weights; cannot evaluate {source}")

    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].float().to(device)
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            s_prob = torch.sigmoid(main_prediction(inferer(image, student)))
            if source == "student":
                prob = s_prob
            else:
                t_prob = torch.sigmoid(main_prediction(inferer(image, teacher)))
                prob = t_prob if source == "teacher" else 0.5 * (s_prob + t_prob)

        pred = (prob > float(threshold)).float()
        row = case_metrics(pred, target, transformed_spacing(image, config))
        row["case_id"] = case_id
        rows.append(row)

    del student
    if teacher is not None:
        del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def fold_state(checkpoint_dir, overwrite=False):
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    latest = checkpoint_dir / "round0_latest.pth"
    best = checkpoint_dir / "best_checkpoint.pth"
    if latest.exists() and best.exists():
        return "complete"
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth")):
        raise RuntimeError(
            f"Fold directory {checkpoint_dir} contains an incomplete/stale run. "
            "Use --overwrite to restart that fold cleanly."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return "new"


def run_fold(args, fold_spec, output_dir):
    idx = int(fold_spec["fold"])
    config = HASSLConfig.from_yaml(args.config)
    apply_baseline(config, args.resize_size, args.epochs)
    namespace_fold(config, output_dir, idx, spatial_aug=args.spatial_aug)
    config._cv_train_ids = list(fold_spec["train_ids"])
    config._cv_val_ids = list(fold_spec["val_ids"])

    # Same fold seeds for baseline and augmented runs. This keeps initialization/order
    # comparable; the only intended experimental difference is the spatial transform.
    fold_seed = args.seed + idx
    config.seed = fold_seed
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    set_determinism(seed=fold_seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    state = fold_state(checkpoint_dir, overwrite=args.overwrite)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(config, "log_dir"):
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    aug_name = "SPATIAL AUG" if args.spatial_aug else "OFF"
    print("\n" + "#" * 88)
    print(f"CV FOLD {idx}/{args.folds - 1}")
    print(f"Train ({len(config._cv_train_ids)}): {', '.join(config._cv_train_ids)}")
    print(f"Held-out ({len(config._cv_val_ids)}): {', '.join(config._cv_val_ids)}")
    print(f"Seed: {fold_seed} | state: {state}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Augmentation: {aug_name}")
    if args.spatial_aug:
        print(
            f"Spatial params: translate=+/-{args.translate_voxels:g} vox | "
            f"rotate=+/-{args.rotate_degrees:g} deg | scale=+/-{args.scale_fraction:.0%}"
        )
    print("Fixed model/training: resize128, DiceCE, random init, dropout=0, LR=1e-4")
    print("#" * 88)

    if state == "new":
        pipeline_module.run_train(config, round_num=0, pretrained_weights=None)
    else:
        print("  Completed fold found; skipping training and re-evaluating best_checkpoint.pth")

    best = checkpoint_dir / "best_checkpoint.pth"
    if not best.exists():
        raise RuntimeError(f"No best checkpoint found at {best}")

    rows = evaluate_fold(config, config._cv_val_ids, best, args.eval_source, args.eval_threshold)
    experiment = "spatial_aug" if args.spatial_aug else "baseline"
    for row in rows:
        row.update(
            fold=idx,
            source=args.eval_source,
            threshold=args.eval_threshold,
            experiment=experiment,
        )

    print(f"\nFOLD {idx} HELD-OUT RESULTS ({args.eval_source} @ {args.eval_threshold:.2f})")
    for row in rows:
        print(
            f"  {row['case_id']}: Dice={row['dice']:.4f} | Prec={row['precision']:.4f} | "
            f"Rec={row['recall']:.4f} | RVE={row['rve']:.1f}% | HD95={row['hd95']:.2f}mm"
        )
    print(
        f"  Fold mean Dice={mean([r['dice'] for r in rows]):.4f} | "
        f"Prec={mean([r['precision'] for r in rows]):.4f} | "
        f"Recall={mean([r['recall'] for r in rows]):.4f}"
    )
    return rows


def write_results(path, rows):
    fields = [
        "fold", "case_id", "dice", "precision", "recall", "gt_fg", "pred_fg",
        "rve", "hd95", "gt_vox", "pred_vox", "source", "threshold", "experiment",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def parse_fold(value, n_folds):
    if value.lower() == "all":
        return list(range(n_folds))
    idx = int(value)
    if not 0 <= idx < n_folds:
        raise ValueError(f"fold must be 0..{n_folds - 1} or 'all'")
    return [idx]


def read_results(path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_complete_summary(rows, n_folds):
    dices = [float(row["dice"]) for row in rows]
    fold_means = [
        np.mean([float(row["dice"]) for row in rows if int(row["fold"]) == idx])
        for idx in range(n_folds)
    ]
    print("\n5-FOLD SUMMARY")
    print(f"Held-out case Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"Median Dice:        {np.median(dices):.4f}")
    print(f"Fold-mean Dice:     {np.mean(fold_means):.4f} +/- {np.std(fold_means):.4f}")
    print(f"Zero-Dice cases:    {sum(x < 1e-6 for x in dices)}/{len(dices)}")
    print(f"Dice < 0.20:        {sum(x < 0.20 for x in dices)}/{len(dices)}")


def print_baseline_comparison(aug_rows, baseline_path):
    baseline_rows = read_results(baseline_path)
    if not baseline_rows:
        print(f"\nBaseline comparison skipped: no results found at {baseline_path}")
        return

    baseline = {row["case_id"]: float(row["dice"]) for row in baseline_rows}
    augmented = {row["case_id"]: float(row["dice"]) for row in aug_rows}
    common = sorted(set(baseline) & set(augmented))
    if not common:
        print("\nBaseline comparison skipped: no common case IDs")
        return

    deltas = [(case_id, baseline[case_id], augmented[case_id], augmented[case_id] - baseline[case_id]) for case_id in common]
    delta_values = np.asarray([x[3] for x in deltas], dtype=float)

    print("\nSPATIAL AUGMENTATION vs BASELINE")
    print(f"Baseline mean Dice:  {np.mean([baseline[x] for x in common]):.4f}")
    print(f"Augmented mean Dice: {np.mean([augmented[x] for x in common]):.4f}")
    print(f"Mean case delta:     {np.mean(delta_values):+.4f}")
    print(f"Improved cases:      {sum(x > 0 for x in delta_values)}/{len(common)}")
    print(f"Delta >= +0.05:      {sum(x >= 0.05 for x in delta_values)}/{len(common)}")
    print(f"Delta <= -0.05:      {sum(x <= -0.05 for x in delta_values)}/{len(common)}")
    print("\nCase-by-case Dice delta")
    for case_id, base_dice, aug_dice, delta in sorted(deltas, key=lambda x: x[3]):
        print(f"  {case_id}: {base_dice:.4f} -> {aug_dice:.4f} ({delta:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="K-fold supervised resize baseline / spatial-augmentation A/B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default="0", help="0..K-1 or all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    parser.add_argument("--eval-threshold", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true", help="Restart selected fold(s) cleanly")
    parser.add_argument("--regenerate-splits", action="store_true")

    parser.add_argument(
        "--spatial-aug",
        action="store_true",
        help="Enable the controlled spatial-augmentation A/B experiment for training only.",
    )
    parser.add_argument("--translate-voxels", type=float, default=12.0)
    parser.add_argument("--rotate-degrees", type=float, default=7.0)
    parser.add_argument("--scale-fraction", type=float, default=0.10)
    parser.add_argument(
        "--baseline-results",
        default=str(BASELINE_OUTPUT_DIR / "cv_results.csv"),
        help="Baseline CSV used only for final case-by-case A/B reporting.",
    )
    args = parser.parse_args()

    if args.folds < 2 or args.epochs < 1 or args.resize_size < 16:
        parser.error("Require folds>=2, epochs>=1, resize-size>=16")
    if not 0 < args.eval_threshold < 1:
        parser.error("--eval-threshold must be between 0 and 1")
    if args.translate_voxels < 0 or args.rotate_degrees < 0:
        parser.error("Spatial augmentation ranges must be non-negative")
    if not 0 <= args.scale_fraction < 1:
        parser.error("--scale-fraction must be in [0, 1)")

    if args.output_dir is None:
        output_dir = SPATIAL_AUG_OUTPUT_DIR if args.spatial_aug else BASELINE_OUTPUT_DIR
    else:
        output_dir = Path(args.output_dir)

    # The augmented experiment MUST reuse the baseline fold assignment unless the user
    # explicitly supplies another frozen manifest. This prevents a misleading A/B result.
    if args.split_manifest:
        split_source = Path(args.split_manifest)
    elif args.spatial_aug:
        split_source = BASELINE_OUTPUT_DIR / "cv_splits.json"
    else:
        split_source = output_dir / "cv_splits.json"

    if args.spatial_aug and not split_source.exists():
        parser.error(
            f"Spatial A/B requires the frozen baseline split manifest, but {split_source} does not exist. "
            "Run the baseline CV first or pass --split-manifest explicitly."
        )
    if args.spatial_aug and args.regenerate_splits:
        parser.error("Do not use --regenerate-splits for the spatial A/B experiment; reuse the baseline folds")

    base = HASSLConfig.from_yaml(args.config)
    cases = collect_cases(base)
    if len(cases) < args.folds:
        parser.error(f"Found {len(cases)} labeled cases for {args.folds} folds")

    manifest = load_or_create_manifest(
        cases,
        args.folds,
        args.seed,
        getattr(base, "patient_id_regex", None),
        split_source,
        regenerate=args.regenerate_splits,
    )
    persist_manifest_copy(manifest, output_dir, split_source)

    try:
        selected = parse_fold(args.fold, args.folds)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    aug_label = "SPATIAL" if args.spatial_aug else "OFF"
    print("=" * 88)
    print("HASSL supervised cross-validation")
    print(f"Labeled cases: {len(cases)} | folds: {args.folds} | running: {selected}")
    print(f"Frozen split source: {split_source}")
    print(f"Experiment output:  {output_dir}")
    print(f"Epochs/fold: {args.epochs} | augmentation: {aug_label}")
    if args.spatial_aug:
        print(
            f"Spatial augmentation: translate=+/-{args.translate_voxels:g} vox, "
            f"rotate=+/-{args.rotate_degrees:g} deg, scale=+/-{args.scale_fraction:.0%}, "
            "in-plane flips"
        )
    print(f"Held-out evaluation: {args.eval_source} @ {args.eval_threshold:.2f}")
    print("=" * 88)

    install_cv_loader_hook(args)
    fold_map = {int(fold["fold"]): fold for fold in manifest["folds"]}
    new_rows = []
    for idx in selected:
        new_rows.extend(run_fold(args, fold_map[idx], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = read_results(results_path)
    selected_set = set(selected)
    existing = [row for row in existing if int(row["fold"]) not in selected_set]
    merged = existing + new_rows
    merged.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    write_results(results_path, merged)
    print(f"\nResults saved: {results_path}")

    completed_folds = {int(row["fold"]) for row in merged}
    if completed_folds == set(range(args.folds)):
        print_complete_summary(merged, args.folds)
        if args.spatial_aug:
            print_baseline_comparison(merged, Path(args.baseline_results))


if __name__ == "__main__":
    main()
