#!/usr/bin/env python3
"""Run deterministic K-fold supervised HASSL baselines on all labeled cases.

The default experiment intentionally matches the successful minimal baseline:
  - whole-volume resize to 128^3 after configured spacing
  - DynUNet / binary segmentation
  - DiceCE loss
  - random initialization
  - no strong augmentation
  - dropout=0
  - fixed LR=1e-4
  - lambda_unsup=0

Each fold receives a fresh checkpoint/cache namespace. The original data_dir/splits.json
is never modified or used for fold membership. Fold assignments are persisted separately
under <output-dir>/cv_splits.json so a later augmentation experiment can reuse the exact
same held-out subjects.

By default fold performance is reported with ensemble (student + EMA teacher)/2 at
threshold 0.50. This deliberately avoids reusing the 0.98 threshold selected on the old
5-case validation split, because several of those cases become held-out CV cases.
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
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
from hassl.data.data_engine import _strip_suffix, get_base_transforms
from hassl.training.trainer import build_network


_ORIGINAL_GET_TRANSFORMS = data_engine.get_base_transforms


def _collect_labeled_cases(config: HASSLConfig) -> List[Dict[str, str]]:
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


def _patient_id(case_id: str, patient_id_regex=None) -> str:
    if patient_id_regex:
        match = re.search(patient_id_regex, case_id)
        return match.group(1) if match else case_id
    return case_id.split("_")[0] if "_" in case_id else case_id


def _create_or_load_folds(
    cases: List[Dict[str, str]],
    n_folds: int,
    seed: int,
    patient_id_regex,
    path: Path,
    regenerate: bool = False,
):
    all_ids = sorted(case["id"] for case in cases)

    if path.exists() and not regenerate:
        with open(path, "r") as f:
            manifest = json.load(f)
        manifest_ids = sorted(
            case_id
            for fold in manifest.get("folds", [])
            for case_id in fold.get("val_ids", [])
        )
        if manifest.get("n_folds") != n_folds:
            raise RuntimeError(
                f"Existing {path} has n_folds={manifest.get('n_folds')}, requested {n_folds}. "
                "Use --regenerate-splits to replace it."
            )
        if manifest_ids != all_ids:
            raise RuntimeError(
                f"Existing {path} does not match the current labeled dataset. "
                "Use --regenerate-splits after confirming the dataset changed intentionally."
            )
        return manifest

    patient_to_ids: Dict[str, List[str]] = {}
    for case in cases:
        pid = _patient_id(case["id"], patient_id_regex)
        patient_to_ids.setdefault(pid, []).append(case["id"])

    patient_ids = sorted(patient_to_ids)
    if len(patient_ids) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from only {len(patient_ids)} unique patients."
        )

    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    patient_chunks = [list(chunk) for chunk in np.array_split(np.asarray(patient_ids, dtype=object), n_folds)]

    folds = []
    for fold_idx, val_patients in enumerate(patient_chunks):
        val_patient_set = set(str(x) for x in val_patients)
        val_ids = sorted(
            case_id
            for pid, ids in patient_to_ids.items()
            if pid in val_patient_set
            for case_id in ids
        )
        val_set = set(val_ids)
        train_ids = sorted(case_id for case_id in all_ids if case_id not in val_set)
        folds.append(
            {
                "fold": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "val_patients": sorted(val_patient_set),
            }
        )

    manifest = {
        "version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "patient_id_regex": patient_id_regex,
        "all_case_ids": all_ids,
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _case_lookup(cases):
    return {case["id"]: case for case in cases}


def _make_dataset(items, transform, use_cache=True):
    if use_cache and items:
        return CacheDataset(data=items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(data=items, transform=transform)


def _install_cv_data_hook():
    """Replace normal frozen-split loaders with explicit fold train/validation IDs."""

    def build_cv_dataloaders(config):
        train_ids = set(getattr(config, "_cv_train_ids"))
        val_ids = set(getattr(config, "_cv_val_ids"))
        cases = _collect_labeled_cases(config)
        by_id = _case_lookup(cases)

        missing = sorted((train_ids | val_ids) - set(by_id))
        if missing:
            raise RuntimeError(f"Fold contains labeled IDs that cannot be found: {missing}")

        # No augmentation in the baseline. Spatial/intensity augmentation will be added
        # later as a separate A/B experiment using the same cv_splits.json.
        train_transform = _ORIGINAL_GET_TRANSFORMS(
            config,
            keys=["image", "label"],
            is_training=True,
            apply_strong_aug=False,
        )
        val_transform = _ORIGINAL_GET_TRANSFORMS(
            config,
            keys=["image", "label"],
            is_training=False,
            apply_strong_aug=False,
        )

        train_items = [by_id[x] for x in sorted(train_ids)]
        val_items = [by_id[x] for x in sorted(val_ids)]
        use_cache = bool(getattr(config, "use_cache_dataset", True))

        train_ds = _make_dataset(train_items, train_transform, use_cache=use_cache)
        val_ds = _make_dataset(val_items, val_transform, use_cache=use_cache)
        empty_unlabeled = Dataset([])

        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            shuffle=True,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        unlabeled_loader = DataLoader(
            empty_unlabeled,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, unlabeled_loader, val_loader, val_transform

    data_engine.build_dataloaders = build_cv_dataloaders
    pipeline_module.build_dataloaders = build_cv_dataloaders


def _apply_baseline_overrides(config, resize_size: int, epochs=None):
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
    if epochs is not None:
        config.train_epochs = int(epochs)


def _namespace_fold(config, output_dir: Path, fold_idx: int):
    config.checkpoint_dir = str(output_dir / "checkpoints" / f"fold_{fold_idx}")
    config.cache_dir = str(output_dir / "cache" / f"fold_{fold_idx}")
    if hasattr(config, "log_dir"):
        config.log_dir = str(output_dir / "logs" / f"fold_{fold_idx}")
    config.experiment_name = f"{config.experiment_name}-cv5-fold{fold_idx}"


def _main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def _spacing_from_image(image, config):
    try:
        affine = image.affine[0] if image.affine.ndim == 3 else image.affine
        return tuple(float(torch.linalg.norm(affine[:3, i]).item()) for i in range(3))
    except Exception:
        return tuple(float(x) for x in config.spacing)


def _metrics(pred, target, spacing):
    tp = float((pred * target).sum().item())
    pred_sum = float(pred.sum().item())
    gt_sum = float(target.sum().item())
    eps = 1e-5
    dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
    precision = (tp + eps) / (pred_sum + eps)
    recall = (tp + eps) / (gt_sum + eps)
    pred_fg = float(pred.mean().item())
    gt_fg = float(target.mean().item())
    rve = abs(pred_sum - gt_sum) / (gt_sum + 1e-8) * 100.0

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
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_fg": gt_fg,
        "pred_fg": pred_fg,
        "rve": rve,
        "hd95": hd95,
        "gt_vox": gt_sum,
        "pred_vox": pred_sum,
    }


@torch.no_grad()
def _evaluate_fold(config, val_ids, checkpoint_path: Path, source: str, threshold: float):
    cases = _collect_labeled_cases(config)
    by_id = _case_lookup(cases)
    items = [by_id[x] for x in sorted(val_ids)]
    transform = _ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()

    if source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"Checkpoint has no EMA teacher weights; cannot evaluate source={source!r}")

    roi_size = tuple(config.spatial_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)
    rows = []

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].float().to(device)
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
        spacing = _spacing_from_image(image, config)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            s_logits = _main_prediction(inferer(image, student))
            s_prob = torch.sigmoid(s_logits)
            if source == "student":
                prob = s_prob
            else:
                t_logits = _main_prediction(inferer(image, teacher))
                t_prob = torch.sigmoid(t_logits)
                prob = t_prob if source == "teacher" else 0.5 * (s_prob + t_prob)

        pred = (prob > float(threshold)).float()
        m = _metrics(pred, target, spacing)
        m["case_id"] = case_id
        rows.append(m)

    del student
    if teacher is not None:
        del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _write_results(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold", "case_id", "dice", "precision", "recall", "gt_fg", "pred_fg",
        "rve", "hd95", "gt_vox", "pred_vox", "source", "threshold",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _prepare_fold_storage(path: Path, resume: bool, overwrite: bool):
    if overwrite and path.exists():
        shutil.rmtree(path)
    latest = path / "round0_latest.pth"
    if latest.exists() and not resume:
        raise RuntimeError(
            f"{latest} already exists. Use --resume to continue this fold or --overwrite to restart it cleanly."
        )
    path.mkdir(parents=True, exist_ok=True)


def _run_fold(args, base_config_path: str, fold_spec, output_dir: Path):
    fold_idx = int(fold_spec["fold"])
    config = HASSLConfig.from_yaml(base_config_path)
    _apply_baseline_overrides(config, args.resize_size, args.epochs)
    _namespace_fold(config, output_dir, fold_idx)
    config._cv_train_ids = list(fold_spec["train_ids"])
    config._cv_val_ids = list(fold_spec["val_ids"])

    # Fold-specific but reproducible initialization. Reusing this script later with
    # augmentation keeps the same fold seed, making the A/B comparison cleaner.
    fold_seed = int(args.seed) + fold_idx
    config.seed = fold_seed
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    set_determinism(seed=fold_seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    _prepare_fold_storage(checkpoint_dir, resume=args.resume, overwrite=args.overwrite)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(config, "log_dir"):
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 88)
    print(f"CV FOLD {fold_idx}/{args.folds - 1}")
    print(f"Train cases ({len(config._cv_train_ids)}): {', '.join(config._cv_train_ids)}")
    print(f"Held-out cases ({len(config._cv_val_ids)}): {', '.join(config._cv_val_ids)}")
    print(f"Seed: {fold_seed}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print("Baseline: resize, DiceCE, random init, no augmentation, dropout=0, LR=1e-4")
    print("#" * 88)

    # Fresh fold namespace means there is no ssl_pretrained.pth unless a user explicitly
    # placed one there. We deliberately pass no pretrained weights for the random-init CV.
    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    best_checkpoint = checkpoint_dir / "best_checkpoint.pth"
    if not best_checkpoint.exists():
        raise RuntimeError(f"Training completed but no best checkpoint was found at {best_checkpoint}")

    rows = _evaluate_fold(
        config,
        config._cv_val_ids,
        best_checkpoint,
        source=args.eval_source,
        threshold=args.eval_threshold,
    )
    for row in rows:
        row["fold"] = fold_idx
        row["source"] = args.eval_source
        row["threshold"] = args.eval_threshold

    print(f"\nFOLD {fold_idx} HELD-OUT RESULTS ({args.eval_source}, threshold={args.eval_threshold:.2f})")
    for row in rows:
        print(
            f"  {row['case_id']}: Dice={row['dice']:.4f} | Prec={row['precision']:.4f} | "
            f"Rec={row['recall']:.4f} | RVE={row['rve']:.1f}% | HD95={row['hd95']:.2f}mm"
        )
    print(
        f"  Fold mean Dice={_mean([r['dice'] for r in rows]):.4f} | "
        f"Prec={_mean([r['precision'] for r in rows]):.4f} | "
        f"Recall={_mean([r['recall'] for r in rows]):.4f} | "
        f"HD95={_mean([r['hd95'] for r in rows]):.2f}mm"
    )
    return rows


def _parse_fold_arg(value: str, n_folds: int):
    if value.lower() == "all":
        return list(range(n_folds))
    try:
        idx = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--fold must be 'all' or an integer fold index") from exc
    if idx < 0 or idx >= n_folds:
        raise argparse.ArgumentTypeError(f"--fold must be between 0 and {n_folds - 1}")
    return [idx]


def main():
    parser = argparse.ArgumentParser(description="Run K-fold minimal supervised resize baseline")
    parser.add_argument("--config", required=True, help="Baseline HASSL YAML, e.g. config_resize128.yaml")
    parser.add_argument("--fold", default="0", help="Fold index to run, or 'all' (default: 0)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="CV split seed (default: 42)")
    parser.add_argument("--resize-size", type=int, default=128, help="Whole-volume cubic resize (default: 128)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per fold (default: 100)")
    parser.add_argument(
        "--output-dir",
        default="experiments/cv5_supervised_resize128",
        help="CV manifests/checkpoints/results root",
    )
    parser.add_argument(
        "--eval-source",
        choices=["student", "teacher", "ensemble"],
        default="ensemble",
        help="Frozen held-out prediction source (default: ensemble)",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.50,
        help="Frozen held-out threshold (default: 0.50; avoids legacy-threshold leakage)",
    )
    parser.add_argument("--resume", action="store_true", help="Allow pipeline resume from an existing fold checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Delete the selected fold checkpoint directory and restart")
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Replace cv_splits.json (only use when intentionally changing split setup/data)",
    )
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if args.resize_size < 16:
        parser.error("--resize-size must be >= 16")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if not (0.0 < args.eval_threshold < 1.0):
        parser.error("--eval-threshold must be between 0 and 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    base_config = HASSLConfig.from_yaml(args.config)
    cases = _collect_labeled_cases(base_config)
    if len(cases) < args.folds:
        parser.error(f"Found only {len(cases)} labeled cases, fewer than {args.folds} folds")

    output_dir = Path(args.output_dir)
    split_path = output_dir / "cv_splits.json"
    manifest = _create_or_load_folds(
        cases,
        n_folds=args.folds,
        seed=args.seed,
        patient_id_regex=getattr(base_config, "patient_id_regex", None),
        path=split_path,
        regenerate=args.regenerate_splits,
    )

    selected_folds = _parse_fold_arg(args.fold, args.folds)
    fold_by_idx = {int(f["fold"]): f for f in manifest["folds"]}

    print("=" * 88)
    print("HASSL supervised cross-validation baseline")
    print(f"Labeled cases: {len(cases)}")
    print(f"Folds: {args.folds}")
    print(f"Split manifest: {split_path}")
    print(f"Running folds: {selected_folds}")
    print(f"Epochs/fold: {args.epochs}")
    print("Augmentation: OFF")
    print(f"Held-out evaluation: {args.eval_source} @ threshold {args.eval_threshold:.2f}")
    print("=" * 88)

    _install_cv_data_hook()

    all_rows = []
    for fold_idx in selected_folds:
        rows = _run_fold(args, args.config, fold_by_idx[fold_idx], output_dir)
        all_rows.extend(rows)

    # Preserve results from individually-run folds by merging with an existing CSV.
    results_path = output_dir / "cv_results.csv"
    existing = []
    if results_path.exists():
        with open(results_path, "r", newline="") as f:
            existing = list(csv.DictReader(f))
        selected = set(selected_folds)
        existing = [row for row in existing if int(row["fold"]) not in selected]

    merged = existing + all_rows
    merged.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    _write_results(results_path, merged)

    print(f"\nResults saved: {results_path}")
    if len(selected_folds) == args.folds:
        dice_values = [float(r["dice"]) for r in merged]
        fold_means = []
        for fold_idx in range(args.folds):
            vals = [float(r["dice"]) for r in merged if int(r["fold"]) == fold_idx]
            if vals:
                fold_means.append(float(np.mean(vals)))
        print("\n5-FOLD SUMMARY")
        print(f"Held-out case Dice: {np.mean(dice_values):.4f} ± {np.std(dice_values):.4f}")
        print(f"Fold-mean Dice:     {np.mean(fold_means):.4f} ± {np.std(fold_means):.4f}")
        print(f"Zero-Dice cases:    {sum(v < 1e-6 for v in dice_values)}/{len(dice_values)}")


if __name__ == "__main__":
    main()
