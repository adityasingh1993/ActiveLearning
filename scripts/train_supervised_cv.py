#!/usr/bin/env python3
"""Run K-fold minimal supervised HASSL baselines on all labeled cases.

Defaults intentionally match the successful diagnostic baseline:
Spacing from config -> whole-volume Resize(128^3), DynUNet, DiceCE, random init,
no augmentation, dropout=0, fixed LR=1e-4, lambda_unsup=0.

Fold membership is stored in <output-dir>/cv_splits.json and does not modify the
project's normal data_dir/splits.json. Completed folds are skipped and re-evaluated,
so it is safe to run fold 0 first and later run --fold all.
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
from hassl.data.data_engine import _strip_suffix
from hassl.training.trainer import build_network

ORIGINAL_GET_TRANSFORMS = data_engine.get_base_transforms


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
        m = re.search(regex, case_id)
        return m.group(1) if m else case_id
    return case_id.split("_")[0] if "_" in case_id else case_id


def load_or_create_folds(cases, n_folds, seed, regex, path, regenerate=False):
    all_ids = sorted(c["id"] for c in cases)
    if path.exists() and not regenerate:
        manifest = json.loads(path.read_text())
        held_out_once = sorted(x for f in manifest.get("folds", []) for x in f.get("val_ids", []))
        if manifest.get("n_folds") != n_folds or held_out_once != all_ids:
            raise RuntimeError(
                f"Existing {path} does not match this dataset/fold count. "
                "Use --regenerate-splits only if that change is intentional."
            )
        return manifest

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
        val_ids = sorted(case for p, ids in groups.items() if p in val_patients for case in ids)
        val_set = set(val_ids)
        folds.append({
            "fold": idx,
            "train_ids": sorted(x for x in all_ids if x not in val_set),
            "val_ids": val_ids,
            "val_patients": sorted(val_patients),
        })

    manifest = {
        "version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "patient_id_regex": regex,
        "all_case_ids": all_ids,
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def make_dataset(items, transform, use_cache):
    if use_cache and items:
        return CacheDataset(items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(items, transform=transform)


def install_cv_loader_hook():
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


def namespace_fold(config, output_dir, fold_idx):
    config.checkpoint_dir = str(output_dir / "checkpoints" / f"fold_{fold_idx}")
    config.cache_dir = str(output_dir / "cache" / f"fold_{fold_idx}")
    if hasattr(config, "log_dir"):
        config.log_dir = str(output_dir / "logs" / f"fold_{fold_idx}")
    config.experiment_name = f"{config.experiment_name}-cv5-fold{fold_idx}"


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
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


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
    namespace_fold(config, output_dir, idx)
    config._cv_train_ids = list(fold_spec["train_ids"])
    config._cv_val_ids = list(fold_spec["val_ids"])

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

    print("\n" + "#" * 88)
    print(f"CV FOLD {idx}/{args.folds - 1}")
    print(f"Train ({len(config._cv_train_ids)}): {', '.join(config._cv_train_ids)}")
    print(f"Held-out ({len(config._cv_val_ids)}): {', '.join(config._cv_val_ids)}")
    print(f"Seed: {fold_seed} | state: {state}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("Baseline: resize128, DiceCE, random init, augmentation OFF, dropout=0, LR=1e-4")
    print("#" * 88)

    if state == "new":
        pipeline_module.run_train(config, round_num=0, pretrained_weights=None)
    else:
        print("  Completed fold found; skipping training and re-evaluating best_checkpoint.pth")

    best = checkpoint_dir / "best_checkpoint.pth"
    if not best.exists():
        raise RuntimeError(f"No best checkpoint found at {best}")
    rows = evaluate_fold(config, config._cv_val_ids, best, args.eval_source, args.eval_threshold)
    for row in rows:
        row.update(fold=idx, source=args.eval_source, threshold=args.eval_threshold)

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
        "rve", "hd95", "gt_vox", "pred_vox", "source", "threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def parse_fold(value, n_folds):
    if value.lower() == "all":
        return list(range(n_folds))
    idx = int(value)
    if not 0 <= idx < n_folds:
        raise ValueError(f"fold must be 0..{n_folds - 1} or 'all'")
    return [idx]


def main():
    p = argparse.ArgumentParser(description="K-fold minimal supervised resize baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", default="0", help="0..K-1 or all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--output-dir", default="experiments/cv5_supervised_resize128")
    p.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    p.add_argument("--eval-threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true", help="Restart selected fold(s) cleanly")
    p.add_argument("--regenerate-splits", action="store_true")
    args = p.parse_args()

    if args.folds < 2 or args.epochs < 1 or args.resize_size < 16:
        p.error("Require folds>=2, epochs>=1, resize-size>=16")
    if not 0 < args.eval_threshold < 1:
        p.error("--eval-threshold must be between 0 and 1")

    base = HASSLConfig.from_yaml(args.config)
    cases = collect_cases(base)
    if len(cases) < args.folds:
        p.error(f"Found {len(cases)} labeled cases for {args.folds} folds")

    output_dir = Path(args.output_dir)
    manifest = load_or_create_folds(
        cases,
        args.folds,
        args.seed,
        getattr(base, "patient_id_regex", None),
        output_dir / "cv_splits.json",
        args.regenerate_splits,
    )
    try:
        selected = parse_fold(args.fold, args.folds)
    except (ValueError, TypeError) as e:
        p.error(str(e))

    print("=" * 88)
    print("HASSL supervised cross-validation baseline")
    print(f"Labeled cases: {len(cases)} | folds: {args.folds} | running: {selected}")
    print(f"Split manifest: {output_dir / 'cv_splits.json'}")
    print(f"Epochs/fold: {args.epochs} | augmentation: OFF")
    print(f"Held-out evaluation: {args.eval_source} @ {args.eval_threshold:.2f}")
    print("=" * 88)

    install_cv_loader_hook()
    fold_map = {int(f["fold"]): f for f in manifest["folds"]}
    new_rows = []
    for idx in selected:
        new_rows.extend(run_fold(args, fold_map[idx], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = []
    if results_path.exists():
        existing = list(csv.DictReader(open(results_path, newline="")))
        existing = [r for r in existing if int(r["fold"]) not in set(selected)]
    merged = existing + new_rows
    merged.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    write_results(results_path, merged)
    print(f"\nResults saved: {results_path}")

    completed_folds = {int(r["fold"]) for r in merged}
    if completed_folds == set(range(args.folds)):
        dices = [float(r["dice"]) for r in merged]
        fold_means = [
            np.mean([float(r["dice"]) for r in merged if int(r["fold"]) == idx])
            for idx in range(args.folds)
        ]
        print("\n5-FOLD SUMMARY")
        print(f"Held-out case Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Fold-mean Dice:     {np.mean(fold_means):.4f} ± {np.std(fold_means):.4f}")
        print(f"Zero-Dice cases:    {sum(x < 1e-6 for x in dices)}/{len(dices)}")


if __name__ == "__main__":
    main()
