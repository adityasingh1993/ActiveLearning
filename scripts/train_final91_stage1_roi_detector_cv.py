#!/usr/bin/env python3
"""Train and gate a true Stage-1 bladder ROI detector on Final91.

This is deliberately not another bladder-boundary segmenter.  The training target is an
expanded cuboid around the HUMAN_GOLD bladder.  At inference the detector probability is
converted into a generous crop proposal, retaining multiple substantial components.  The
experiment may proceed to a Stage-2 crop segmenter only if the out-of-fold original47 crop
proposals contain essentially every bladder voxel while still reducing the field of view.

Leakage controls
----------------
* The exact original47 folds are read from the frozen CV manifest.
* All 44 labels acquired after original47 are train-only in every fold.
* External31 is never discovered, loaded, or evaluated by this script.

Examples
--------
  # Fastest useful first check: the historically difficult fold.
  python scripts/train_final91_stage1_roi_detector_cv.py \
    --config config_resize128.yaml --fold 2 --gpu 0

  # Complete the out-of-fold detector gate.
  python scripts/train_final91_stage1_roi_detector_cv.py \
    --config config_resize128.yaml --fold all --gpu 0
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
OUTPUT = Path("experiments/final91_stage1_roi_detector_cv")

EXPECTED_SOURCE = 47
EXPECTED_TOTAL = 91
EXPECTED_EXTRA = EXPECTED_TOTAL - EXPECTED_SOURCE


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
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
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def parse_fold(value):
    value = str(value).strip().lower()
    if value == "all":
        return list(range(5))
    fold = int(value)
    if fold not in range(5):
        raise ValueError("--fold must be all or 0..4")
    return [fold]


def median(values):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def detector_recipe(args):
    return {
        "version": "final91_stage1_expanded_box_detector_cv_v1",
        "purpose": "gated_stage1_roi_localizer_not_boundary_segmenter",
        "architecture": "MONAI_UNet_3D",
        "channels": [8, 16, 32, 64, 128],
        "strides": [2, 2, 2, 2],
        "num_res_units": 1,
        "resize_size": [args.resize_size] * 3,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "BCEWithLogits_plus_soft_Dice",
        "target_box_margin_fraction": args.target_box_margin,
        "detector_threshold": args.detector_threshold,
        "inference_crop_margin_fraction": args.inference_margin,
        "min_component_relative_size": args.min_component_relative_size,
        "minimum_component_voxels": args.min_component_voxels,
        "augmentation": {
            "translation_voxels": 4,
            "translation_probability": 0.5,
            "left_right_flip_probability": 0.5,
        },
        "validation_every_n_epochs": args.validation_every_n_epochs,
        "checkpoint_selection": "heldout_expanded_box_dice",
        "gate": {
            "maximum_cases_below_0p99_gt_crop_coverage": 0,
            "maximum_full_volume_fallbacks": 0,
            "minimum_mean_gt_crop_coverage": 0.995,
            "maximum_median_crop_fraction": 0.60,
        },
        "heldout_evaluation": "exact_frozen_original47",
        "external31_access": False,
    }


def dry_run(args):
    print("=" * 112)
    print("FINAL91 STAGE-1 ROI DETECTOR — DRY RUN")
    print(f"Folds:                    {parse_fold(args.fold)}")
    print(f"Grid:                     {args.resize_size}^3")
    print(f"Epochs:                   {args.epochs}")
    print(f"Validate every:           {args.validation_every_n_epochs} epochs")
    print(f"Target box margin:        {args.target_box_margin:.0%}")
    print(f"Detector threshold:       {args.detector_threshold:.2f}")
    print(f"Inference crop margin:    {args.inference_margin:.0%}")
    print("Held out:                 exact frozen original47 folds")
    print("Newer 44 HUMAN_GOLD:      train-only")
    print("External31:               NOT ACCESSED")
    print("Stage 2:                  BLOCKED until the full detector gate passes")
    print("=" * 112)


def run(args):
    # Heavy imports intentionally occur only for a real run, so --dry-run and --help work on
    # orchestration machines that do not have the CUDA/MONAI environment installed.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from monai.data import CacheDataset, DataLoader
    from monai.networks.nets import UNet
    from monai.transforms import Compose, MapTransform, RandAffined, RandFlipd
    from monai.utils import set_determinism
    from scipy import ndimage

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from scripts.audit_round1_labels import discover_round1_cases

    class ExpandedBoxTargetd(MapTransform):
        """Preserve bladder GT and replace the segmentation target with an expanded box."""

        def __init__(self, key="label", preserved_key="bladder_label", margin_fraction=0.15):
            super().__init__([key])
            self.key = key
            self.preserved_key = preserved_key
            self.margin_fraction = float(margin_fraction)

        def __call__(self, data):
            result = dict(data)
            label = result[self.key]
            bladder = (label > 0.5).to(dtype=label.dtype)
            result[self.preserved_key] = bladder.clone()
            spatial = bladder[0] if bladder.ndim == 4 else bladder
            coords = torch.nonzero(spatial > 0, as_tuple=False)
            if coords.numel() == 0:
                raise RuntimeError(f"Empty bladder label after preprocessing for {result.get('id', '?')}")
            lo = coords.min(dim=0).values
            hi = coords.max(dim=0).values
            extent = hi - lo + 1
            margin = torch.ceil(extent.float() * self.margin_fraction).long()
            shape = torch.as_tensor(spatial.shape, device=lo.device)
            lo = torch.maximum(lo - margin, torch.zeros_like(lo))
            hi = torch.minimum(hi + margin, shape - 1)
            box = torch.zeros_like(spatial)
            box[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1
            result[self.key] = box.unsqueeze(0) if bladder.ndim == 4 else box
            return result

    def make_transforms(config, training):
        base = data_engine.get_base_transforms(
            config, keys=["image", "label"], is_training=training, apply_strong_aug=False
        )
        steps = list(getattr(base, "transforms", [base]))
        if training:
            steps.extend([
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
        steps.append(ExpandedBoxTargetd(margin_fraction=args.target_box_margin))
        return Compose(steps)

    def make_loader(items, transform, training, config):
        # CacheDataset stops caching at the first random transform. Validation is therefore
        # fully resident in RAM; deterministic train preprocessing is cached as well.
        dataset = CacheDataset(
            data=items, transform=transform, cache_rate=1.0, copy_cache=False,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=training,
            num_workers=0 if not training else int(getattr(config, "num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )

    def build_model():
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=1,
            norm="INSTANCE",
        )

    def loss_value(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        denominator = prob.sum(dim=dims) + target.sum(dim=dims)
        soft_dice = ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
        return bce + (1.0 - soft_dice)

    def binary_metrics(pred, target):
        intersection = float((pred * target).sum().item())
        pred_sum = float(pred.sum().item())
        target_sum = float(target.sum().item())
        return {
            "dice": (2.0 * intersection + 1e-5) / (pred_sum + target_sum + 1e-5),
            "precision": (intersection + 1e-5) / (pred_sum + 1e-5),
            "recall": (intersection + 1e-5) / (target_sum + 1e-5),
        }

    @torch.no_grad()
    def validate_box(model, loader, device):
        model.eval()
        losses, dices, precisions, recalls = [], [], [], []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].float().to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(image)
                loss = loss_value(logits, target)
            pred = (torch.sigmoid(logits) >= args.detector_threshold).float()
            metrics = binary_metrics(pred, target)
            losses.append(float(loss.item()))
            dices.append(metrics["dice"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
        return {
            "loss": float(np.mean(losses)),
            "dice": float(np.mean(dices)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
        }

    def bounds_from_mask(mask):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        return coords.min(axis=0).astype(int), coords.max(axis=0).astype(int)

    def expand_bounds(lo, hi, shape, fraction):
        extent = hi - lo + 1
        margin = np.ceil(extent.astype(float) * float(fraction)).astype(int)
        lo = np.maximum(lo - margin, 0)
        hi = np.minimum(hi + margin, np.asarray(shape, dtype=int) - 1)
        return lo.astype(int), hi.astype(int)

    def component_crop(probability, threshold=None, margin=None, policy="substantial"):
        threshold = args.detector_threshold if threshold is None else float(threshold)
        margin = args.inference_margin if margin is None else float(margin)
        raw = probability >= threshold
        labeled, count = ndimage.label(raw)
        if count == 0:
            shape = np.asarray(raw.shape, dtype=int)
            return np.zeros(3, dtype=int), shape - 1, 0, 1
        sizes = np.bincount(labeled.ravel())[1:]
        ranked = np.argsort(sizes)[::-1] + 1
        if policy == "largest":
            kept_labels = ranked[:1]
        elif policy == "top2":
            kept_labels = ranked[:2]
        elif policy == "substantial":
            minimum = max(
                args.min_component_voxels,
                int(math.ceil(float(sizes.max()) * args.min_component_relative_size)),
            )
            kept_labels = np.where(sizes >= minimum)[0] + 1
        else:
            raise ValueError(f"Unknown component policy: {policy}")
        kept = np.isin(labeled, kept_labels)
        bounds = bounds_from_mask(kept)
        if bounds is None:
            shape = np.asarray(raw.shape, dtype=int)
            return np.zeros(3, dtype=int), shape - 1, int(count), 1
        lo, hi = bounds
        lo, hi = expand_bounds(lo, hi, raw.shape, margin)
        return lo, hi, int(count), 0

    @torch.no_grad()
    def evaluate_crop_gate(model, loader, device, fold):
        model.eval()
        rows = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target_box = batch["label"].float().to(device, non_blocking=True)
            bladder = batch["bladder_label"].float().to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(image))
            predicted_box = (probability >= args.detector_threshold).float()
            box_metrics = binary_metrics(predicted_box, target_box)

            prob_np = probability[0, 0].float().cpu().numpy()
            bladder_np = bladder[0, 0].cpu().numpy() > 0.5
            lo, hi, components, fallback = component_crop(prob_np)
            crop = np.zeros_like(bladder_np, dtype=bool)
            crop[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
            gt_voxels = int(bladder_np.sum())
            inside = int(np.logical_and(bladder_np, crop).sum())
            coverage = float(inside / max(gt_voxels, 1))
            crop_fraction = float(crop.mean())

            gt_bounds = bounds_from_mask(bladder_np)
            if gt_bounds is None:
                raise RuntimeError("A held-out bladder label became empty after preprocessing")
            gt_lo, gt_hi = gt_bounds
            gt_center = 0.5 * (gt_lo.astype(float) + gt_hi.astype(float))
            crop_center = 0.5 * (lo.astype(float) + hi.astype(float))
            center_error = float(np.linalg.norm(crop_center - gt_center))
            center_error_normalized = center_error / float(np.linalg.norm(np.asarray(bladder_np.shape)))

            case_value = batch["id"]
            case_id = str(case_value[0] if isinstance(case_value, (list, tuple)) else case_value)
            rows.append({
                "case_id": case_id,
                "fold": int(fold),
                "detector_box_dice": box_metrics["dice"],
                "detector_box_precision": box_metrics["precision"],
                "detector_box_recall": box_metrics["recall"],
                "gt_crop_coverage": coverage,
                "crop_fraction": crop_fraction,
                "center_error_voxels": center_error,
                "center_error_normalized": center_error_normalized,
                "detector_components": components,
                "full_volume_fallback": fallback,
                "gt_voxels": gt_voxels,
                "crop_z0": int(lo[0]), "crop_z1": int(hi[0]),
                "crop_y0": int(lo[1]), "crop_y1": int(hi[1]),
                "crop_x0": int(lo[2]), "crop_x1": int(hi[2]),
            })
        return rows

    @torch.no_grad()
    def sweep_crop_postprocessing(model, loader, device, fold):
        """Reuse one checkpoint to test compact crop rules without retraining or selection on External31."""
        thresholds = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
        margins = (0.10, 0.20, 0.30, 0.40)
        policies = ("substantial", "top2", "largest")
        model.eval()
        rows = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            bladder = batch["bladder_label"][0, 0].cpu().numpy() > 0.5
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(image))[0, 0].float().cpu().numpy()
            case_value = batch["id"]
            case_id = str(case_value[0] if isinstance(case_value, (list, tuple)) else case_value)
            gt_voxels = int(bladder.sum())
            for threshold in thresholds:
                for margin in margins:
                    for policy in policies:
                        lo, hi, components, fallback = component_crop(
                            probability,
                            threshold=threshold,
                            margin=margin,
                            policy=policy,
                        )
                        crop = np.zeros_like(bladder, dtype=bool)
                        crop[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
                        rows.append({
                            "case_id": case_id,
                            "fold": int(fold),
                            "threshold": threshold,
                            "margin": margin,
                            "component_policy": policy,
                            "gt_crop_coverage": float(np.logical_and(bladder, crop).sum() / max(gt_voxels, 1)),
                            "crop_fraction": float(crop.mean()),
                            "detector_components": components,
                            "full_volume_fallback": fallback,
                            "crop_z0": int(lo[0]), "crop_z1": int(hi[0]),
                            "crop_y0": int(lo[1]), "crop_y1": int(hi[1]),
                            "crop_x0": int(lo[2]), "crop_x1": int(hi[2]),
                        })
        return rows

    def preflight_dataset():
        config = HASSLConfig.from_yaml(args.config)
        config.preprocessing_mode = "resize"
        config.spatial_size = (int(args.resize_size),) * 3

        source_dir = Path(args.source_cv_dir)
        source_path = source_dir / "cv_splits.json"
        source_manifest = read_json(source_path)
        source_ids = sorted(str(x) for x in source_manifest.get("all_case_ids", []))
        if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
            raise RuntimeError("Source manifest is not the exact frozen original47")
        if len(source_manifest.get("folds", [])) != 5:
            raise RuntimeError("Source manifest must contain five folds")

        audit = read_json(args.audit_metadata)
        if not audit.get("all_visible_labels_passed_audit", False):
            raise RuntimeError("Final91 live-label audit is not passing")
        if not audit.get("selection_provenance_enforced", False):
            raise RuntimeError("Final91 audit does not enforce selection provenance")
        audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
        if len(audited_ids) != EXPECTED_TOTAL or len(set(audited_ids)) != EXPECTED_TOTAL:
            raise RuntimeError("Final91 audit must contain exactly 91 unique HUMAN_GOLD IDs")

        _, discovered_source, by_id, _ = discover_round1_cases(config, source_path)
        current_ids = sorted(str(x) for x in by_id)
        if current_ids != audited_ids:
            raise RuntimeError("Live labels changed after the passing Final91 audit")
        if sorted(str(x) for x in discovered_source) != source_ids:
            raise RuntimeError("Frozen source IDs changed during discovery")
        extra_ids = sorted(set(current_ids) - set(source_ids))
        if len(extra_ids) != EXPECTED_EXTRA:
            raise RuntimeError(f"Expected {EXPECTED_EXTRA} train-only cases, found {len(extra_ids)}")

        fold_specs = []
        held_out = []
        for original in source_manifest["folds"]:
            fold = int(original["fold"])
            val_ids = sorted(str(x) for x in original["val_ids"])
            train_ids = sorted(set(str(x) for x in original["train_ids"]) | set(extra_ids))
            if set(train_ids) & set(val_ids):
                raise RuntimeError(f"Fold {fold}: train/validation overlap")
            if set(extra_ids) & set(val_ids):
                raise RuntimeError(f"Fold {fold}: train-only case leaked into validation")
            fold_specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
            held_out.extend(val_ids)
        if sorted(held_out) != source_ids:
            raise RuntimeError("Each original47 case must be held out exactly once")
        return config, by_id, extra_ids, fold_specs, source_path

    def train_fold(config_template, by_id, spec, output_dir, device):
        fold = int(spec["fold"])
        fold_seed = args.seed + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        set_determinism(seed=fold_seed)

        fold_dir = output_dir / "checkpoints" / f"fold_{fold}"
        best_path = fold_dir / "best_detector.pth"
        if args.overwrite and fold_dir.exists():
            shutil.rmtree(fold_dir)
        if fold_dir.exists() and not best_path.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"Incomplete detector fold at {fold_dir}; rerun with --overwrite")
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.postprocess_sweep and not best_path.exists():
            raise FileNotFoundError(
                f"--postprocess-sweep requires an existing checkpoint: {best_path}"
            )

        val_t = make_transforms(config_template, training=False)
        val_loader = make_loader([by_id[x] for x in spec["val_ids"]], val_t, False, config_template)

        model = build_model().to(device)
        if not best_path.exists():
            train_t = make_transforms(config_template, training=True)
            train_loader = make_loader(
                [by_id[x] for x in spec["train_ids"]], train_t, True, config_template
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            best_dice = -1.0
            best_epoch = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                train_losses, train_dices, train_precisions, train_recalls = [], [], [], []
                for batch in train_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    target = batch["label"].float().to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                        logits = model(image)
                        loss = loss_value(logits, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    pred = (torch.sigmoid(logits.detach()) >= args.detector_threshold).float()
                    metrics = binary_metrics(pred, target)
                    train_losses.append(float(loss.item()))
                    train_dices.append(metrics["dice"])
                    train_precisions.append(metrics["precision"])
                    train_recalls.append(metrics["recall"])

                validate_now = epoch == 1 or epoch == args.epochs or epoch % args.validation_every_n_epochs == 0
                prefix = (
                    f"Fold {fold} | epoch {epoch:03d}/{args.epochs} | "
                    f"train loss={np.mean(train_losses):.4f} dice={np.mean(train_dices):.4f} "
                    f"precision={np.mean(train_precisions):.4f} recall={np.mean(train_recalls):.4f}"
                )
                if validate_now:
                    val = validate_box(model, val_loader, device)
                    print(
                        prefix
                        + f" | val loss={val['loss']:.4f} box_dice={val['dice']:.4f} "
                        + f"precision={val['precision']:.4f} recall={val['recall']:.4f}"
                    )
                    if val["dice"] > best_dice:
                        best_dice = val["dice"]
                        best_epoch = epoch
                        torch.save({
                            "model_state": model.state_dict(),
                            "fold": fold,
                            "epoch": epoch,
                            "best_val_box_dice": best_dice,
                            "recipe": detector_recipe(args),
                            "train_ids": spec["train_ids"],
                            "val_ids": spec["val_ids"],
                        }, best_path)
                else:
                    print(prefix + " | val=SKIPPED")
            print(f"Fold {fold}: best epoch={best_epoch}, val box Dice={best_dice:.4f}")
        else:
            print(f"Fold {fold}: reusing completed detector checkpoint {best_path}")

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        if checkpoint.get("val_ids") != spec["val_ids"]:
            raise RuntimeError(f"Fold {fold}: checkpoint validation IDs differ from frozen split")
        model.load_state_dict(checkpoint["model_state"])
        rows = evaluate_crop_gate(model, val_loader, device, fold)
        sweep_rows = (
            sweep_crop_postprocessing(model, val_loader, device, fold)
            if args.postprocess_sweep
            else []
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows, sweep_rows

    config, by_id, extra_ids, fold_specs, source_manifest_path = preflight_dataset()
    selected_folds = parse_fold(args.fold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        **detector_recipe(args),
        "config": str(args.config),
        "audit_metadata": str(args.audit_metadata),
        "source_manifest": str(source_manifest_path),
        "n_total_human_gold": EXPECTED_TOTAL,
        "n_frozen_heldout": EXPECTED_SOURCE,
        "n_train_only_extra": len(extra_ids),
        "train_only_extra_ids": extra_ids,
        "folds": fold_specs,
    }
    plan_path = output_dir / "detector_cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing detector plan differs: {plan_path}; use a fresh --output-dir")
    if not plan_path.exists():
        write_json(plan_path, plan)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-1 detector training requires CUDA")
    print("=" * 112)
    print("FINAL91 STAGE-1 EXPANDED-BOX ROI DETECTOR")
    print(f"Device:                    {device}")
    print(f"Running folds:             {selected_folds}")
    print(f"HUMAN_GOLD:                {EXPECTED_TOTAL}")
    print(f"Train-only newer labels:   {len(extra_ids)}")
    print(f"Detector grid:             {args.resize_size}^3")
    print(f"Epochs / validation:       {args.epochs} / every {args.validation_every_n_epochs}")
    print("External31:                NOT ACCESSED")
    print("=" * 112)

    fold_map = {int(x["fold"]): x for x in fold_specs}
    fresh_rows = []
    sweep_rows = []
    for fold in selected_folds:
        fold_rows, fold_sweep_rows = train_fold(
            config, by_id, fold_map[fold], output_dir, device
        )
        fresh_rows.extend(fold_rows)
        sweep_rows.extend(fold_sweep_rows)

    if args.postprocess_sweep:
        sweep_case_path = output_dir / "detector_postprocess_sweep_cases.csv"
        write_csv(sweep_case_path, sweep_rows)
        sweep_summary = []
        settings = sorted({
            (float(x["threshold"]), float(x["margin"]), str(x["component_policy"]))
            for x in sweep_rows
        })
        for threshold, margin, policy in settings:
            rows = [
                x for x in sweep_rows
                if float(x["threshold"]) == threshold
                and float(x["margin"]) == margin
                and str(x["component_policy"]) == policy
            ]
            coverage = [float(x["gt_crop_coverage"]) for x in rows]
            crop_fraction = [float(x["crop_fraction"]) for x in rows]
            misses = sum(x < 0.99 for x in coverage)
            fallbacks = sum(int(x["full_volume_fallback"]) for x in rows)
            candidate = (
                misses == 0
                and fallbacks == 0
                and float(np.mean(coverage)) >= 0.995
                and median(crop_fraction) <= 0.60
            )
            sweep_summary.append({
                "threshold": threshold,
                "margin": margin,
                "component_policy": policy,
                "n": len(rows),
                "mean_gt_crop_coverage": float(np.mean(coverage)),
                "minimum_gt_crop_coverage": float(np.min(coverage)),
                "crop_misses_below_0p99": misses,
                "median_crop_fraction": median(crop_fraction),
                "mean_crop_fraction": float(np.mean(crop_fraction)),
                "full_volume_fallbacks": fallbacks,
                "fold_gate_candidate": candidate,
            })
        sweep_summary.sort(key=lambda x: (
            not x["fold_gate_candidate"],
            int(x["crop_misses_below_0p99"]),
            -float(x["mean_gt_crop_coverage"]),
            float(x["median_crop_fraction"]),
        ))
        sweep_summary_path = output_dir / "detector_postprocess_sweep_summary.csv"
        write_csv(sweep_summary_path, sweep_summary)
        write_json(output_dir / "detector_postprocess_sweep_best.json", {
            "version": "final91_stage1_fold_postprocess_sweep_v1",
            "folds": selected_folds,
            "external31_access": False,
            "best_setting": sweep_summary[0],
            "passing_settings": [x for x in sweep_summary if x["fold_gate_candidate"]],
            "interpretation": (
                "A passing Fold-2 setting justifies completing Stage-1 OOF CV; "
                "it does not authorize Stage 2 until all five folds pass."
            ),
        })
        print("\n" + "=" * 112)
        print("CHECKPOINT-ONLY ROI POST-PROCESSING SWEEP — TOP SETTINGS")
        for row in sweep_summary[:10]:
            print(
                f"thr={row['threshold']:.2f} margin={row['margin']:.2f} "
                f"policy={row['component_policy']:<11} | "
                f"coverage mean/min={row['mean_gt_crop_coverage']:.4f}/{row['minimum_gt_crop_coverage']:.4f} | "
                f"misses={row['crop_misses_below_0p99']} | "
                f"median crop={row['median_crop_fraction']:.4f} | "
                f"candidate={row['fold_gate_candidate']}"
            )
        print(f"Summary: {sweep_summary_path}")
        print("=" * 112)

    metrics_path = output_dir / "detector_oof_metrics.csv"
    existing = read_csv(metrics_path)
    kept = [row for row in existing if int(row["fold"]) not in selected_folds]
    combined = kept + fresh_rows
    combined.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    write_csv(metrics_path, combined)

    fold_summaries = []
    for fold in sorted({int(row["fold"]) for row in combined}):
        rows = [row for row in combined if int(row["fold"]) == fold]
        fold_summaries.append({
            "fold": fold,
            "n": len(rows),
            "mean_detector_box_dice": float(np.mean([float(x["detector_box_dice"]) for x in rows])),
            "mean_gt_crop_coverage": float(np.mean([float(x["gt_crop_coverage"]) for x in rows])),
            "minimum_gt_crop_coverage": float(np.min([float(x["gt_crop_coverage"]) for x in rows])),
            "median_crop_fraction": median(float(x["crop_fraction"]) for x in rows),
            "crop_misses_below_0p99": sum(float(x["gt_crop_coverage"]) < 0.99 for x in rows),
            "full_volume_fallbacks": sum(int(float(x["full_volume_fallback"])) for x in rows),
        })
    write_csv(output_dir / "detector_fold_summary.csv", fold_summaries)

    completed_folds = {int(x["fold"]) for x in combined}
    case_ids = {str(x["case_id"]) for x in combined}
    full_oof = completed_folds == set(range(5)) and len(case_ids) == EXPECTED_SOURCE and len(combined) == EXPECTED_SOURCE
    coverage_values = [float(x["gt_crop_coverage"]) for x in combined]
    crop_values = [float(x["crop_fraction"]) for x in combined]
    misses = sum(x < 0.99 for x in coverage_values)
    fallbacks = sum(int(float(x["full_volume_fallback"])) for x in combined)
    summary = {
        "version": "final91_stage1_roi_detector_gate_v1",
        "complete_original47_oof": full_oof,
        "completed_folds": sorted(completed_folds),
        "n_cases": len(combined),
        "mean_detector_box_dice": float(np.mean([float(x["detector_box_dice"]) for x in combined])),
        "mean_gt_crop_coverage": float(np.mean(coverage_values)),
        "minimum_gt_crop_coverage": float(np.min(coverage_values)),
        "median_crop_fraction": median(crop_values),
        "mean_crop_fraction": float(np.mean(crop_values)),
        "crop_misses_below_0p99": misses,
        "full_volume_fallbacks": fallbacks,
        "mean_center_error_normalized": float(np.mean([float(x["center_error_normalized"]) for x in combined])),
        "criteria": detector_recipe(args)["gate"],
        "gate_pass": bool(
            full_oof
            and misses == 0
            and fallbacks == 0
            and float(np.mean(coverage_values)) >= 0.995
            and median(crop_values) <= 0.60
        ),
        "stage2_authorized": False,
        "folds": fold_summaries,
    }
    summary["stage2_authorized"] = summary["gate_pass"]
    write_json(output_dir / "detector_gate_summary.json", summary)

    print("\n" + "=" * 112)
    print("STAGE-1 ROI DETECTOR GATE")
    print(f"OOF complete:              {summary['complete_original47_oof']}")
    print(f"Cases / folds:             {summary['n_cases']} / {summary['completed_folds']}")
    print(f"Mean detector box Dice:    {summary['mean_detector_box_dice']:.4f}")
    print(f"Mean/min GT coverage:      {summary['mean_gt_crop_coverage']:.4f} / {summary['minimum_gt_crop_coverage']:.4f}")
    print(f"Crop misses (<0.99):       {summary['crop_misses_below_0p99']}")
    print(f"Median/mean crop fraction: {summary['median_crop_fraction']:.4f} / {summary['mean_crop_fraction']:.4f}")
    print(f"Full-volume fallbacks:     {summary['full_volume_fallbacks']}")
    print(f"GATE PASS:                 {summary['gate_pass']}")
    print(f"STAGE 2 AUTHORIZED:        {summary['stage2_authorized']}")
    if not full_oof:
        print("Stage 2 remains blocked until all five frozen folds are present.")
    print("=" * 112)
    print(f"Metrics: {metrics_path}")
    print(f"Gate:    {output_dir / 'detector_gate_summary.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description="Final91 true Stage-1 expanded-box ROI detector CV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--fold", default="all", help="all or 0..4")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; default 0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--resize-size", type=int, default=96)
    parser.add_argument("--validation-every-n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-box-margin", type=float, default=0.15)
    parser.add_argument("--detector-threshold", type=float, default=0.30)
    parser.add_argument("--inference-margin", type=float, default=0.40)
    parser.add_argument("--min-component-relative-size", type=float, default=0.05)
    parser.add_argument("--min-component-voxels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--postprocess-sweep",
        action="store_true",
        help="Reuse existing fold checkpoint and sweep threshold/margin/component policy without training",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(parser, args):
    if args.postprocess_sweep and args.overwrite:
        parser.error("--postprocess-sweep cannot be combined with --overwrite")
    if args.epochs < 1:
        parser.error("--epochs must be >=1")
    if args.resize_size < 32 or args.resize_size % 16:
        parser.error("--resize-size must be >=32 and divisible by 16")
    if args.validation_every_n_epochs < 1:
        parser.error("--validation-every-n-epochs must be >=1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be >0 and weight decay must be >=0")
    for name in ("target_box_margin", "detector_threshold", "inference_margin", "min_component_relative_size"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.detector_threshold <= 0:
        parser.error("--detector-threshold must be >0")
    if args.min_component_voxels < 1:
        parser.error("--min-component-voxels must be >=1")
    try:
        parse_fold(args.fold)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.dry_run:
        dry_run(args)
        return
    # Set this before importing torch so the logical CUDA device inside the process is cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
