#!/usr/bin/env python3
"""Train a true single-object 3D bladder bounding-box localizer on Final91.

Unlike the rejected mask-based ROI detector, this model directly predicts exactly six values:
normalized bladder center (z, y, x) and box size (d, h, w).  It cannot emit multiple connected
components.  A safety-expanded predicted box is evaluated on the exact frozen original47 OOF
folds before any Stage-2 segmenter is authorized.

All 44 post-original HUMAN_GOLD cases are train-only. External31 is never accessed.
"""

import argparse
import csv
import json
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
OUTPUT = Path("experiments/final91_stage1_direct_bbox_cv")
EXPECTED_SOURCE = 47
EXPECTED_TOTAL = 91
EXPECTED_EXTRA = 44


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_fold(value):
    value = str(value).strip().lower()
    if value == "all":
        return list(range(5))
    try:
        folds = sorted({int(x.strip()) for x in value.replace("+", ",").split(",") if x.strip()})
    except ValueError as exc:
        raise ValueError("--fold must be all, 0..4, or a comma-separated subset") from exc
    if not folds or any(x not in range(5) for x in folds):
        raise ValueError("--fold must be all, 0..4, or a comma-separated subset")
    return folds


def median(values):
    values = sorted(float(x) for x in values)
    if not values:
        return float("nan")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])


def recipe(args):
    return {
        "version": "final91_stage1_direct_bbox_cv_v1",
        "task": "single_object_direct_3d_bbox_regression",
        "prediction": "normalized_center_zyx_plus_size_dhw",
        "architecture": "coordinate_augmented_3d_encoder_spatial_fc",
        "input_channels": ["ultrasound", "z_coordinate", "y_coordinate", "x_coordinate"],
        "encoder_channels": [16, 32, 64, 128],
        "adaptive_feature_grid": [4, 4, 4],
        "resize_size": [args.resize_size] * 3,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": {
            "center_smooth_l1_weight": 2.0,
            "size_smooth_l1_weight": 1.0,
            "bbox_iou_loss_weight": 2.0,
        },
        "augmentation": {
            "translation_voxels": 4,
            "translation_probability": 0.5,
            "left_right_flip_probability": 0.5,
        },
        "safety_margin_per_side": args.safety_margin,
        "validation_every_n_epochs": args.validation_every_n_epochs,
        "checkpoint_selection": [
            "fewest_gt_crop_coverage_below_0p99",
            "highest_mean_gt_crop_coverage",
            "highest_mean_bbox_iou",
            "smallest_median_crop_fraction",
        ],
        "gate": {
            "maximum_crop_misses_below_0p99": 0,
            "minimum_mean_gt_crop_coverage": 0.995,
            "minimum_mean_bbox_iou": 0.20,
            "maximum_median_crop_fraction": 0.60,
        },
        "heldout_evaluation": "exact_frozen_original47",
        "external31_access": False,
    }


def dry_run(args):
    print("=" * 112)
    print("FINAL91 STAGE-1 DIRECT BOUNDING-BOX LOCALIZER — DRY RUN")
    print(f"Folds:                    {parse_fold(args.fold)}")
    print(f"Input:                    full image resized to {args.resize_size}^3")
    print("Output:                   one center (z,y,x) + one size (d,h,w)")
    print(f"Epochs:                   {args.epochs}")
    print(f"Validate every:           {args.validation_every_n_epochs} epochs")
    print(f"Safety margin per side:   {args.safety_margin:.0%}")
    print("Held out:                 exact frozen original47")
    print("Newer 44 HUMAN_GOLD:      train-only")
    print("External31:               NOT ACCESSED")
    print("Stage 2:                  BLOCKED until full direct-box gate passes")
    print("=" * 112)


def run(args):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import Compose, MapTransform, RandAffined, RandFlipd
    from monai.utils import set_determinism

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from scripts.audit_round1_labels import discover_round1_cases

    class BoundingBoxTargetd(MapTransform):
        def __init__(self, label_key="label", preserved_key="bladder_label"):
            super().__init__([label_key])
            self.label_key = label_key
            self.preserved_key = preserved_key

        def __call__(self, data):
            result = dict(data)
            label = (result[self.label_key] > 0.5).to(dtype=result[self.label_key].dtype)
            result[self.preserved_key] = label.clone()
            spatial = label[0] if label.ndim == 4 else label
            coords = torch.nonzero(spatial > 0, as_tuple=False)
            if coords.numel() == 0:
                raise RuntimeError(f"Empty bladder after preprocessing: {result.get('id', '?')}")
            lo = coords.min(dim=0).values.float()
            hi = coords.max(dim=0).values.float()
            shape = torch.as_tensor(spatial.shape, dtype=torch.float32, device=lo.device)
            center = 0.5 * (lo + hi) / torch.clamp(shape - 1.0, min=1.0)
            size = (hi - lo + 1.0) / shape
            result["bbox_center"] = center
            result["bbox_size"] = size
            return result

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            groups = min(8, out_channels)
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.block(x)

    class DirectBBoxRegressor3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                ConvBlock(4, 16, stride=2),
                ConvBlock(16, 32, stride=2),
                ConvBlock(32, 64, stride=2),
                ConvBlock(64, 128, stride=2),
            )
            self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
            self.shared = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4 * 4, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 128),
                nn.GELU(),
            )
            self.center_head = nn.Linear(128, 3)
            self.size_head = nn.Linear(128, 3)

        @staticmethod
        def coordinate_channels(image):
            batch, _, depth, height, width = image.shape
            z = torch.linspace(-1.0, 1.0, depth, device=image.device, dtype=image.dtype)
            y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            coordinates = torch.stack([zz, yy, xx], dim=0).unsqueeze(0)
            return coordinates.expand(batch, -1, -1, -1, -1)

        def forward(self, image):
            features = self.encoder(torch.cat([image, self.coordinate_channels(image)], dim=1))
            shared = self.shared(self.pool(features))
            center = torch.sigmoid(self.center_head(shared))
            size = 0.02 + 0.98 * torch.sigmoid(self.size_head(shared))
            return center, size

    def make_transforms(config, training):
        base = data_engine.get_base_transforms(
            config, keys=["image", "label"], is_training=training, apply_strong_aug=False
        )
        steps = list(getattr(base, "transforms", [base]))
        if training:
            steps.extend([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=["image", "label"], prob=0.5,
                    rotate_range=(0.0, 0.0, 0.0),
                    translate_range=(4.0, 4.0, 4.0),
                    scale_range=(0.0, 0.0, 0.0),
                    mode=("bilinear", "nearest"), padding_mode="zeros",
                ),
            ])
        steps.append(BoundingBoxTargetd())
        return Compose(steps)

    def make_loader(items, transform, training, config):
        workers = int(getattr(config, "num_workers", 0))
        dataset = CacheDataset(
            items, transform=transform, cache_rate=1.0, copy_cache=False, num_workers=workers
        )
        return DataLoader(
            dataset, batch_size=1, shuffle=training,
            num_workers=workers if training else 0,
            pin_memory=torch.cuda.is_available(),
        )

    def box_corners(center, size):
        return center - 0.5 * size, center + 0.5 * size

    def box_iou(center_a, size_a, center_b, size_b):
        lo_a, hi_a = box_corners(center_a, size_a)
        lo_b, hi_b = box_corners(center_b, size_b)
        intersection = torch.relu(torch.minimum(hi_a, hi_b) - torch.maximum(lo_a, lo_b)).prod(dim=1)
        volume_a = torch.clamp(size_a, min=1e-6).prod(dim=1)
        volume_b = torch.clamp(size_b, min=1e-6).prod(dim=1)
        union = volume_a + volume_b - intersection
        return intersection / torch.clamp(union, min=1e-6)

    def loss_value(pred_center, pred_size, target_center, target_size):
        center_loss = F.smooth_l1_loss(pred_center, target_center)
        size_loss = F.smooth_l1_loss(pred_size, target_size)
        iou = box_iou(pred_center, pred_size, target_center, target_size)
        iou_loss = 1.0 - iou.mean()
        total = 2.0 * center_loss + size_loss + 2.0 * iou_loss
        return total, center_loss, size_loss, iou_loss

    def proposed_crop(center, size, spatial_shape):
        shape = np.asarray(spatial_shape, dtype=float)
        center_vox = np.asarray(center, dtype=float) * np.maximum(shape - 1.0, 1.0)
        extent = np.asarray(size, dtype=float) * shape
        expanded = extent * (1.0 + 2.0 * float(args.safety_margin))
        lo = np.floor(center_vox - 0.5 * expanded).astype(int)
        hi = np.ceil(center_vox + 0.5 * expanded).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shape.astype(int) - 1)
        crop = np.zeros(tuple(int(x) for x in spatial_shape), dtype=bool)
        crop[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        return crop, lo, hi

    def sample_metrics(pred_center, pred_size, target_center, target_size, bladder, case_ids, fold):
        rows = []
        for index in range(pred_center.shape[0]):
            gt = bladder[index, 0].detach().cpu().numpy() > 0.5
            crop, lo, hi = proposed_crop(
                pred_center[index].detach().cpu().numpy(),
                pred_size[index].detach().cpu().numpy(),
                gt.shape,
            )
            coverage = float(np.logical_and(crop, gt).sum() / max(int(gt.sum()), 1))
            iou = float(box_iou(
                pred_center[index:index + 1], pred_size[index:index + 1],
                target_center[index:index + 1], target_size[index:index + 1],
            )[0].item())
            center_error = float(torch.linalg.norm(
                pred_center[index] - target_center[index]
            ).item())
            rows.append({
                "case_id": str(case_ids[index]), "fold": int(fold),
                "bbox_iou": iou,
                "center_error_normalized": center_error,
                "gt_crop_coverage": coverage,
                "crop_fraction": float(crop.mean()),
                "pred_center_z": float(pred_center[index, 0].item()),
                "pred_center_y": float(pred_center[index, 1].item()),
                "pred_center_x": float(pred_center[index, 2].item()),
                "pred_size_d": float(pred_size[index, 0].item()),
                "pred_size_h": float(pred_size[index, 1].item()),
                "pred_size_w": float(pred_size[index, 2].item()),
                "gt_center_z": float(target_center[index, 0].item()),
                "gt_center_y": float(target_center[index, 1].item()),
                "gt_center_x": float(target_center[index, 2].item()),
                "gt_size_d": float(target_size[index, 0].item()),
                "gt_size_h": float(target_size[index, 1].item()),
                "gt_size_w": float(target_size[index, 2].item()),
                "crop_z0": int(lo[0]), "crop_z1": int(hi[0]),
                "crop_y0": int(lo[1]), "crop_y1": int(hi[1]),
                "crop_x0": int(lo[2]), "crop_x1": int(hi[2]),
            })
        return rows

    def summarize(rows):
        coverage = [float(x["gt_crop_coverage"]) for x in rows]
        return {
            "n": len(rows),
            "mean_bbox_iou": float(np.mean([float(x["bbox_iou"]) for x in rows])),
            "mean_center_error_normalized": float(np.mean([float(x["center_error_normalized"]) for x in rows])),
            "mean_gt_crop_coverage": float(np.mean(coverage)),
            "minimum_gt_crop_coverage": float(np.min(coverage)),
            "crop_misses_below_0p99": int(sum(x < 0.99 for x in coverage)),
            "crop_miss_case_ids": sorted(str(x["case_id"]) for x in rows if float(x["gt_crop_coverage"]) < 0.99),
            "median_crop_fraction": median(float(x["crop_fraction"]) for x in rows),
            "mean_crop_fraction": float(np.mean([float(x["crop_fraction"]) for x in rows])),
        }

    @torch.no_grad()
    def evaluate(model, loader, device, fold):
        model.eval()
        rows = []
        losses = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target_center = batch["bbox_center"].float().to(device)
            target_size = batch["bbox_size"].float().to(device)
            bladder = batch["bladder_label"].float().to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                pred_center, pred_size = model(image)
                total, center_loss, size_loss, iou_loss = loss_value(
                    pred_center, pred_size, target_center, target_size
                )
            raw_ids = batch["id"]
            case_ids = list(raw_ids) if isinstance(raw_ids, (list, tuple)) else [str(raw_ids)]
            rows.extend(sample_metrics(
                pred_center, pred_size, target_center, target_size, bladder, case_ids, fold
            ))
            losses.append({
                "total": float(total.item()), "center": float(center_loss.item()),
                "size": float(size_loss.item()), "iou": float(iou_loss.item()),
            })
        summary = summarize(rows)
        summary.update({
            "loss": float(np.mean([x["total"] for x in losses])),
            "center_loss": float(np.mean([x["center"] for x in losses])),
            "size_loss": float(np.mean([x["size"] for x in losses])),
            "iou_loss": float(np.mean([x["iou"] for x in losses])),
        })
        return rows, summary

    def preflight():
        config = HASSLConfig.from_yaml(args.config)
        config.preprocessing_mode = "resize"
        config.spatial_size = (args.resize_size,) * 3
        source_path = Path(args.source_cv_dir) / "cv_splits.json"
        manifest = read_json(source_path)
        source_ids = sorted(str(x) for x in manifest.get("all_case_ids", []))
        if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
            raise RuntimeError("Source manifest is not frozen original47")
        audit = read_json(args.audit_metadata)
        if not audit.get("all_visible_labels_passed_audit", False) or not audit.get("selection_provenance_enforced", False):
            raise RuntimeError("Final91 audit must be passing and provenance-enforced")
        audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
        if len(audited_ids) != EXPECTED_TOTAL:
            raise RuntimeError("Final91 audit must contain exactly 91 labels")
        _, discovered_source, by_id, _ = discover_round1_cases(config, source_path)
        current_ids = sorted(str(x) for x in by_id)
        if current_ids != audited_ids or sorted(str(x) for x in discovered_source) != source_ids:
            raise RuntimeError("Live labels or frozen source changed after Final91 audit")
        extra_ids = sorted(set(current_ids) - set(source_ids))
        if len(extra_ids) != EXPECTED_EXTRA:
            raise RuntimeError("Expected exactly 44 train-only Final91 extras")
        specs, held_out = [], []
        for original in manifest.get("folds", []):
            fold = int(original["fold"])
            val_ids = sorted(str(x) for x in original["val_ids"])
            train_ids = sorted(set(str(x) for x in original["train_ids"]) | set(extra_ids))
            if set(train_ids) & set(val_ids) or set(extra_ids) & set(val_ids):
                raise RuntimeError(f"Fold {fold}: leakage detected")
            specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
            held_out.extend(val_ids)
        if sorted(held_out) != source_ids or len(specs) != 5:
            raise RuntimeError("Frozen folds do not cover original47 exactly once")
        return config, by_id, extra_ids, specs, source_path

    def train_fold(config, by_id, spec, output_dir, device):
        fold = int(spec["fold"])
        seed = args.seed + fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        set_determinism(seed=seed)
        fold_dir = output_dir / "checkpoints" / f"fold_{fold}"
        best_path = fold_dir / "best_bbox_localizer.pth"
        if args.overwrite and fold_dir.exists():
            shutil.rmtree(fold_dir)
        if fold_dir.exists() and not best_path.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"Incomplete fold at {fold_dir}; use --overwrite")
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fold {fold}: caching validation={len(spec['val_ids'])} cases")
        val_loader = make_loader(
            [by_id[x] for x in spec["val_ids"]], make_transforms(config, False), False, config
        )
        model = DirectBBoxRegressor3D().to(device)
        if not best_path.exists():
            print(f"Fold {fold}: caching training={len(spec['train_ids'])} cases")
            train_loader = make_loader(
                [by_id[x] for x in spec["train_ids"]], make_transforms(config, True), True, config
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
            )
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            best_key = None
            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss, train_iou, train_coverage = [], [], []
                for batch in train_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    target_center = batch["bbox_center"].float().to(device)
                    target_size = batch["bbox_size"].float().to(device)
                    bladder = batch["bladder_label"].float().to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=True):
                        pred_center, pred_size = model(image)
                        total, _, _, _ = loss_value(pred_center, pred_size, target_center, target_size)
                    scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
                    metrics = sample_metrics(
                        pred_center.detach(), pred_size.detach(), target_center, target_size,
                        bladder, ["train"], fold,
                    )[0]
                    train_loss.append(float(total.item()))
                    train_iou.append(float(metrics["bbox_iou"]))
                    train_coverage.append(float(metrics["gt_crop_coverage"]))
                scheduler.step()
                validate_now = epoch == 1 or epoch == args.epochs or epoch % args.validation_every_n_epochs == 0
                prefix = (
                    f"Fold {fold} | epoch {epoch:03d}/{args.epochs} | "
                    f"train loss={np.mean(train_loss):.4f} bbox_iou={np.mean(train_iou):.4f} "
                    f"crop_coverage={np.mean(train_coverage):.4f}"
                )
                if validate_now:
                    _, val = evaluate(model, val_loader, device, fold)
                    print(
                        prefix + f" | val loss={val['loss']:.4f} bbox_iou={val['mean_bbox_iou']:.4f} "
                        f"coverage={val['mean_gt_crop_coverage']:.4f} min={val['minimum_gt_crop_coverage']:.4f} "
                        f"misses={val['crop_misses_below_0p99']} median_crop={val['median_crop_fraction']:.4f}"
                    )
                    key = (
                        -int(val["crop_misses_below_0p99"]),
                        float(val["mean_gt_crop_coverage"]),
                        float(val["mean_bbox_iou"]),
                        -float(val["median_crop_fraction"]),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        torch.save({
                            "model_state": model.state_dict(), "fold": fold, "epoch": epoch,
                            "selection_key": list(key), "validation_summary": val,
                            "recipe": recipe(args), "train_ids": spec["train_ids"],
                            "val_ids": spec["val_ids"],
                        }, best_path)
                else:
                    print(prefix + " | val=SKIPPED")
        else:
            print(f"Fold {fold}: reusing {best_path}")
        state = torch.load(best_path, map_location=device, weights_only=False)
        if state.get("val_ids") != spec["val_ids"] or state.get("recipe") != recipe(args):
            raise RuntimeError(f"Fold {fold}: checkpoint provenance differs from current run")
        model.load_state_dict(state["model_state"])
        rows, summary = evaluate(model, val_loader, device, fold)
        print(
            f"Fold {fold} best epoch={state['epoch']} | misses={summary['crop_misses_below_0p99']} | "
            f"coverage={summary['mean_gt_crop_coverage']:.4f} | IoU={summary['mean_bbox_iou']:.4f}"
        )
        del model; torch.cuda.empty_cache()
        return rows

    config, by_id, extra_ids, specs, source_path = preflight()
    selected_folds = parse_fold(args.fold)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        **recipe(args), "config": args.config, "audit_metadata": args.audit_metadata,
        "source_manifest": str(source_path), "n_total_human_gold": 91,
        "n_train_only_extra": len(extra_ids), "train_only_extra_ids": extra_ids,
        "folds": specs,
    }
    plan_path = output_dir / "direct_bbox_cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing plan differs: {plan_path}; use a fresh --output-dir")
    if not plan_path.exists(): write_json(plan_path, plan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Direct bounding-box training requires CUDA")
    print("=" * 112)
    print("FINAL91 STAGE-1 DIRECT 3D BOUNDING-BOX LOCALIZER")
    print(f"Folds: {selected_folds} | grid={args.resize_size}^3 | epochs={args.epochs} | margin={args.safety_margin:.0%}")
    print("Output: exactly one normalized center + size | External31: NOT ACCESSED")
    print("=" * 112)
    spec_map = {int(x["fold"]): x for x in specs}
    fresh = []
    for fold in selected_folds:
        fresh.extend(train_fold(config, by_id, spec_map[fold], output_dir, device))
    metrics_path = output_dir / "direct_bbox_oof_metrics.csv"
    existing = read_csv(metrics_path)
    combined = [x for x in existing if int(x["fold"]) not in selected_folds] + fresh
    combined.sort(key=lambda x: (int(x["fold"]), str(x["case_id"])))
    write_csv(metrics_path, combined)
    fold_rows = []
    for fold in sorted({int(x["fold"]) for x in combined}):
        fold_summary = summarize([x for x in combined if int(x["fold"]) == fold])
        fold_summary["fold"] = fold
        fold_summary["crop_miss_case_ids"] = ";".join(fold_summary["crop_miss_case_ids"])
        fold_rows.append(fold_summary)
    write_csv(output_dir / "direct_bbox_fold_summary.csv", fold_rows)
    overall = summarize(combined)
    complete = (
        len(combined) == EXPECTED_SOURCE
        and len({str(x["case_id"]) for x in combined}) == EXPECTED_SOURCE
        and {int(x["fold"]) for x in combined} == set(range(5))
    )
    gate = recipe(args)["gate"]
    passed = bool(
        complete
        and overall["crop_misses_below_0p99"] <= gate["maximum_crop_misses_below_0p99"]
        and overall["mean_gt_crop_coverage"] >= gate["minimum_mean_gt_crop_coverage"]
        and overall["mean_bbox_iou"] >= gate["minimum_mean_bbox_iou"]
        and overall["median_crop_fraction"] <= gate["maximum_median_crop_fraction"]
    )
    summary = {
        "version": "final91_stage1_direct_bbox_gate_v1",
        "complete_original47_oof": complete,
        "completed_folds": sorted({int(x["fold"]) for x in combined}),
        **overall, "criteria": gate, "gate_pass": passed, "stage2_authorized": passed,
        "external31_access": False,
    }
    write_json(output_dir / "direct_bbox_gate_summary.json", summary)
    print("\n" + "=" * 112)
    print("DIRECT BOUNDING-BOX LOCALIZER GATE")
    print(f"OOF complete:              {complete}")
    print(f"Cases / folds:             {len(combined)} / {summary['completed_folds']}")
    print(f"Mean box IoU:              {overall['mean_bbox_iou']:.4f}")
    print(f"Mean/min GT coverage:      {overall['mean_gt_crop_coverage']:.4f} / {overall['minimum_gt_crop_coverage']:.4f}")
    print(f"Crop misses (<0.99):       {overall['crop_misses_below_0p99']}")
    print(f"Missed case IDs:           {', '.join(overall['crop_miss_case_ids']) or 'none'}")
    print(f"Median/mean crop fraction: {overall['median_crop_fraction']:.4f} / {overall['mean_crop_fraction']:.4f}")
    print(f"GATE PASS:                 {passed}")
    print(f"STAGE 2 AUTHORIZED:        {passed}")
    print("=" * 112)


def build_parser():
    parser = argparse.ArgumentParser(description="Final91 direct single-object 3D bbox localizer CV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--fold", default="2", help="all, 0..4, or comma-separated subset; default 2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--resize-size",
        type=int,
        default=128,
        help=(
            "Final padded cubic detector grid; default 128 matches Final91. The aspect-ratio-"
            "preserved anatomy may occupy fewer slices (for example 18x128x128) inside this grid."
        ),
    )
    parser.add_argument("--validation-every-n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--safety-margin", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser(); args = parser.parse_args()
    if args.epochs < 1 or args.validation_every_n_epochs < 1:
        parser.error("epochs and validation cadence must be >=1")
    if args.resize_size < 32 or args.resize_size % 16:
        parser.error("--resize-size must be >=32 and divisible by16")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be >0 and weight decay >=0")
    if not 0 <= args.safety_margin <= 1:
        parser.error("--safety-margin must be in [0,1]")
    try: parse_fold(args.fold)
    except ValueError as exc: parser.error(str(exc))
    if args.dry_run:
        dry_run(args); return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
