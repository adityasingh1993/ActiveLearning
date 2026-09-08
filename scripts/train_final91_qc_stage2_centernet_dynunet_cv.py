#!/usr/bin/env python3
"""Train Stage-2 DynUNet on CenterNet-localized bladder crops.

Stage 1 is frozen by consuming the gate-approved OOF crop coordinates written by
``train_final91_stage1_centernet3d_cv.py``. Training uses GT-derived crops with safe randomized
margins and jitter; held-out evaluation uses only the corresponding fold's actual CenterNet crop.
Every crop prediction is resized and pasted back into the full 128^3 model grid before metrics are
calculated. Raw and largest-connected-component results are both reported.

Optional robust-training flags add a GT-safe wide-crop mixture, fold-local small-bladder
oversampling, and mild ultrasound appearance augmentation. They require a fresh output directory;
with all options disabled, the historical locked recipe and checkpoint identity are unchanged.

By default, the live 91-label dataset is unchanged: the documented uncertain case 9435... is
excluded from training and scoring, leaving 90 QC-trainable and 46 frozen-source scoring cases.
The explicit Final136 mode instead trains with all 136 current cases, including 9435..., while
retaining leakage-free validation on the 42 currently available frozen-source OOF cases. The exact
four missing frozen IDs must match the dedicated Final136 audit. External31 is never accessed.
"""

import argparse
import csv
import hashlib
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

from scripts.stage2_cohort_contract import (
    final136_mode as is_final136_count,
    project_frozen_oof_folds,
    resolve_stage2_cohort,
)

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
STAGE1_DIR = Path("experiments/final91_qc_stage1_centernet3d_cv")
QC90_STAGE1_DIR = Path("experiments/final91_label_qc90_centernet3d_oof")
FULL_VOLUME_BASELINE = Path("experiments/round5_cv_91_a3/cv_results.csv")
OUTPUT = Path("experiments/final91_qc_stage2_centernet_dynunet_cv")
QC90_OUTPUT = Path("experiments/final91_label_qc90_two_stage_oof")

EXPECTED_LIVE = 91
EXPECTED_SOURCE = 47
EXPECTED_EXTRA = 44
EXPECTED_QC_TRAINABLE = 90
EXPECTED_SCORABLE = 46
QUARANTINED_CASE_IDS = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c",
)


def qc90_mode(args):
    return str(getattr(args, "split_scope", "original46")) == "qc90"


def final136_mode(args):
    return is_final136_count(getattr(args, "expected_live", EXPECTED_LIVE))


def resolved_stage1_dir(args):
    value = Path(args.stage1_dir)
    return QC90_STAGE1_DIR if qc90_mode(args) and value == STAGE1_DIR else value


def resolved_output_dir(args):
    value = Path(args.output_dir)
    return QC90_OUTPUT if qc90_mode(args) and value == OUTPUT else value


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def mean(values):
    values = [float(x) for x in values]
    finite = [x for x in values if math.isfinite(x)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def median(values):
    values = sorted(float(x) for x in values)
    if not values:
        return float("nan")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])


def stage2_training_options(args):
    from hassl.data.stage2_training import Stage2TrainingOptions

    return Stage2TrainingOptions(
        output_size=int(args.crop_size),
        margin_min=float(args.train_margin_min),
        margin_max=float(args.train_margin_max),
        center_jitter=float(args.train_center_jitter),
        wide_crop_probability=float(getattr(args, "train_wide_crop_probability", 0.0)),
        wide_margin_min=float(getattr(args, "train_wide_margin_min", 0.60)),
        wide_margin_max=float(getattr(args, "train_wide_margin_max", 1.00)),
        wide_center_jitter=float(getattr(args, "train_wide_center_jitter", 0.20)),
        small_bladder_quantile=float(getattr(args, "small_bladder_quantile", 1.0 / 3.0)),
        small_bladder_weight=float(getattr(args, "small_bladder_weight", 1.0)),
        mild_appearance_augmentation=bool(
            getattr(args, "mild_appearance_augmentation", False)
        ),
    ).validate()


def recipe(args):
    from hassl.data.stage2_training import add_training_recipe_extensions

    training_options = stage2_training_options(args)
    result = {
        "version": "final91_qc_stage2_centernet_dynunet_cv_v1",
        "architecture": "DynUNet",
        "input": "CenterNet-localized crop resized to 128^3",
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "ema_decay": float(args.ema_decay),
        "loss": "DiceCE",
        "deep_supervision": True,
        "prediction": "Student+EMA 50/50 ensemble",
        "threshold": float(args.threshold),
        "crop_size": [int(args.crop_size)] * 3,
        "training_crop": {
            "source": "GT bounding box only",
            "margin_per_side_range": [float(args.train_margin_min), float(args.train_margin_max)],
            "center_jitter_fraction_of_gt_size": float(args.train_center_jitter),
            "guarantee": "crop unioned with GT bounds so foreground is never cut",
        },
        "validation_crop": {
            "source": "frozen Stage-1 OOF CenterNet coordinates",
            "stage1_margin_per_side": 0.50,
            "stage1_gate_required": True,
        },
        "augmentation_after_crop": {
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip_probability": 0.5,
            "lr_flip_axis_after_ras": 0,
        },
        "validation_every_n_epochs": int(args.validation_every_n_epochs),
        "checkpoint_selection": "highest full-grid raw ensemble Dice; LCC Dice tie-break",
        "evaluation_grid": "full 128^3 model space after paste-back",
        "reported_postprocessing": ["raw", "largest_6_connected_component"],
        "quarantined_case_ids": list(QUARANTINED_CASE_IDS),
        "external31_access": False,
    }
    if qc90_mode(args):
        result.update({
            "version": "final91_label_qc90_stage2_dynunet_oof_v1",
            "label_qc_only": True,
            "heldout_evaluation": "all_90_qc_clean_cases_exactly_once",
            "validation_crop": {
                "source": "matching-fold OOF CenterNet coordinates",
                "stage1_margin_per_side": 0.50,
                "stage1_complete_qc90_oof_required": True,
                "gt_aware_diagnostic_fallback": (
                    "full 128^3 grid only when OOF detector crop covers <99% of GT; "
                    "fallback is recorded per case and is not a deployable inference rule"
                ),
            },
        })
    if final136_mode(args):
        result.update({
            "version": "final136_stage2_centernet_dynunet_cv_v1",
            "dataset_scope": "all_136_current_training_cases",
            "heldout_evaluation": (
                "all currently available frozen-original47 QC cases exactly once"
            ),
            "quarantined_case_ids": [],
            "intentionally_included_prior_quarantine_ids": list(QUARANTINED_CASE_IDS),
            "missing_frozen_policy": "exact approved four-case allowlist",
        })
    if training_options.robust_enabled:
        if final136_mode(args):
            result["version"] = "final136_stage2_centernet_dynunet_cv_robust_v1"
        elif qc90_mode(args):
            result["version"] = "final91_label_qc90_stage2_dynunet_oof_robust_v1"
        else:
            result["version"] = "final91_qc_stage2_centernet_dynunet_cv_robust_v1"
        result["training_profile"] = "oversegmentation_robust_v1"
        result = add_training_recipe_extensions(result, training_options)
    return result


def dry_run(args):
    print("=" * 120)
    if final136_mode(args):
        print("FINAL136 STAGE-2 CENTERNET -> DYNUNET — DRY RUN")
    else:
        print(
            "FINAL91 LABEL-QC90 STAGE-2 OOF — DRY RUN"
            if qc90_mode(args) else "FINAL91-QC STAGE-2 CENTERNET -> DYNUNET — DRY RUN"
        )
    print(f"Folds:                    {parse_fold(args.fold)}")
    if final136_mode(args):
        print("Live / trainable:         136 / 136")
        print("OOF scorable:             42 available original-QC cases")
        print("Per fold:                 train=136 minus usable frozen validation subset")
        print("Prior 9435 quarantine:    INCLUDED as train-only")
        print("Missing frozen:           exact approved four-case allowlist")
    else:
        print("Live / QC trainable:      91 / 90")
        print(f"OOF scorable:             {90 if qc90_mode(args) else 46}")
    if qc90_mode(args):
        print("Per fold:                 train=72 / validation=18")
        print("Crop miss policy:         record detector miss + use full grid for QC segmentation")
        print("Role:                     annotation diagnosis; not final model validation")
    if not final136_mode(args):
        print(f"Quarantined:              {QUARANTINED_CASE_IDS[0]}")
    print(f"Frozen Stage-1:           {resolved_stage1_dir(args)}")
    print(
        f"Training crop:            GT box + random {args.train_margin_min:.0%}-"
        f"{args.train_margin_max:.0%} margin + safe {args.train_center_jitter:.0%} center jitter"
    )
    if args.train_wide_crop_probability > 0.0:
        print(
            f"Wide-crop mixture:        p={args.train_wide_crop_probability:.0%}; "
            f"margin={args.train_wide_margin_min:.0%}-{args.train_wide_margin_max:.0%}; "
            f"jitter={args.train_wide_center_jitter:.0%}"
        )
    if args.small_bladder_weight > 1.0:
        print(
            f"Small-bladder sampling:   bottom {args.small_bladder_quantile:.0%} at "
            f"{args.small_bladder_weight:.2f}x weight"
        )
    print(
        "Mild appearance aug:      "
        + ("contrast + intensity + speckle" if args.mild_appearance_augmentation else "OFF")
    )
    print("Validation crop:          actual frozen OOF CenterNet coordinates")
    print(f"Crop network input:       {args.crop_size}^3")
    print(f"Epochs / validation:      {args.epochs} / every {args.validation_every_n_epochs}")
    print("Segmentation recipe:      DynUNet + DiceCE + AdamW + Student/EMA")
    print("Evaluation:               paste to full 128^3; report RAW and LCC")
    print("External31:               NOT ACCESSED")
    print("=" * 120)


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from monai.data import CacheDataset, DataLoader
    from monai.metrics import HausdorffDistanceMetric
    from monai.transforms import Compose, MapTransform
    from monai.utils import set_determinism

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("Stage-2 RAW/LCC evaluation requires scipy") from exc

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from hassl.data.stage2_training import (
        build_small_bladder_sampler,
        build_stage2_training_transform,
        crop_and_resize_pair,
        plain_tensor,
    )
    from hassl.training.ema import EMATeacher
    from hassl.training.losses import CombinedSegLoss
    from hassl.training.trainer import build_network, compute_multiscale_loss
    from scripts.audit_round1_labels import discover_round1_cases

    def spacing_from_image(image, fallback):
        try:
            affine = image.affine
            if affine.ndim == 3:
                affine = affine[0]
            return torch.tensor([
                float(torch.linalg.norm(affine[:3, index]).item()) for index in range(3)
            ], dtype=torch.float32)
        except Exception:
            return torch.tensor([float(x) for x in fallback], dtype=torch.float32)

    class FixedCenterNetCropd(MapTransform):
        def __init__(self, crop_rows, output_size, fallback_spacing):
            super().__init__(["image", "label"])
            self.crop_rows = crop_rows
            self.output_size = int(output_size)
            fallback_spacing = fallback_spacing or (1.0, 1.0, 1.0)
            self.fallback_spacing = tuple(float(x) for x in fallback_spacing)

        def __call__(self, data):
            result = dict(data)
            case_id = str(result.get("id", ""))
            if case_id not in self.crop_rows:
                raise RuntimeError(f"No frozen CenterNet crop for validation case {case_id}")
            row = self.crop_rows[case_id]
            bounds = tuple(int(float(row[key])) for key in (
                "crop_z0", "crop_z1", "crop_y0", "crop_y1", "crop_x0", "crop_x1"
            ))
            label = plain_tensor(result["label"]).float()
            result["full_label"] = label.clone()
            result["full_spacing"] = spacing_from_image(
                result["image"], self.fallback_spacing
            )
            result["crop_bounds"] = torch.tensor(bounds, dtype=torch.int64)
            result["image"], result["label"] = crop_and_resize_pair(
                result["image"], result["label"], bounds, self.output_size
            )
            return result

    def make_transforms(config, training, crop_rows):
        base = data_engine.get_base_transforms(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )
        if training:
            return build_stage2_training_transform(base, stage2_training_options(args))
        steps = list(getattr(base, "transforms", [base]))
        steps.append(FixedCenterNetCropd(crop_rows, args.crop_size, config.spacing))
        return Compose(steps)

    def make_loader(items, transform, training, config, sampler=None):
        workers = int(getattr(config, "num_workers", 0))
        dataset = CacheDataset(
            items, transform=transform, cache_rate=1.0,
            copy_cache=False, num_workers=workers,
        )
        return DataLoader(
            dataset, batch_size=1, shuffle=bool(training and sampler is None), sampler=sampler,
            num_workers=workers if training else 0,
            pin_memory=torch.cuda.is_available(),
        )

    def main_prediction(output):
        if isinstance(output, (list, tuple)):
            return output[0]
        if torch.is_tensor(output) and output.ndim == 6:
            return output[:, 0]
        return output

    def keep_largest_component(mask):
        mask = np.asarray(mask).astype(bool, copy=False)
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
        labels, count = ndimage.label(mask, structure=structure)
        if count == 0:
            return mask.astype(np.uint8), 0
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        largest = int(np.argmax(sizes)) + 1
        return (labels == largest).astype(np.uint8), int(count)

    def binary_metrics(pred, target, spacing):
        pred = pred.float()
        target = target.float()
        tp = float((pred * target).sum().item())
        pred_sum = float(pred.sum().item())
        gt_sum = float(target.sum().item())
        eps = 1e-5
        hd95 = float("nan")
        try:
            metric = HausdorffDistanceMetric(
                include_background=True, percentile=95, reduction="mean"
            )
            metric(y_pred=pred, y=target, spacing=tuple(float(x) for x in spacing))
            value = float(metric.aggregate().item())
            if math.isfinite(value):
                hd95 = value
        except Exception:
            pass
        signed_rve = (pred_sum - gt_sum) / (gt_sum + 1e-8) * 100.0
        return {
            "dice": (2.0 * tp + eps) / (pred_sum + gt_sum + eps),
            "precision": (tp + eps) / (pred_sum + eps),
            "recall": (tp + eps) / (gt_sum + eps),
            "signed_rve": signed_rve,
            "abs_rve": abs(signed_rve),
            "hd95": hd95,
            "gt_vox": gt_sum,
            "pred_vox": pred_sum,
        }

    def summarize(rows, prefix):
        return {
            "n": len(rows),
            "mean_dice": mean(row[f"{prefix}_dice"] for row in rows),
            "median_dice": median(row[f"{prefix}_dice"] for row in rows),
            "mean_precision": mean(row[f"{prefix}_precision"] for row in rows),
            "mean_recall": mean(row[f"{prefix}_recall"] for row in rows),
            "median_signed_rve_pct": median(row[f"{prefix}_signed_rve"] for row in rows),
            "median_abs_rve_pct": median(row[f"{prefix}_abs_rve"] for row in rows),
            "mean_hd95": mean(row[f"{prefix}_hd95"] for row in rows),
            "dice_lt_0p50": sum(float(row[f"{prefix}_dice"]) < 0.50 for row in rows),
            "dice_lt_0p70": sum(float(row[f"{prefix}_dice"]) < 0.70 for row in rows),
            "dice_ge_0p80": sum(float(row[f"{prefix}_dice"]) >= 0.80 for row in rows),
        }

    def paste_probability(probability, full_shape, bounds):
        z0, z1, y0, y1, x0, x1 = [int(x) for x in bounds]
        crop_shape = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
        resized = F.interpolate(
            probability.float(), size=crop_shape, mode="trilinear", align_corners=False
        )
        full = torch.zeros(
            (probability.shape[0], probability.shape[1], *full_shape),
            device=probability.device, dtype=probability.dtype,
        )
        full[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = resized
        return full

    @torch.no_grad()
    def evaluate(student, teacher, loader, device, fold, crop_rows, prediction_dir=None):
        student.eval()
        teacher.eval()
        rows = []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            full_target = batch["full_label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            bounds = [int(x) for x in batch["crop_bounds"][0].tolist()]
            spacing = [float(x) for x in batch["full_spacing"][0].tolist()]
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob = torch.sigmoid(main_prediction(student(image)))
                teacher_prob = torch.sigmoid(main_prediction(teacher(image)))
                probability = 0.5 * (student_prob + teacher_prob)
            full_probability = paste_probability(
                probability, tuple(int(x) for x in full_target.shape[-3:]), bounds
            )
            raw = (full_probability > float(args.threshold)).float()
            raw_np = raw[0, 0].detach().cpu().numpy().astype(np.uint8)
            lcc_np, component_count = keep_largest_component(raw_np)
            lcc = torch.from_numpy(lcc_np[None, None].astype(np.float32)).to(device)
            raw_metrics = binary_metrics(raw, full_target, spacing)
            lcc_metrics = binary_metrics(lcc, full_target, spacing)
            stage1 = crop_rows[case_id]
            row = {
                "fold": int(fold), "case_id": case_id,
                "component_count_raw": int(component_count),
                "stage1_bbox_iou": float(stage1["bbox_iou"]),
                "stage1_gt_crop_coverage": float(stage1["gt_crop_coverage"]),
                "stage1_crop_fraction": float(stage1["crop_fraction"]),
                "stage1_center_confidence": float(stage1["center_confidence"]),
                "stage1_center_peak_margin": float(stage1.get("center_peak_margin", 0.0)),
                "qc_full_grid_fallback": int(stage1.get("qc_full_grid_fallback", 0)),
                "crop_z0": bounds[0], "crop_z1": bounds[1],
                "crop_y0": bounds[2], "crop_y1": bounds[3],
                "crop_x0": bounds[4], "crop_x1": bounds[5],
            }
            for prefix, metrics in (("raw", raw_metrics), ("lcc", lcc_metrics)):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            row["lcc_delta_dice"] = row["lcc_dice"] - row["raw_dice"]
            rows.append(row)
            if prediction_dir is not None:
                prediction_dir = Path(prediction_dir)
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prediction_dir / f"{case_id}.npz",
                    raw=raw_np.astype(np.uint8),
                    lcc=lcc_np.astype(np.uint8),
                    fold=np.asarray([int(fold)], dtype=np.int16),
                )
        return rows, summarize(rows, "raw"), summarize(rows, "lcc")

    def preflight():
        config = HASSLConfig.from_yaml(args.config)
        if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
            raise RuntimeError("Stage 2 requires prototype DynUNet Student+EMA")
        if int(config.num_classes) != 1:
            raise RuntimeError("Stage 2 requires binary num_classes=1")
        config.preprocessing_mode = "resize"
        config.spatial_size = (128, 128, 128)
        config.loss_type = "dice_ce"
        config.include_boundary = False
        config.dropout = 0.0

        source_path = Path(args.source_cv_dir) / "cv_splits.json"
        manifest = read_json(source_path)
        source_ids = sorted(str(x) for x in manifest.get("all_case_ids", []))
        if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
            raise RuntimeError("Source manifest is not frozen original47")

        audit = read_json(args.audit_metadata)
        _, discovered_source, by_id, _ = discover_round1_cases(
            config,
            source_path,
            require_all_frozen=not args.allow_missing_frozen,
        )
        current_ids = sorted(str(x) for x in by_id)
        if sorted(str(x) for x in discovered_source) != source_ids:
            raise RuntimeError("Frozen original47 source manifest changed")
        cohort = resolve_stage2_cohort(
            audit,
            source_ids,
            current_ids,
            expected_live=args.expected_live,
            expected_trainable=args.expected_trainable,
            include_quarantined=args.include_quarantined,
            allow_missing_frozen=args.allow_missing_frozen,
            split_scope=args.split_scope,
        )
        training_ids = cohort["training_ids"]
        quarantine = cohort["training_quarantined_case_ids"]
        stage1_quarantine = cohort["stage1_quarantined_case_ids"]
        extras = cohort["extra_ids"]
        scorable = (
            training_ids if qc90_mode(args) else cohort["scorable_ids"]
        )

        stage1_dir = resolved_stage1_dir(args)
        stage1_gate = read_json(stage1_dir / "centernet3d_gate_summary.json")
        if not stage1_gate.get("stage2_authorized", False):
            raise RuntimeError("Stage-1 CenterNet OOF has not authorized Stage 2")
        if qc90_mode(args):
            if not stage1_gate.get("complete_qc90_oof", False):
                raise RuntimeError("Stage-1 is not complete for all 90 QC-clean labels")
            if stage1_gate.get("split_scope") != "qc90":
                raise RuntimeError("Stage-1 split scope is not qc90")
        else:
            if not stage1_gate.get("gate_pass", False):
                raise RuntimeError("Stage-1 CenterNet safety gate has not passed")
            if not stage1_gate.get("complete_original46_qc_oof", False):
                raise RuntimeError("Stage-1 gate is not complete for original46-QC")
        if sorted(stage1_gate.get("quarantined_case_ids", [])) != stage1_quarantine:
            raise RuntimeError("Stage-1 quarantine provenance differs")
        stage1_plan = read_json(stage1_dir / "centernet3d_cv_plan.json")
        if stage1_plan.get("resize_size") != [128, 128, 128]:
            raise RuntimeError("Stage-1 did not use the frozen 128^3 grid")
        if abs(float(stage1_plan.get("safety_margin_per_side", -1)) - 0.50) > 1e-8:
            raise RuntimeError("Stage-1 did not use the locked 50% safety margin")

        crop_rows_list = read_csv(stage1_dir / "centernet3d_oof_metrics.csv")
        all_crop_rows = {str(row["case_id"]): dict(row) for row in crop_rows_list}
        expected_stage1_oof = sorted(set(source_ids) - set(stage1_quarantine))
        if qc90_mode(args):
            expected_stage1_oof = scorable
        if (
            len(crop_rows_list) != len(expected_stage1_oof)
            or set(all_crop_rows) != set(expected_stage1_oof)
        ):
            raise RuntimeError("Stage-1 OOF crop table differs from its frozen population")
        crop_rows = {case_id: all_crop_rows[case_id] for case_id in scorable}
        if set(crop_rows) & set(stage1_quarantine):
            raise RuntimeError("Quarantined case appears in Stage-1 crop table")
        for case_id, row in crop_rows.items():
            coverage = float(row["gt_crop_coverage"])
            row["qc_full_grid_fallback"] = int(qc90_mode(args) and coverage < 0.99)
            if coverage < 0.99 and not qc90_mode(args):
                raise RuntimeError(f"Unsafe Stage-1 crop for {case_id}")
            if int(row["qc_full_grid_fallback"]):
                for key, value in {
                    "crop_z0": 0, "crop_z1": 127, "crop_y0": 0,
                    "crop_y1": 127, "crop_x0": 0, "crop_x1": 127,
                }.items():
                    row[f"predicted_{key}"] = row[key]
                    row[key] = value
            bounds = [int(float(row[key])) for key in (
                "crop_z0", "crop_z1", "crop_y0", "crop_y1", "crop_x0", "crop_x1"
            )]
            for lo, hi in ((bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5])):
                if lo < 0 or hi > 127 or lo > hi:
                    raise RuntimeError(f"Invalid Stage-1 crop bounds for {case_id}: {bounds}")

        fold_specs, held_out = [], []
        checkpoint_provenance = []
        if qc90_mode(args):
            projected_specs = [
                {
                    "fold": int(original["fold"]),
                    "train_ids": sorted(str(x) for x in original["train_ids"]),
                    "val_ids": sorted(str(x) for x in original["val_ids"]),
                }
                for original in stage1_plan.get("folds", [])
            ]
            checkpoint_val_ids = {
                int(spec["fold"]): spec["val_ids"] for spec in projected_specs
            }
        else:
            projected_specs, checkpoint_val_ids = project_frozen_oof_folds(
                manifest.get("folds", []),
                training_ids,
                scorable,
                stage1_quarantine,
            )
        for projected in projected_specs:
            fold = int(projected["fold"])
            val_ids = projected["val_ids"]
            train_ids = projected["train_ids"]
            if set(train_ids) & set(val_ids) or set(quarantine) & (set(train_ids) | set(val_ids)):
                raise RuntimeError(f"Fold {fold}: leakage or quarantine violation")
            if any(int(float(crop_rows[x]["fold"])) != fold for x in val_ids):
                raise RuntimeError(f"Fold {fold}: Stage-1 crop fold mismatch")
            checkpoint = stage1_dir / "checkpoints" / f"fold_{fold}" / "best_centernet3d.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if sorted(str(x) for x in state.get("val_ids", [])) != checkpoint_val_ids[fold]:
                raise RuntimeError(f"Fold {fold}: CenterNet checkpoint val IDs differ")
            checkpoint_record = {
                "fold": fold, "checkpoint": str(checkpoint),
                "selected_epoch": int(state.get("epoch", -1)),
                "validation_summary": state.get("validation_summary", {}),
            }
            if final136_mode(args):
                checkpoint_record["usable_validation_ids"] = val_ids
            checkpoint_provenance.append(checkpoint_record)
            fold_specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
            held_out.extend(val_ids)
        if sorted(held_out) != scorable or len(fold_specs) != 5:
            raise RuntimeError("Stage-2 folds do not cover the requested QC population exactly once")

        baseline = {}
        if not qc90_mode(args):
            baseline_rows = read_csv(args.full_volume_baseline)
            baseline = {str(row["case_id"]): row for row in baseline_rows}
            if not set(scorable).issubset(baseline):
                raise RuntimeError("Full-volume Final91 baseline is missing a comparison case")
            for spec in fold_specs:
                for case_id in spec["val_ids"]:
                    if int(float(baseline[case_id]["fold"])) != int(spec["fold"]):
                        raise RuntimeError(f"Baseline fold mismatch for {case_id}")
        return (
            config, by_id, extras, quarantine, scorable, fold_specs,
            crop_rows, checkpoint_provenance, baseline, source_path, cohort,
        )

    def train_metrics(logits, target):
        probability = torch.sigmoid(main_prediction(logits).detach())
        pred = (probability > float(args.threshold)).float()
        target = (target > 0.5).float()
        reduce_dims = tuple(range(1, pred.ndim))
        tp = (pred * target).sum(dim=reduce_dims)
        pred_sum = pred.sum(dim=reduce_dims)
        gt_sum = target.sum(dim=reduce_dims)
        return {
            "dice": float(((2 * tp) / (pred_sum + gt_sum + 1e-8)).mean().item()),
            "precision": float((tp / (pred_sum + 1e-8)).mean().item()),
            "recall": float((tp / (gt_sum + 1e-8)).mean().item()),
        }

    def fold_state(fold_dir):
        if args.overwrite and fold_dir.exists():
            shutil.rmtree(fold_dir)
        marker = fold_dir / "training_complete.json"
        best = fold_dir / "best_checkpoint.pth"
        if marker.exists() and best.exists():
            return "complete"
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"Incomplete Stage-2 fold at {fold_dir}; use --overwrite")
        fold_dir.mkdir(parents=True, exist_ok=True)
        return "new"

    def train_fold(config, by_id, spec, crop_rows, output_dir, device):
        fold = int(spec["fold"])
        seed = int(args.seed) + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_determinism(seed=seed)
        fold_dir = output_dir / "checkpoints" / f"fold_{fold}"
        state_name = fold_state(fold_dir)
        best_path = fold_dir / "best_checkpoint.pth"
        print("\n" + "#" * 120)
        print(f"STAGE-2 FOLD {fold} | train={len(spec['train_ids'])} | validation={len(spec['val_ids'])}")
        print("Train crops: safe randomized GT boxes | Validation crops: frozen OOF CenterNet boxes")
        print(f"State: {state_name} | checkpoint: {best_path}")
        print("#" * 120)

        val_loader = make_loader(
            [by_id[x] for x in spec["val_ids"]],
            make_transforms(config, False, crop_rows), False, config,
        )
        if state_name == "new":
            train_items = [by_id[x] for x in spec["train_ids"]]
            sampler, sampling_audit = build_small_bladder_sampler(
                train_items, stage2_training_options(args), seed=seed + 10000
            )
            write_json(fold_dir / "training_sampling.json", sampling_audit)
            train_loader = make_loader(
                train_items,
                make_transforms(config, True, crop_rows), True, config, sampler=sampler,
            )
            student = build_network("dynunet", 1, 0.0).to(device)
            ema = EMATeacher(student).to(device)
            optimizer = torch.optim.AdamW(
                student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            criterion = CombinedSegLoss(1, loss_type="dice_ce", include_boundary=False)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            best_key = None
            for epoch in range(1, args.epochs + 1):
                student.train()
                losses, dices, precisions, recalls = [], [], [], []
                wide_draws, gt_fractions = [], []
                for batch in train_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    target = batch["label"].float().to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=True):
                        logits = student(image)
                        loss = compute_multiscale_loss(criterion, logits, target)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(student, decay=args.ema_decay)
                    metrics = train_metrics(logits, target)
                    losses.append(float(loss.item()))
                    dices.append(metrics["dice"])
                    precisions.append(metrics["precision"])
                    recalls.append(metrics["recall"])
                    wide_draws.append(float(batch["training_crop_is_wide"].float().mean().item()))
                    gt_fractions.append(float(target.float().mean().item()))
                validate_now = (
                    epoch == 1 or epoch == args.epochs
                    or epoch % args.validation_every_n_epochs == 0
                )
                prefix = (
                    f"Fold {fold} | epoch {epoch:03d}/{args.epochs} | "
                    f"TRAIN loss={mean(losses):.4f} dice={mean(dices):.4f} "
                    f"prec={mean(precisions):.4f} rec={mean(recalls):.4f} "
                    f"wide={mean(wide_draws):.2f} gt_frac={mean(gt_fractions):.4f}"
                )
                if validate_now:
                    rows, raw_summary, lcc_summary = evaluate(
                        student, ema.shadow, val_loader, device, fold, crop_rows
                    )
                    print(
                        prefix + f" | VALID raw_dice={raw_summary['mean_dice']:.4f} "
                        f"lcc_dice={lcc_summary['mean_dice']:.4f} "
                        f"prec={raw_summary['mean_precision']:.4f} "
                        f"rec={raw_summary['mean_recall']:.4f} "
                        f"|RVE|={raw_summary['median_abs_rve_pct']:.2f}%"
                    )
                    key = (float(raw_summary["mean_dice"]), float(lcc_summary["mean_dice"]))
                    if best_key is None or key > best_key:
                        best_key = key
                        torch.save({
                            "net_A": student.state_dict(), "teacher": ema.state_dict(),
                            "optimizer": optimizer.state_dict(), "fold": fold,
                            "epoch": epoch, "best_key": list(key),
                            "validation_raw": raw_summary, "validation_lcc": lcc_summary,
                            "train_ids": spec["train_ids"], "val_ids": spec["val_ids"],
                            "training_sampling": sampling_audit,
                            "recipe": recipe(args),
                        }, best_path)
                else:
                    print(prefix + f" | VALID skipped (every {args.validation_every_n_epochs})")
            write_json(fold_dir / "training_complete.json", {
                "fold": fold, "epochs": int(args.epochs), "best_checkpoint": str(best_path),
                "training_sampling": sampling_audit,
                "recipe": recipe(args),
            })
            del student, ema
            torch.cuda.empty_cache()
        else:
            print("Completed fold found; re-evaluating its best checkpoint")

        state = torch.load(best_path, map_location=device, weights_only=False)
        if state.get("train_ids") != spec["train_ids"] or state.get("val_ids") != spec["val_ids"]:
            raise RuntimeError(f"Fold {fold}: checkpoint split provenance differs")
        if state.get("recipe") != recipe(args):
            raise RuntimeError(f"Fold {fold}: checkpoint recipe differs")
        student = build_network("dynunet", 1, 0.0).to(device)
        teacher = build_network("dynunet", 1, 0.0).to(device)
        student.load_state_dict(state["net_A"])
        teacher.load_state_dict(state["teacher"])
        prediction_dir = (
            output_dir / "oof_model_grid_masks" if args.save_oof_masks else None
        )
        rows, raw_summary, lcc_summary = evaluate(
            student, teacher, val_loader, device, fold, crop_rows, prediction_dir
        )
        print(
            f"Fold {fold} best epoch={state['epoch']} | "
            f"RAW Dice={raw_summary['mean_dice']:.4f} | "
            f"LCC Dice={lcc_summary['mean_dice']:.4f} | "
            f"RAW precision={raw_summary['mean_precision']:.4f} | "
            f"recall={raw_summary['mean_recall']:.4f}"
        )
        del student, teacher
        torch.cuda.empty_cache()
        return rows

    def compare_to_baseline(rows, baseline, expected_scorable, missing_frozen):
        comparisons = []
        for row in rows:
            case_id = str(row["case_id"])
            base = baseline[case_id]
            base_gt = float(base.get("gt_vox", 0.0))
            base_pred = float(base.get("pred_vox", 0.0))
            base_signed_rve = (base_pred - base_gt) / (base_gt + 1e-8) * 100.0
            comparisons.append({
                "fold": int(row["fold"]), "case_id": case_id,
                "fullvolume_dice": float(base["dice"]),
                "stage2_raw_dice": float(row["raw_dice"]),
                "stage2_lcc_dice": float(row["lcc_dice"]),
                "raw_delta_dice": float(row["raw_dice"]) - float(base["dice"]),
                "lcc_delta_dice": float(row["lcc_dice"]) - float(base["dice"]),
                "fullvolume_precision": float(base["precision"]),
                "stage2_raw_precision": float(row["raw_precision"]),
                "stage2_lcc_precision": float(row["lcc_precision"]),
                "fullvolume_recall": float(base["recall"]),
                "stage2_raw_recall": float(row["raw_recall"]),
                "stage2_lcc_recall": float(row["lcc_recall"]),
                "fullvolume_signed_rve": base_signed_rve,
                "stage2_raw_signed_rve": float(row["raw_signed_rve"]),
                "stage2_lcc_signed_rve": float(row["lcc_signed_rve"]),
                "fullvolume_hd95": float(base["hd95"]),
                "stage2_raw_hd95": float(row["raw_hd95"]),
                "stage2_lcc_hd95": float(row["lcc_hd95"]),
            })
        base_dice = [x["fullvolume_dice"] for x in comparisons]
        raw_delta = [x["raw_delta_dice"] for x in comparisons]
        lcc_delta = [x["lcc_delta_dice"] for x in comparisons]
        completed_folds = sorted({int(x["fold"]) for x in comparisons})
        complete_available = (
            len(comparisons) == int(expected_scorable)
            and len({str(x["case_id"]) for x in comparisons}) == int(expected_scorable)
            and completed_folds == list(range(5))
        )
        summary = {
            "version": "final91_qc_stage2_vs_fullvolume_v1",
            "n": len(comparisons),
            "completed_folds": completed_folds,
            "complete_original46_qc": (
                complete_available
                and int(expected_scorable) == EXPECTED_SCORABLE
                and not missing_frozen
            ),
            "mean_dice": {
                "fullvolume_final91_a3": mean(base_dice),
                "stage2_raw": mean(x["stage2_raw_dice"] for x in comparisons),
                "stage2_lcc": mean(x["stage2_lcc_dice"] for x in comparisons),
            },
            "mean_delta_dice": {
                "stage2_raw_minus_fullvolume": mean(raw_delta),
                "stage2_lcc_minus_fullvolume": mean(lcc_delta),
            },
            "case_effects_raw": {
                "improved": sum(x > 1e-6 for x in raw_delta),
                "worsened": sum(x < -1e-6 for x in raw_delta),
                "improved_ge_0p05": sum(x >= 0.05 for x in raw_delta),
                "worsened_le_minus_0p05": sum(x <= -0.05 for x in raw_delta),
            },
            "case_effects_lcc": {
                "improved": sum(x > 1e-6 for x in lcc_delta),
                "worsened": sum(x < -1e-6 for x in lcc_delta),
                "improved_ge_0p05": sum(x >= 0.05 for x in lcc_delta),
                "worsened_le_minus_0p05": sum(x <= -0.05 for x in lcc_delta),
            },
            "comparison_note": (
                "Fold 2 is architecture-only because the quarantined case was held out there. "
                "In other folds, the historical Final91 baseline trained with that case while "
                "Stage 2 correctly excludes it."
            ),
        }
        if final136_mode(args):
            summary.update({
                "version": "final136_stage2_vs_fullvolume_available_frozen_v1",
                "complete_available_frozen_source_oof": complete_available,
                "n_live_human_gold": int(args.expected_live),
                "n_training_cases": int(
                    args.expected_trainable
                    or (
                        args.expected_live
                        if args.include_quarantined
                        else args.expected_live - 1
                    )
                ),
                "n_available_frozen_source_scorable": int(expected_scorable),
                "missing_frozen_case_ids": list(missing_frozen),
                "external31_access": False,
                "comparison_note": (
                    "Final136 uses all 136 current labels for fold training except each usable "
                    "frozen-source validation subset. The historical 9435... case is included "
                    "as train-only; the exact four approved missing frozen cases are not scored."
                ),
            })
        return comparisons, summary

    (
        config, by_id, extras, quarantine, scorable, specs,
        crop_rows, stage1_checkpoints, baseline, source_path, cohort,
    ) = preflight()
    selected_folds = parse_fold(args.fold)
    output_dir = resolved_output_dir(args)
    stage1_dir = resolved_stage1_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        **recipe(args),
        "config": args.config,
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "audit_metadata": args.audit_metadata,
        "audit_metadata_sha256": file_sha256(args.audit_metadata),
        "stage1_dir": str(stage1_dir),
        "stage1_gate": str(stage1_dir / "centernet3d_gate_summary.json"),
        "stage1_gate_sha256": file_sha256(
            stage1_dir / "centernet3d_gate_summary.json"
        ),
        "stage1_oof_crops": str(stage1_dir / "centernet3d_oof_metrics.csv"),
        "stage1_oof_crops_sha256": file_sha256(
            stage1_dir / "centernet3d_oof_metrics.csv"
        ),
        "stage1_checkpoints": stage1_checkpoints,
        "full_volume_baseline": None if qc90_mode(args) else args.full_volume_baseline,
        "full_volume_baseline_sha256": (
            None if qc90_mode(args) else file_sha256(args.full_volume_baseline)
        ),
        "n_live_human_gold": int(args.expected_live),
        "n_qc_trainable": len(cohort["training_ids"]),
        "n_scorable_source": len(scorable),
        "quarantined_case_ids": quarantine,
        "train_only_extra_ids": [] if qc90_mode(args) else extras,
        "folds": specs,
    }
    if final136_mode(args):
        plan.update({
            "missing_frozen_case_ids": cohort["missing_frozen_case_ids"],
            "stage1_quarantined_case_ids": cohort["stage1_quarantined_case_ids"],
            "intentionally_included_prior_quarantine_ids": (
                cohort["intentionally_included_prior_quarantine_ids"]
            ),
            "train_only_case_ids": cohort["train_only_ids"],
            "training_scope": "all136_current_with_available_frozen_source_oof",
        })
    if qc90_mode(args):
        plan.update({"split_scope": "qc90", "save_oof_masks": bool(args.save_oof_masks)})
    plan_path = output_dir / "stage2_cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing Stage-2 plan differs: {plan_path}; use a fresh output directory")
    if not plan_path.exists():
        write_json(plan_path, plan)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-2 DynUNet training requires CUDA")
    print("=" * 120)
    if final136_mode(args):
        print("FINAL136 TWO-STAGE BLADDER SEGMENTATION — STAGE 2 CV")
    else:
        print(
            "FINAL91 FIVE-FOLD LABEL-QC — TWO-STAGE OOF"
            if qc90_mode(args) else "FINAL91-QC TWO-STAGE BLADDER SEGMENTATION — STAGE 2"
        )
    print(f"Running folds:            {selected_folds}")
    print(
        f"Data:                     {args.expected_live} live | "
        f"{len(cohort['training_ids'])} trainable | {len(scorable)} OOF-scorable"
    )
    if quarantine:
        print(f"Quarantined:              {', '.join(quarantine)}")
    else:
        print(f"Prior 9435 quarantine:    INCLUDED as train-only")
    if cohort["missing_frozen_case_ids"]:
        print(f"Missing frozen allowed:   {len(cohort['missing_frozen_case_ids'])} exact IDs")
    if qc90_mode(args):
        print("Frozen Stage 1:           matching QC90 OOF CenterNet; every case held out")
        print("Detector crop miss:       full-grid fallback for diagnosis only; recorded per case")
    else:
        print(
            f"Frozen Stage 1:           CenterNet gate PASS; "
            f"{len(scorable)}/{len(scorable)} available-case coverage"
        )
    print("Training crop:            GT + randomized safe margin/jitter")
    if args.train_wide_crop_probability > 0.0:
        print(
            f"Wide crop:                p={args.train_wide_crop_probability:.0%} | "
            f"margin={args.train_wide_margin_min:.0%}-{args.train_wide_margin_max:.0%} | "
            f"jitter={args.train_wide_center_jitter:.0%}"
        )
    if args.small_bladder_weight > 1.0:
        print(
            f"Small oversampling:       bottom {args.small_bladder_quantile:.0%} | "
            f"weight={args.small_bladder_weight:.2f}x | fold-local training labels only"
        )
    print(
        "Appearance augmentation: "
        + ("mild contrast/intensity/speckle" if args.mild_appearance_augmentation else "OFF")
    )
    print("Validation crop:          actual OOF CenterNet crop")
    print("Stage 2:                  DynUNet + DiceCE + Student/EMA ensemble @ .50")
    print("Metrics:                  full-grid RAW + LCC")
    print("External31:               NOT ACCESSED")
    print("=" * 120)

    spec_map = {int(x["fold"]): x for x in specs}
    fresh = []
    for fold in selected_folds:
        fresh.extend(train_fold(config, by_id, spec_map[fold], crop_rows, output_dir, device))

    results_path = output_dir / "stage2_oof_metrics.csv"
    existing = read_csv(results_path)
    combined = [row for row in existing if int(float(row["fold"])) not in selected_folds] + fresh
    combined.sort(key=lambda row: (int(float(row["fold"])), str(row["case_id"])))
    write_csv(results_path, combined)

    fold_summaries = []
    for fold in sorted({int(float(row["fold"])) for row in combined}):
        subset = [row for row in combined if int(float(row["fold"])) == fold]
        fold_summaries.append({
            "fold": fold, "n": len(subset),
            **{f"raw_{key}": value for key, value in summarize(subset, "raw").items() if key != "n"},
            **{f"lcc_{key}": value for key, value in summarize(subset, "lcc").items() if key != "n"},
        })
    write_csv(output_dir / "stage2_fold_summary.csv", fold_summaries)

    print("\n" + "=" * 120)
    if qc90_mode(args):
        raw_summary = summarize(combined, "raw")
        lcc_summary = summarize(combined, "lcc")
        completed_folds = sorted({int(float(row["fold"])) for row in combined})
        complete = (
            len(combined) == EXPECTED_QC_TRAINABLE
            and len({str(row["case_id"]) for row in combined}) == EXPECTED_QC_TRAINABLE
            and completed_folds == list(range(5))
        )
        summary = {
            "version": "final91_label_qc90_two_stage_oof_summary_v1",
            "role": "heldout annotation diagnosis; not final model performance validation",
            "n": len(combined), "completed_folds": completed_folds,
            "complete_qc90_oof": complete,
            "quarantined_case_ids": quarantine,
            "raw": raw_summary, "lcc": lcc_summary,
            "qc_full_grid_fallbacks": sum(
                int(float(row.get("qc_full_grid_fallback", 0))) for row in combined
            ),
            "external31_access": False,
        }
        write_json(output_dir / "stage2_qc90_oof_summary.json", summary)
        print("FINAL91 FIVE-FOLD LABEL-QC — TWO-STAGE OOF")
        print(f"OOF complete:         {complete}")
        print(f"Cases / folds:        {len(combined)} / {completed_folds}")
        print(f"Mean RAW/LCC Dice:    {raw_summary['mean_dice']:.4f} / {lcc_summary['mean_dice']:.4f}")
        print(f"RAW Dice <0.70:       {raw_summary['dice_lt_0p70']}")
        print(f"RAW Dice <0.50:       {raw_summary['dice_lt_0p50']}")
        print(f"QC crop fallbacks:    {summary['qc_full_grid_fallbacks']}")
        print("Interpretation:       low OOF Dice selects review candidates; it does not prove bad GT")
    else:
        comparisons, comparison_summary = compare_to_baseline(
            combined,
            baseline,
            expected_scorable=len(scorable),
            missing_frozen=cohort["missing_frozen_case_ids"],
        )
        write_csv(output_dir / "stage2_vs_fullvolume_case_comparison.csv", comparisons)
        write_json(output_dir / "stage2_vs_fullvolume_summary.json", comparison_summary)
        print(
            "FINAL136 STAGE-2 vs FULL-VOLUME FINAL91-A3 ON AVAILABLE FROZEN CASES"
            if final136_mode(args) else "STAGE-2 vs FULL-VOLUME FINAL91-A3"
        )
        print(f"Cases / folds:       {len(combined)} / {comparison_summary['completed_folds']}")
        print(
            f"Mean Dice:           {comparison_summary['mean_dice']['fullvolume_final91_a3']:.4f} -> "
            f"RAW {comparison_summary['mean_dice']['stage2_raw']:.4f} "
            f"({comparison_summary['mean_delta_dice']['stage2_raw_minus_fullvolume']:+.4f})"
        )
        print(
            f"LCC diagnostic:      {comparison_summary['mean_dice']['stage2_lcc']:.4f} "
            f"({comparison_summary['mean_delta_dice']['stage2_lcc_minus_fullvolume']:+.4f} vs full-volume)"
        )
        print(
            f"RAW case effects:    improved={comparison_summary['case_effects_raw']['improved']} | "
            f"worsened={comparison_summary['case_effects_raw']['worsened']} | "
            f"+>=.05={comparison_summary['case_effects_raw']['improved_ge_0p05']} | "
            f"<=-.05={comparison_summary['case_effects_raw']['worsened_le_minus_0p05']}"
        )
        if final136_mode(args):
            print(
                "Complete available frozen-source OOF: "
                f"{comparison_summary['complete_available_frozen_source_oof']}"
            )
        else:
            print(f"Complete original46-QC: {comparison_summary['complete_original46_qc']}")
    print(f"Results:             {results_path}")
    print("=" * 120)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audited Final91/Final136 Stage-2 CenterNet-cropped DynUNet CV"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--stage1-dir", default=str(STAGE1_DIR))
    parser.add_argument("--full-volume-baseline", default=str(FULL_VOLUME_BASELINE))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument(
        "--expected-live",
        type=int,
        choices=(91, 136),
        default=EXPECTED_LIVE,
        help="audited live cohort size; 136 activates the explicit Final136 contract",
    )
    parser.add_argument("--expected-trainable", type=int, default=None)
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="include the historical 9435... case as train-only (required for Final136)",
    )
    parser.add_argument(
        "--allow-missing-frozen",
        action="store_true",
        help="allow only the exact four frozen IDs approved by the Final136 audit",
    )
    parser.add_argument(
        "--split-scope", choices=("original46", "qc90"), default="original46",
        help=(
            "original46 preserves frozen-source performance CV; qc90 holds out all 90 "
            "Final91 clean labels once"
        ),
    )
    parser.add_argument("--fold", default="2", help="all, 0..4, or comma-separated subset; default 2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-every-n-epochs", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--train-margin-min", type=float, default=0.40)
    parser.add_argument("--train-margin-max", type=float, default=0.60)
    parser.add_argument("--train-center-jitter", type=float, default=0.10)
    parser.add_argument(
        "--train-wide-crop-probability", type=float, default=0.0,
        help="probability of replacing the locked crop with a wider GT-safe crop",
    )
    parser.add_argument("--train-wide-margin-min", type=float, default=0.60)
    parser.add_argument("--train-wide-margin-max", type=float, default=1.00)
    parser.add_argument("--train-wide-center-jitter", type=float, default=0.20)
    parser.add_argument(
        "--small-bladder-quantile", type=float, default=1.0 / 3.0,
        help="training-only native physical-volume quantile defining SMALL",
    )
    parser.add_argument(
        "--small-bladder-weight", type=float, default=1.0,
        help="WeightedRandomSampler weight for SMALL training cases; 1 disables oversampling",
    )
    parser.add_argument(
        "--mild-appearance-augmentation", action="store_true",
        help="add mild contrast, intensity-scale, and multiplicative-speckle augmentation",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--save-oof-masks", action="store_true",
        help="save each final held-out RAW/LCC 128^3 mask for the native review-pack builder",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        parse_fold(args.fold)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs < 1 or args.validation_every_n_epochs < 1:
        parser.error("epochs and validation cadence must be >=1")
    if args.expected_trainable is not None and args.expected_trainable < 1:
        parser.error("--expected-trainable must be >=1")
    if final136_mode(args):
        if args.split_scope != "original46":
            parser.error("Final136 requires --split-scope original46")
        if not args.include_quarantined or not args.allow_missing_frozen:
            parser.error(
                "Final136 requires --include-quarantined and --allow-missing-frozen"
            )
        if args.expected_trainable not in (None, 136):
            parser.error("Final136 requires --expected-trainable 136")
        if Path(args.audit_metadata).resolve() == AUDIT.resolve():
            parser.error("Final136 requires its explicit --audit-metadata path")
        if resolved_output_dir(args).resolve() == OUTPUT.resolve():
            parser.error("Final136 requires a fresh explicit --output-dir")
    elif args.include_quarantined or args.allow_missing_frozen:
        parser.error(
            "Final91 mode cannot use --include-quarantined or --allow-missing-frozen"
        )
    if args.crop_size < 32 or args.crop_size % 16:
        parser.error("--crop-size must be >=32 and divisible by 16")
    if not 0 <= args.train_margin_min <= args.train_margin_max <= 1:
        parser.error("Require 0 <= train-margin-min <= train-margin-max <= 1")
    if not 0 <= args.train_center_jitter <= 0.5:
        parser.error("--train-center-jitter must be in [0,0.5]")
    if not 0 <= args.train_wide_crop_probability <= 1:
        parser.error("--train-wide-crop-probability must be in [0,1]")
    if not 0 <= args.train_wide_margin_min <= args.train_wide_margin_max <= 2:
        parser.error("Require 0 <= train-wide-margin-min <= train-wide-margin-max <= 2")
    if not 0 <= args.train_wide_center_jitter <= 0.5:
        parser.error("--train-wide-center-jitter must be in [0,0.5]")
    if not 0 < args.small_bladder_quantile < 1:
        parser.error("--small-bladder-quantile must be in (0,1)")
    if args.small_bladder_weight < 1:
        parser.error("--small-bladder-weight must be >=1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be >0 and weight decay >=0")
    if not 0 < args.ema_decay < 1:
        parser.error("--ema-decay must be in (0,1)")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be in (0,1)")
    robust_requested = bool(
        args.train_wide_crop_probability > 0
        or args.small_bladder_weight > 1
        or args.mild_appearance_augmentation
    )
    locked_output = QC90_OUTPUT if qc90_mode(args) else OUTPUT
    if robust_requested and resolved_output_dir(args).resolve() == locked_output.resolve():
        parser.error("Robust Stage-2 experiments require a fresh explicit --output-dir")
    if args.dry_run:
        dry_run(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
