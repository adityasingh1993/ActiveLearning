#!/usr/bin/env python3
"""Locked External31 evaluation of the Final91-QC two-stage CV committee.

Each of the five frozen pipelines consists of its fold's 3D CenterNet localizer and matching
Stage-2 DynUNet Student/EMA checkpoint.  A pipeline predicts one crop, segments that crop, and
pastes its probability back into the full 128^3 model grid.  The five full-grid probabilities are
averaged before exact native-grid inversion.  RAW @ 0.50 is primary; fixed 26-connected LCC is a
diagnostic.  External labels are never used for localization, model selection, threshold tuning,
or post-processing selection.

The evaluator intentionally refuses to run until Stage-2 OOF evaluation contains all 46 QC-clean
original cases across folds 0..4.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
STAGE1_DIR = Path("experiments/final91_qc_stage1_centernet3d_cv")
STAGE2_DIR = Path("experiments/final91_qc_stage2_centernet_dynunet_cv")
OUTPUT = Path("experiments/external31_final91_qc_two_stage_cv_committee_locked")
FULL_VOLUME_FINAL91 = Path("experiments/external31_final91_a3_locked/external31_case_metrics.csv")
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
EXPECTED_EXTERNAL = 31
EXPECTED_OOF = 46
EXPECTED_LIVE = 91
THRESHOLD = 0.50
SAFETY_MARGIN = 0.50
QUARANTINED_CASE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def summarize(rows):
    dice = np.asarray([float(x["dice"]) for x in rows], dtype=float)
    precision = np.asarray([float(x["precision"]) for x in rows], dtype=float)
    recall = np.asarray([float(x["recall"]) for x in rows], dtype=float)
    signed = np.asarray([float(x["signed_rve_pct"]) for x in rows], dtype=float)
    hd95 = np.asarray([float(x["hd95_mm"]) for x in rows], dtype=float)
    finite_hd95 = hd95[np.isfinite(hd95)]
    return {
        "n": len(rows),
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "mean_hd95_mm": float(np.mean(finite_hd95)) if finite_hd95.size else float("inf"),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def paired_comparison(current_rows, baseline_path, output_dir):
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"Full-volume Final91 external metrics absent; comparison skipped: {baseline_path}")
        return None
    old_rows = read_csv(baseline_path)
    if any("mode" in row for row in old_rows):
        old_rows = [row for row in old_rows if str(row.get("mode", "")).upper() == "ENSEMBLE"]
    old = {str(row["case_id"]): row for row in old_rows}
    new = {str(row["case_id"]): row for row in current_rows}
    if set(old) != set(new) or len(new) != EXPECTED_EXTERNAL:
        raise RuntimeError("Two-stage and full-volume Final91 metrics do not contain identical 31 IDs")
    rows = []
    for case_id in sorted(new):
        before, after = old[case_id], new[case_id]
        rows.append({
            "case_id": case_id,
            "fullvolume_final91_dice": float(before["dice"]),
            "two_stage_dice": float(after["dice"]),
            "delta_dice": float(after["dice"]) - float(before["dice"]),
            "fullvolume_final91_precision": float(before["precision"]),
            "two_stage_precision": float(after["precision"]),
            "fullvolume_final91_recall": float(before["recall"]),
            "two_stage_recall": float(after["recall"]),
        })
    delta = np.asarray([row["delta_dice"] for row in rows], dtype=float)
    summary = {
        "n": len(rows),
        "fullvolume_final91_mean_dice": float(np.mean([
            row["fullvolume_final91_dice"] for row in rows
        ])),
        "two_stage_mean_dice": float(np.mean([row["two_stage_dice"] for row in rows])),
        "delta_mean_dice": float(np.mean(delta)),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "improved_ge_0p05": int(np.sum(delta >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
    }
    write_csv(output_dir / "two_stage_vs_fullvolume_final91_external31_cases.csv", rows)
    (output_dir / "two_stage_vs_fullvolume_final91_external31_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def dry_run(args):
    print("=" * 120)
    print("FINAL91-QC TWO-STAGE CV COMMITTEE — LOCKED EXTERNAL31 DRY RUN")
    print("Required OOF:        46 QC-clean cases across folds 0..4")
    print("Fold pipeline:       CenterNet -> 50% margin crop -> Student+EMA Stage-2")
    print("Committee:           mean of five pasted full-grid probabilities")
    print("Primary:             RAW @ threshold 0.50")
    print("Diagnostic:          fixed 26-connected LCC @ threshold 0.50")
    print("Native geometry:     exact MONAI inversion before scoring")
    print(f"Images / GT:         {args.image_dir} / {args.gt_dir}")
    print(f"Output:              {args.output_dir}")
    print("External labels:     evaluation only")
    print("=" * 120)


def run(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.data import DataLoader, Dataset

    from hassl.compat import build_invertd
    from hassl.config import HASSLConfig
    from hassl.data.data_engine import get_base_transforms
    from hassl.training.trainer import build_network
    import scripts.train_supervised_cv as cv
    from scripts.benchmark_final91_a3_external31 import (
        collect_exact,
        hd95_mm,
        largest_connected_component,
    )
    from scripts.validate_external_threshold_31 import (
        binary_metrics,
        invert_probability_exact,
        normalize_native_probability,
        read_gt_binary,
    )

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels), nn.GELU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels), nn.GELU(),
            )

        def forward(self, value):
            return self.block(value)

    class CenterNet3D(nn.Module):
        """Exact architecture used by train_final91_stage1_centernet3d_cv.py."""

        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(4, 24, stride=2)
            self.enc2 = ConvBlock(24, 48, stride=2)
            self.enc3 = ConvBlock(48, 96, stride=2)
            self.enc4 = ConvBlock(96, 192, stride=2)
            self.up3 = nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2)
            self.fuse3 = ConvBlock(192, 96)
            self.up2 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
            self.fuse2 = ConvBlock(96, 64)
            self.head_features = nn.Sequential(
                nn.Conv3d(64, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64), nn.GELU(),
            )
            self.heatmap_head = nn.Conv3d(64, 1, 1)
            self.offset_head = nn.Conv3d(64, 3, 1)
            self.size_head = nn.Conv3d(64, 3, 1)
            nn.init.constant_(self.heatmap_head.bias, -2.19)

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
            value = torch.cat([image, self.coordinate_channels(image)], dim=1)
            c1 = self.enc1(value)
            c2 = self.enc2(c1)
            c3 = self.enc3(c2)
            c4 = self.enc4(c3)
            p3 = self.fuse3(torch.cat([self.up3(c4), c3], dim=1))
            p2 = self.fuse2(torch.cat([self.up2(p3), c2], dim=1))
            features = self.head_features(p2)
            return {
                "heatmap_logits": self.heatmap_head(features),
                "offset": torch.sigmoid(self.offset_head(features)),
                "size": 0.01 + 0.99 * torch.sigmoid(self.size_head(features)),
            }

    def gather_at(head, center_index):
        batch, channels, _, height, width = head.shape
        linear = (
            center_index[:, 0] * height * width
            + center_index[:, 1] * width
            + center_index[:, 2]
        )
        return head.reshape(batch, channels, -1).gather(
            2, linear.view(batch, 1, 1).expand(-1, channels, 1)
        ).squeeze(2)

    def decode_peak(outputs):
        heatmap = torch.sigmoid(outputs["heatmap_logits"])
        batch, _, depth, height, width = heatmap.shape
        flattened = heatmap.reshape(batch, -1)
        top_values, top_indices = torch.topk(flattened, k=2, dim=1)
        linear = top_indices[:, 0]
        center_index = torch.stack([
            linear // (height * width), (linear // width) % height, linear % width,
        ], dim=1)
        shape = torch.as_tensor(
            (depth, height, width), device=heatmap.device, dtype=heatmap.dtype
        )
        offset = gather_at(outputs["offset"], center_index)
        size = gather_at(outputs["size"], center_index)
        center = (center_index.to(offset.dtype) + offset) / torch.clamp(shape - 1.0, min=1.0)
        return center, size, top_values[:, 0], top_values[:, 0] - top_values[:, 1]

    def crop_bounds(center, size, spatial_shape):
        shape = np.asarray(spatial_shape, dtype=float)
        center_voxels = np.asarray(center, dtype=float) * np.maximum(shape - 1.0, 1.0)
        extent = np.asarray(size, dtype=float) * shape
        expanded = extent * (1.0 + 2.0 * SAFETY_MARGIN)
        lo = np.maximum(np.floor(center_voxels - 0.5 * expanded).astype(int), 0)
        hi = np.minimum(
            np.ceil(center_voxels + 0.5 * expanded).astype(int), shape.astype(int) - 1
        )
        return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))

    def crop_image(image, bounds):
        z0, z1, y0, y1, x0, x1 = bounds
        cropped = image[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        return F.interpolate(cropped, size=(128, 128, 128), mode="trilinear", align_corners=False)

    def paste_probability(probability, bounds):
        z0, z1, y0, y1, x0, x1 = bounds
        resized = F.interpolate(
            probability.float(), size=(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1),
            mode="trilinear", align_corners=False,
        )
        full = torch.zeros(
            (probability.shape[0], probability.shape[1], 128, 128, 128),
            device=probability.device, dtype=probability.dtype,
        )
        full[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = resized
        return full

    def main_prediction(output):
        if isinstance(output, (list, tuple)):
            return output[0]
        if torch.is_tensor(output) and output.ndim == 6:
            return output[:, 0]
        return output

    audit = read_json(args.audit_metadata)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final91 live-label audit is not passing")
    training_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(training_ids) != EXPECTED_LIVE or QUARANTINED_CASE_ID not in training_ids:
        raise RuntimeError("Final91 audit/quarantine provenance is not the expected live91 state")

    stage1_dir, stage2_dir = Path(args.stage1_dir), Path(args.stage2_dir)
    stage1_gate = read_json(stage1_dir / "centernet3d_gate_summary.json")
    if not stage1_gate.get("gate_pass", False) or not stage1_gate.get("stage2_authorized", False):
        raise RuntimeError("Stage-1 CenterNet gate has not passed")
    if not stage1_gate.get("complete_original46_qc_oof", False):
        raise RuntimeError("Stage-1 CenterNet OOF is not complete for original46-QC")
    if stage1_gate.get("quarantined_case_ids") != [QUARANTINED_CASE_ID]:
        raise RuntimeError("Stage-1 quarantine provenance differs")
    stage1_plan = read_json(stage1_dir / "centernet3d_cv_plan.json")
    if stage1_plan.get("resize_size") != [128, 128, 128]:
        raise RuntimeError("Stage-1 checkpoints were not trained on the locked 128^3 grid")
    if abs(float(stage1_plan.get("safety_margin_per_side", -1)) - SAFETY_MARGIN) > 1e-8:
        raise RuntimeError("Stage-1 checkpoints do not use the locked 50% crop margin")

    stage2_summary = read_json(stage2_dir / "stage2_vs_fullvolume_summary.json")
    stage2_rows = read_csv(stage2_dir / "stage2_oof_metrics.csv")
    completed_folds = sorted({int(float(row["fold"])) for row in stage2_rows})
    oof_ids = [str(row["case_id"]) for row in stage2_rows]
    if (
        not stage2_summary.get("complete_original46_qc", False)
        or len(stage2_rows) != EXPECTED_OOF
        or len(set(oof_ids)) != EXPECTED_OOF
        or completed_folds != list(range(5))
        or QUARANTINED_CASE_ID in set(oof_ids)
    ):
        raise RuntimeError(
            "External31 remains blocked: Stage-2 must contain all 46 QC-clean OOF cases "
            "across folds 0..4. Finish missing folds with --fold 0,1,3,4 (or the actual "
            "missing subset), then rerun this evaluator."
        )

    stage1_checkpoints, stage2_checkpoints = [], []
    for fold in range(5):
        first = stage1_dir / "checkpoints" / f"fold_{fold}" / "best_centernet3d.pth"
        second = stage2_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        marker = stage2_dir / "checkpoints" / f"fold_{fold}" / "training_complete.json"
        for path in (first, second, marker):
            if not path.exists():
                raise FileNotFoundError(path)
        first_state = torch.load(first, map_location="cpu", weights_only=False)
        second_state = torch.load(second, map_location="cpu", weights_only=False)
        if int(first_state.get("fold", -1)) != fold or int(second_state.get("fold", -1)) != fold:
            raise RuntimeError(f"Fold {fold}: checkpoint fold provenance differs")
        second_recipe = second_state.get("recipe", {})
        if (
            second_recipe.get("prediction") != "Student+EMA 50/50 ensemble"
            or abs(float(second_recipe.get("threshold", -1)) - THRESHOLD) > 1e-8
            or second_recipe.get("crop_size") != [128, 128, 128]
            or second_recipe.get("external31_access") is not False
        ):
            raise RuntimeError(f"Fold {fold}: Stage-2 recipe is not the locked definition")
        stage1_checkpoints.append((first, first_state))
        stage2_checkpoints.append((second, second_state))

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Two-stage External31 requires prototype DynUNet")
    cv.apply_baseline(config, resize_size=128, epochs=1)
    images = collect_exact(Path(args.image_dir), config.image_suffix)
    labels = collect_exact(Path(args.gt_dir), config.label_suffix)
    common = sorted(set(images) & set(labels))
    if len(common) != EXPECTED_EXTERNAL or set(images) != set(labels):
        raise RuntimeError(
            f"Frozen External31 mismatch: paired={len(common)}, images={len(images)}, labels={len(labels)}"
        )
    overlap = sorted(set(common) & set(training_ids))
    if overlap:
        raise RuntimeError("EXTERNAL/TRAINING LEAKAGE: " + ", ".join(overlap))

    output_dir = Path(args.output_dir)
    primary_path = output_dir / "external31_two_stage_raw_case_metrics.csv"
    if primary_path.exists() and not args.overwrite:
        raise RuntimeError(f"External result already exists: {primary_path}. Use --overwrite explicitly.")

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=False, to_tensor=True,
    )
    items = [{"id": case_id, "image": str(images[case_id])} for case_id in common]
    loader = DataLoader(
        Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Two-stage committee evaluation requires CUDA")

    pipelines = []
    for fold, ((_, first_state), (_, second_state)) in enumerate(
        zip(stage1_checkpoints, stage2_checkpoints)
    ):
        localizer = CenterNet3D().to(device)
        student = build_network("dynunet", 1, 0.0).to(device)
        teacher = build_network("dynunet", 1, 0.0).to(device)
        localizer.load_state_dict(first_state["model_state"])
        student.load_state_dict(second_state["net_A"])
        teacher.load_state_dict(second_state["teacher"])
        localizer.eval(); student.eval(); teacher.eval()
        pipelines.append((fold, localizer, student, teacher))

    print("=" * 120)
    print("FINAL91-QC TWO-STAGE CV COMMITTEE — LOCKED EXTERNAL31")
    print(f"Cases / overlap:     {len(common)} / {len(overlap)}")
    print("Stage-1:             five-fold 3D CenterNet committee members")
    print("Stage-2:             matching-fold DynUNet Student+EMA @ 0.50")
    print("Committee:           mean of five pasted full-grid probabilities")
    print("Primary / diagnostic: RAW / fixed 26-connected LCC")
    print("External GT:         evaluation only")
    print("=" * 120)

    raw_rows, lcc_rows, localizer_rows = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch["id"]
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            image = batch["image"].to(device)
            probability_sum = torch.zeros_like(image, dtype=torch.float32)
            fold_crop_fractions, fold_confidences = [], []
            for fold, localizer, student, teacher in pipelines:
                with torch.amp.autocast(device.type, enabled=True):
                    outputs = localizer(image)
                    center, size, confidence, peak_margin = decode_peak(outputs)
                bounds = crop_bounds(
                    center[0].float().cpu().numpy(), size[0].float().cpu().numpy(), image.shape[-3:]
                )
                crop = crop_image(image, bounds)
                with torch.amp.autocast(device.type, enabled=True):
                    student_probability = torch.sigmoid(main_prediction(student(crop)))
                    teacher_probability = torch.sigmoid(main_prediction(teacher(crop)))
                    fold_probability = 0.5 * (student_probability + teacher_probability)
                probability_sum += paste_probability(fold_probability, bounds).float()
                z0, z1, y0, y1, x0, x1 = bounds
                crop_fraction = float(
                    ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) / (128 ** 3)
                )
                fold_crop_fractions.append(crop_fraction)
                fold_confidences.append(float(confidence[0].item()))
                localizer_rows.append({
                    "case_id": case_id, "fold": fold,
                    "center_confidence": float(confidence[0].item()),
                    "center_peak_margin": float(peak_margin[0].item()),
                    "crop_fraction": crop_fraction,
                    "pred_center_z": float(center[0, 0].item()),
                    "pred_center_y": float(center[0, 1].item()),
                    "pred_center_x": float(center[0, 2].item()),
                    "pred_size_d": float(size[0, 0].item()),
                    "pred_size_h": float(size[0, 1].item()),
                    "pred_size_w": float(size[0, 2].item()),
                    "crop_z0": z0, "crop_z1": z1, "crop_y0": y0,
                    "crop_y1": y1, "crop_x0": x0, "crop_x1": x1,
                })
            committee_probability = probability_sum / len(pipelines)
            native_probability = invert_probability_exact(
                committee_probability, batch, inverse_transform, index=0
            )
            reference, probability_zyx = normalize_native_probability(
                native_probability, images[case_id]
            )
            gt = read_gt_binary(labels[case_id], reference)
            raw = probability_zyx > THRESHOLD
            lcc, component_count = largest_connected_component(raw)
            raw_metrics = binary_metrics(raw, gt)
            raw_metrics["hd95_mm"] = hd95_mm(raw, gt, reference.GetSpacing())
            lcc_metrics = binary_metrics(lcc, gt)
            lcc_metrics["hd95_mm"] = hd95_mm(lcc, gt, reference.GetSpacing())
            diagnostics = {
                "mean_center_confidence": float(np.mean(fold_confidences)),
                "minimum_center_confidence": float(np.min(fold_confidences)),
                "mean_crop_fraction": float(np.mean(fold_crop_fractions)),
                "minimum_crop_fraction": float(np.min(fold_crop_fractions)),
                "maximum_crop_fraction": float(np.max(fold_crop_fractions)),
            }
            raw_rows.append({
                "case_id": case_id, "mode": "ENSEMBLE", "postprocessing": "RAW",
                "threshold": THRESHOLD, "component_count": component_count,
                **diagnostics, **raw_metrics,
            })
            lcc_rows.append({
                "case_id": case_id, "mode": "ENSEMBLE", "postprocessing": "LCC_26",
                "threshold": THRESHOLD, "component_count": int(lcc.any()),
                **diagnostics, **lcc_metrics,
            })
            print(
                f"[{index:02d}/31] {case_id} | RAW/LCC={raw_metrics['dice']:.4f}/"
                f"{lcc_metrics['dice']:.4f} | components={component_count} | "
                f"conf={diagnostics['mean_center_confidence']:.3f} | "
                f"crop={diagnostics['mean_crop_fraction']:.3f}"
            )

    raw_rows.sort(key=lambda row: str(row["case_id"]))
    lcc_rows.sort(key=lambda row: str(row["case_id"]))
    localizer_rows.sort(key=lambda row: (str(row["case_id"]), int(row["fold"])))
    raw_summary, lcc_summary = summarize(raw_rows), summarize(lcc_rows)
    raw_by_id = {row["case_id"]: row for row in raw_rows}
    lcc_by_id = {row["case_id"]: row for row in lcc_rows}
    lcc_delta = np.asarray([
        float(lcc_by_id[case_id]["dice"]) - float(raw_by_id[case_id]["dice"])
        for case_id in sorted(raw_by_id)
    ])
    lcc_comparison = {
        "n": EXPECTED_EXTERNAL,
        "lcc_minus_raw_mean_dice": float(np.mean(lcc_delta)),
        "lcc_improved": int(np.sum(lcc_delta > 1e-6)),
        "lcc_worsened": int(np.sum(lcc_delta < -1e-6)),
        "lcc_improved_ge_0p05": int(np.sum(lcc_delta >= 0.05)),
        "lcc_worsened_le_minus_0p05": int(np.sum(lcc_delta <= -0.05)),
        "raw_multicomponent_cases": int(np.sum([
            int(row["component_count"]) > 1 for row in raw_rows
        ])),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(primary_path, raw_rows)
    write_csv(output_dir / "external31_two_stage_lcc_case_metrics.csv", lcc_rows)
    write_csv(output_dir / "external31_localizer_fold_diagnostics.csv", localizer_rows)
    write_csv(output_dir / "external31_two_stage_summary.csv", [
        {"postprocessing": "RAW", **raw_summary},
        {"postprocessing": "LCC_26", **lcc_summary},
    ])
    historical = paired_comparison(raw_rows, args.full_volume_final91_metrics, output_dir)
    metadata = {
        "version": "final91_qc_two_stage_cv_committee_external31_v1",
        "n_external": EXPECTED_EXTERNAL,
        "n_oof_gate_cases": EXPECTED_OOF,
        "completed_folds": list(range(5)),
        "quarantined_training_case_id": QUARANTINED_CASE_ID,
        "pipeline": "per-fold CenterNet crop -> matching Stage-2 Student+EMA -> paste full grid",
        "committee": "unweighted mean of five full-grid fold probabilities",
        "threshold": THRESHOLD,
        "stage1_safety_margin_per_side": SAFETY_MARGIN,
        "primary_postprocessing": "raw_no_lcc",
        "diagnostic_postprocessing": "largest_26_connected_component",
        "external_gt_usage": "evaluation_only",
        "raw_summary": raw_summary,
        "lcc_summary": lcc_summary,
        "raw_vs_lcc": lcc_comparison,
        "fullvolume_final91_comparison": historical,
        "stage1_checkpoints": [str(path) for path, _ in stage1_checkpoints],
        "stage2_checkpoints": [str(path) for path, _ in stage2_checkpoints],
        "warning": (
            "External31 is a frozen historical comparison benchmark, not a pristine prospective set. "
            "RAW remains primary; LCC is diagnostic only."
        ),
    }
    (output_dir / "external31_two_stage_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("FINAL91-QC TWO-STAGE CV COMMITTEE — EXTERNAL31 RAW vs LCC")
    print(f"Mean Dice:          {raw_summary['mean_dice']:.4f} -> {lcc_summary['mean_dice']:.4f} "
          f"({lcc_comparison['lcc_minus_raw_mean_dice']:+.4f})")
    print(f"Median Dice:        {raw_summary['median_dice']:.4f} -> {lcc_summary['median_dice']:.4f}")
    print(f"Precision:          {raw_summary['mean_precision']:.4f} -> {lcc_summary['mean_precision']:.4f}")
    print(f"Recall:             {raw_summary['mean_recall']:.4f} -> {lcc_summary['mean_recall']:.4f}")
    print(f"Median signed RVE:  {raw_summary['median_signed_rve_pct']:+.2f}% -> "
          f"{lcc_summary['median_signed_rve_pct']:+.2f}%")
    print(f"Median |RVE|:       {raw_summary['median_abs_rve_pct']:.2f}% -> "
          f"{lcc_summary['median_abs_rve_pct']:.2f}%")
    print(f"Mean HD95:          {raw_summary['mean_hd95_mm']:.3f} -> "
          f"{lcc_summary['mean_hd95_mm']:.3f} mm")
    print(f"Dice <0.70:         {raw_summary['dice_lt_0p70']} -> {lcc_summary['dice_lt_0p70']}")
    print(f"Dice <0.50:         {raw_summary['dice_lt_0p50']} -> {lcc_summary['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {raw_summary['dice_ge_0p80']} -> {lcc_summary['dice_ge_0p80']}")
    print(f"Raw multicomponent: {lcc_comparison['raw_multicomponent_cases']}/31")
    print(f"LCC case effects:   improved={lcc_comparison['lcc_improved']} | "
          f"worsened={lcc_comparison['lcc_worsened']} | "
          f"+>=.05={lcc_comparison['lcc_improved_ge_0p05']} | "
          f"<=-.05={lcc_comparison['lcc_worsened_le_minus_0p05']}")
    if historical is not None:
        print(f"Vs full Final91:    {historical['fullvolume_final91_mean_dice']:.4f} -> "
              f"{historical['two_stage_mean_dice']:.4f} ({historical['delta_mean_dice']:+.4f})")
    print("RAW is the predeclared primary result; LCC is diagnostic.")
    print(f"Results:            {output_dir}")
    print("=" * 120)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Locked Final91-QC two-stage CV committee evaluation on External31"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--stage1-dir", default=str(STAGE1_DIR))
    parser.add_argument("--stage2-dir", default=str(STAGE2_DIR))
    parser.add_argument("--full-volume-final91-metrics", default=str(FULL_VOLUME_FINAL91))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_EXTERNAL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.expected_count != EXPECTED_EXTERNAL:
        parser.error("Frozen benchmark requires --expected-count 31")
    if abs(float(args.threshold) - THRESHOLD) > 1e-8:
        parser.error("External benchmark is locked to --threshold 0.50")
    if args.dry_run:
        dry_run(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
