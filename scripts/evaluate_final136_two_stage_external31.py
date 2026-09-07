#!/usr/bin/env python3
"""Evaluate the final all136 CenterNet -> DynUNet pipeline on frozen External31.

The evaluator requires the audited 136-case training cohort and the matching final CenterNet and
Student/EMA DynUNet checkpoints. One predicted box with the locked 50% margin defines the crop;
the Stage-2 probability is pasted into the full 128^3 grid and exactly inverted to native geometry.
RAW @ 0.50 is primary and fixed 26-connected LCC is diagnostic. External31 is evaluation-only and
must not overlap the 136 training cases.
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

AUDIT = Path("experiments/final136_two_stage_all136/audit/final136_live_label_audit.json")
STAGE1_FINAL = Path("experiments/final136_two_stage_all136/stage1")
STAGE2_FINAL = Path("experiments/final136_two_stage_all136/stage2")
STAGE2_CV = Path("experiments/final91_qc_stage2_centernet_dynunet_cv")
OUTPUT = Path("experiments/external31_final136_two_stage_all136_locked")
FULL_VOLUME_FINAL91 = Path("experiments/external31_final91_a3_locked/external31_case_metrics.csv")
FINAL91_TWO_STAGE_METRICS = Path(
    "experiments/external31_final91_qc_two_stage_all90_locked/"
    "external31_all90_two_stage_raw_case_metrics.csv"
)
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
EXPECTED_EXTERNAL = 31
EXPECTED_TRAINING = 136
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
    dice = np.asarray([float(x["dice"]) for x in rows])
    precision = np.asarray([float(x["precision"]) for x in rows])
    recall = np.asarray([float(x["recall"]) for x in rows])
    signed = np.asarray([float(x["signed_rve_pct"]) for x in rows])
    hd95 = np.asarray([float(x["hd95_mm"]) for x in rows])
    finite_hd95 = hd95[np.isfinite(hd95)]
    return {
        "n": len(rows), "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)), "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)), "mean_recall": float(np.mean(recall)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "mean_hd95_mm": float(np.mean(finite_hd95)) if finite_hd95.size else float("inf"),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def compare(current, baseline_path, baseline_label, output_dir, filename):
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"{baseline_label} case metrics absent; comparison skipped: {baseline_path}")
        return None
    previous = read_csv(baseline_path)
    if any("mode" in row for row in previous):
        previous = [row for row in previous if str(row.get("mode", "")).upper() == "ENSEMBLE"]
    before = {str(row["case_id"]): row for row in previous}
    after = {str(row["case_id"]): row for row in current}
    if set(before) != set(after) or len(after) != EXPECTED_EXTERNAL:
        raise RuntimeError(f"{baseline_label} and Final136 two-stage metrics do not contain identical IDs")
    rows = []
    for case_id in sorted(after):
        delta = float(after[case_id]["dice"]) - float(before[case_id]["dice"])
        rows.append({
            "case_id": case_id, "baseline_label": baseline_label,
            "baseline_dice": float(before[case_id]["dice"]),
            "final136_two_stage_dice": float(after[case_id]["dice"]),
            "delta_dice": delta,
            "baseline_precision": float(before[case_id]["precision"]),
            "final136_two_stage_precision": float(after[case_id]["precision"]),
            "baseline_recall": float(before[case_id]["recall"]),
            "final136_two_stage_recall": float(after[case_id]["recall"]),
        })
    delta = np.asarray([row["delta_dice"] for row in rows])
    summary = {
        "baseline_label": baseline_label, "n": len(rows),
        "baseline_mean_dice": float(np.mean([row["baseline_dice"] for row in rows])),
        "final136_two_stage_mean_dice": float(np.mean([row["final136_two_stage_dice"] for row in rows])),
        "delta_mean_dice": float(np.mean(delta)),
        "improved": int(np.sum(delta > 1e-6)), "worsened": int(np.sum(delta < -1e-6)),
        "improved_ge_0p05": int(np.sum(delta >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
    }
    write_csv(output_dir / f"{filename}_cases.csv", rows)
    (output_dir / f"{filename}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def dry_run(args):
    print("=" * 116)
    print("FINAL136 ALL136 TWO-STAGE — FROZEN EXTERNAL31 DRY RUN")
    print("Pipeline:             one final CenterNet -> one DynUNet Student+EMA")
    print("Crop:                 single predicted box + locked 50% margin")
    print("Primary:              native RAW @ threshold 0.50")
    print("Diagnostic:           native fixed 26-connected LCC")
    print("Comparisons:          full-volume Final91 and final all90 two-stage Final91")
    print("External labels:      evaluation only")
    print(f"Output:               {args.output_dir}")
    print("=" * 116)


def run(args):
    import torch
    import torch.nn.functional as F
    from monai.data import DataLoader, Dataset

    from hassl.compat import build_invertd
    from hassl.config import HASSLConfig
    from hassl.data.data_engine import get_base_transforms
    from hassl.models.centernet3d import CenterNet3D, decode_peak
    from hassl.training.trainer import build_network
    import scripts.train_supervised_cv as cv
    from scripts.benchmark_final91_a3_external31 import (
        collect_exact, hd95_mm, largest_connected_component,
    )
    from scripts.validate_external_threshold_31 import (
        binary_metrics, invert_probability_exact, normalize_native_probability, read_gt_binary,
    )

    def main_prediction(output):
        if isinstance(output, (list, tuple)):
            return output[0]
        if torch.is_tensor(output) and output.ndim == 6:
            return output[:, 0]
        return output

    def crop_bounds(center, size):
        shape = np.asarray((128, 128, 128), dtype=float)
        center_voxels = np.asarray(center, dtype=float) * (shape - 1.0)
        expanded = np.asarray(size, dtype=float) * shape * (1.0 + 2.0 * SAFETY_MARGIN)
        lo = np.maximum(np.floor(center_voxels - 0.5 * expanded).astype(int), 0)
        hi = np.minimum(np.ceil(center_voxels + 0.5 * expanded).astype(int), 127)
        return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))

    def crop_image(image, bounds):
        z0, z1, y0, y1, x0, x1 = bounds
        value = image[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        return F.interpolate(value, size=(128, 128, 128), mode="trilinear", align_corners=False)

    def paste_probability(probability, bounds):
        z0, z1, y0, y1, x0, x1 = bounds
        value = F.interpolate(
            probability.float(), size=(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1),
            mode="trilinear", align_corners=False,
        )
        full = torch.zeros(
            (probability.shape[0], probability.shape[1], 128, 128, 128),
            device=probability.device, dtype=probability.dtype,
        )
        full[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = value
        return full

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    training_ids = sorted(str(x) for x in audit.get("training_case_ids", []))
    if (
        not audit.get("all_training_labels_passed_audit", False)
        or not audit.get("training_scope_provenance_enforced", False)
        or len(audited_ids) != EXPECTED_TRAINING
        or len(training_ids) != EXPECTED_TRAINING
        or audited_ids != training_ids
    ):
        raise RuntimeError("Final136 audit is not passing or does not select exactly all 136 IDs")
    if audit.get("excluded_training_case_ids", []) or audit.get("quarantined_case_ids", []):
        raise RuntimeError("Final136 audit unexpectedly excludes one or more current cases")
    if QUARANTINED_CASE_ID not in training_ids:
        raise RuntimeError("Final136 must intentionally include the historical 9435... case")
    stage2_cv_summary = read_json(Path(args.stage2_cv_dir) / "stage2_vs_fullvolume_summary.json")
    if not stage2_cv_summary.get("complete_original46_qc", False):
        raise RuntimeError("Complete Stage-2 original46-QC OOF is required")

    stage1_dir, stage2_dir = Path(args.stage1_final_dir), Path(args.stage2_final_dir)
    stage1_checkpoint = stage1_dir / "final_centernet3d.pth"
    stage2_checkpoint = stage2_dir / "final_stage2_dynunet.pth"
    stage1_metadata = read_json(stage1_dir / "final_centernet3d_metadata.json")
    stage2_metadata = read_json(stage2_dir / "final_stage2_metadata.json")
    first = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    second = torch.load(stage2_checkpoint, map_location="cpu", weights_only=False)
    expected_train_ids = training_ids
    if (
        first.get("train_ids") != expected_train_ids
        or second.get("train_ids") != expected_train_ids
        or stage1_metadata.get("train_ids") != expected_train_ids
        or stage2_metadata.get("train_ids") != expected_train_ids
    ):
        raise RuntimeError("Final two-stage checkpoints are not trained on the exact same 136 cases")
    for name, state, metadata in (
        ("Stage 1", first, stage1_metadata),
        ("Stage 2", second, stage2_metadata),
    ):
        if state.get("quarantined_case_ids", []) or metadata.get("quarantined_case_ids", []):
            raise RuntimeError(f"{name} unexpectedly excludes one or more Final136 cases")
        if state.get("intentionally_included_prior_quarantine_ids") != [QUARANTINED_CASE_ID]:
            raise RuntimeError(f"{name} checkpoint does not record intentional 9435... inclusion")
        if metadata.get("intentionally_included_prior_quarantine_ids") != [QUARANTINED_CASE_ID]:
            raise RuntimeError(f"{name} metadata does not record intentional 9435... inclusion")
    if (
        first.get("recipe", {}).get("external31_access") is not False
        or second.get("recipe", {}).get("external31_access") is not False
        or stage1_metadata.get("external31_access") is not False
        or stage2_metadata.get("external31_access") is not False
    ):
        raise RuntimeError("Final checkpoint metadata indicates external-data access")
    if (
        first.get("recipe", {}).get("architecture") != "centernet3d_fpn"
        or first.get("recipe", {}).get("resize_size") != [128, 128, 128]
        or abs(
            float(first.get("recipe", {}).get("safety_margin_per_side", -1))
            - SAFETY_MARGIN
        )
        > 1e-8
    ):
        raise RuntimeError("Final136 CenterNet checkpoint differs from the locked recipe")
    if second.get("recipe", {}).get("architecture") != "DynUNet":
        raise RuntimeError("Final136 Stage-2 checkpoint is not the locked DynUNet")
    if abs(float(second.get("recipe", {}).get("threshold", -1)) - THRESHOLD) > 1e-8:
        raise RuntimeError("Final Stage-2 checkpoint does not use locked threshold 0.50")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Final two-stage evaluation requires prototype DynUNet")
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
    primary_path = output_dir / "external31_final136_two_stage_raw_case_metrics.csv"
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
        raise RuntimeError("Final two-stage evaluation requires CUDA")
    localizer = CenterNet3D().to(device)
    student = build_network("dynunet", 1, 0.0).to(device)
    teacher = build_network("dynunet", 1, 0.0).to(device)
    localizer.load_state_dict(first["model_state"])
    student.load_state_dict(second["net_A"])
    teacher.load_state_dict(second["teacher"])
    localizer.eval(); student.eval(); teacher.eval()

    print("=" * 116)
    print("FINAL136 ALL136 TWO-STAGE — LOCKED EXTERNAL31")
    print(f"Cases / overlap:       {len(common)} / {len(overlap)}")
    print("Pipeline:              one CenterNet crop -> one Stage-2 Student+EMA")
    print("Primary / diagnostic: RAW / fixed 26-connected LCC")
    print("External GT:           evaluation only")
    print("=" * 116)

    raw_rows, lcc_rows, localizer_rows = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            image = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=True):
                outputs = localizer(image)
                center, size, confidence, peak_margin = decode_peak(outputs)
            bounds = crop_bounds(center[0].float().cpu().numpy(), size[0].float().cpu().numpy())
            crop = crop_image(image, bounds)
            with torch.amp.autocast(device.type, enabled=True):
                student_probability = torch.sigmoid(main_prediction(student(crop)))
                teacher_probability = torch.sigmoid(main_prediction(teacher(crop)))
                probability = 0.5 * (student_probability + teacher_probability)
            model_grid_probability = paste_probability(probability, bounds)
            # Preserve the exact MONAI MetaTensor history carried by the processed image.
            # A newly allocated torch.zeros tensor has no inverse Resize/Orientation trace.
            full_probability = image.clone()
            full_probability.zero_()
            full_probability.copy_(model_grid_probability.to(dtype=full_probability.dtype))
            native_probability = invert_probability_exact(
                full_probability, batch, inverse_transform, index=0
            )
            reference, probability_zyx = normalize_native_probability(
                native_probability, images[case_id]
            )
            gt = read_gt_binary(labels[case_id], reference)
            raw = probability_zyx > THRESHOLD
            lcc, component_count = largest_connected_component(raw)
            raw_metrics, lcc_metrics = binary_metrics(raw, gt), binary_metrics(lcc, gt)
            raw_metrics["hd95_mm"] = hd95_mm(raw, gt, reference.GetSpacing())
            lcc_metrics["hd95_mm"] = hd95_mm(lcc, gt, reference.GetSpacing())
            z0, z1, y0, y1, x0, x1 = bounds
            crop_fraction = float(
                ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) / (128 ** 3)
            )
            diagnostics = {
                "center_confidence": float(confidence[0].item()),
                "center_peak_margin": float(peak_margin[0].item()),
                "crop_fraction": crop_fraction,
                "crop_z0": z0, "crop_z1": z1, "crop_y0": y0,
                "crop_y1": y1, "crop_x0": x0, "crop_x1": x1,
            }
            localizer_rows.append({"case_id": case_id, **diagnostics})
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
                f"conf={confidence[0].item():.3f} | crop={crop_fraction:.3f}"
            )

    raw_rows.sort(key=lambda row: str(row["case_id"]))
    lcc_rows.sort(key=lambda row: str(row["case_id"]))
    localizer_rows.sort(key=lambda row: str(row["case_id"]))
    raw_summary, lcc_summary = summarize(raw_rows), summarize(lcc_rows)
    raw_by_id = {row["case_id"]: row for row in raw_rows}
    lcc_by_id = {row["case_id"]: row for row in lcc_rows}
    lcc_delta = np.asarray([
        float(lcc_by_id[x]["dice"]) - float(raw_by_id[x]["dice"]) for x in sorted(raw_by_id)
    ])
    lcc_comparison = {
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
    write_csv(output_dir / "external31_final136_two_stage_lcc_case_metrics.csv", lcc_rows)
    write_csv(output_dir / "external31_final136_localizer_diagnostics.csv", localizer_rows)
    write_csv(output_dir / "external31_final136_two_stage_summary.csv", [
        {"postprocessing": "RAW", **raw_summary},
        {"postprocessing": "LCC_26", **lcc_summary},
    ])
    full_comparison = compare(
        raw_rows, args.full_volume_final91_metrics, "FULL_VOLUME_FINAL91", output_dir,
        "final136_two_stage_vs_fullvolume_final91",
    )
    final91_two_stage_comparison = compare(
        raw_rows, args.final91_two_stage_metrics, "FINAL91_ALL90_TWO_STAGE", output_dir,
        "final136_two_stage_vs_final91_all90_two_stage",
    )
    metadata = {
        "version": "final136_two_stage_all136_external31_v1",
        "n_external": EXPECTED_EXTERNAL,
        "pipeline": "single final all136 CenterNet -> single final all136 DynUNet Student+EMA",
        "threshold": THRESHOLD, "stage1_safety_margin_per_side": SAFETY_MARGIN,
        "primary_postprocessing": "raw_no_lcc",
        "diagnostic_postprocessing": "largest_26_connected_component",
        "native_inversion": (
            "full-grid probability carried by a clone of the processed image MetaTensor; "
            "prediction receives a deep copy of the exact image applied_operations trace"
        ),
        "external_gt_usage": "evaluation_only",
        "raw_summary": raw_summary, "lcc_summary": lcc_summary,
        "raw_vs_lcc": lcc_comparison,
        "fullvolume_final91_comparison": full_comparison,
        "final91_all90_two_stage_comparison": final91_two_stage_comparison,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "interpretation_role": (
            "locked post-training evaluation of the expanded all136 model on the unchanged "
            "External31 benchmark"
        ),
    }
    (output_dir / "external31_final136_two_stage_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 116)
    print("FINAL136 ALL136 TWO-STAGE — EXTERNAL31 RAW vs LCC")
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
    if full_comparison is not None:
        print(f"Vs full Final91:    {full_comparison['baseline_mean_dice']:.4f} -> "
              f"{full_comparison['final136_two_stage_mean_dice']:.4f} "
              f"({full_comparison['delta_mean_dice']:+.4f})")
    if final91_two_stage_comparison is not None:
        print(f"Vs Final91 2-stage: {final91_two_stage_comparison['baseline_mean_dice']:.4f} -> "
              f"{final91_two_stage_comparison['final136_two_stage_mean_dice']:.4f} "
              f"({final91_two_stage_comparison['delta_mean_dice']:+.4f})")
    print("RAW remains primary; LCC remains diagnostic.")
    print(f"Results:            {output_dir}")
    print("=" * 116)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate final all136 CenterNet-DynUNet pipeline on frozen External31"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--stage1-final-dir", default=str(STAGE1_FINAL))
    parser.add_argument("--stage2-final-dir", default=str(STAGE2_FINAL))
    parser.add_argument("--stage2-cv-dir", default=str(STAGE2_CV))
    parser.add_argument("--full-volume-final91-metrics", default=str(FULL_VOLUME_FINAL91))
    parser.add_argument("--final91-two-stage-metrics", default=str(FINAL91_TWO_STAGE_METRICS))
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
