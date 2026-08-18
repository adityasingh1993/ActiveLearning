#!/usr/bin/env python3
"""Evaluate automatic predicted-ROI refinement on frozen external31.

This is the bridge between the GT-oracle ROI feasibility result and a practical OFFLINE
active-learning/auto-labeling refiner. The production model remains single-stage.

Protocol
--------
1. Run the frozen Final62 full-volume EMA model.
2. Propose an ROI using ONLY the EMA probability map plus a spatial/size prior learned from
   the 62 audited HUMAN_GOLD training labels. External GT is not used to choose the ROI.
3. Crop the native image with a generous safety margin.
4. Run the already-trained oracle-ROI EMA model inside that automatically proposed crop.
5. Paste the refined prediction back into the full native volume and evaluate against the
   frozen external31 GT.

The external GT is used only after ROI proposal for evaluation (Dice, ROI target coverage,
etc.). This external31 set is a development benchmark, not a pristine confirmatory cohort.

The ROI proposer extracts connected components at fixed probability levels 0.30/0.50/0.70
and ranks them using a training-only prior:
  - normalized target centroid prior,
  - foreground-volume-fraction prior,
  - component mean probability.
A probability-peak fallback is used only when no component survives the minimum size.

Outputs
-------
  external31_predicted_roi_case_comparison.csv
  external31_predicted_roi_summary.json
  predicted_crops/*.mha
"""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("evaluate_predicted_roi_refinement_external31.py requires SimpleITK") from exc

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError("evaluate_predicted_roi_refinement_external31.py requires scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.oracle_roi_utils import paste_crop_into_full
from scripts.evaluate_oracle_roi_external31 import baseline_ema_rows, summarize
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

DEFAULT_FULL_CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_ROI_CHECKPOINT = Path("experiments/oracle_roi_final62_v1/checkpoints/final_checkpoint.pth")
DEFAULT_POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
DEFAULT_ROUND2_AUDIT = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
DEFAULT_ROUND2_AUDIT_CSV = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.csv"
)
DEFAULT_BASELINE_CASES = Path(
    "experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv"
)
DEFAULT_ORACLE_CASES = Path(
    "experiments/external31_oracle_roi_final62_v1/external31_oracle_roi_case_comparison.csv"
)
DEFAULT_ORACLE_SUMMARY = Path(
    "experiments/external31_oracle_roi_final62_v1/external31_oracle_roi_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_predicted_roi_refinement_v1")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
            writer.writerow({key: row.get(key, "") for key in fields})


def robust_center_scale(values, floor):
    arr = np.asarray(values, dtype=float)
    center = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - center), axis=0)
    scale = 1.4826 * mad
    scale = np.maximum(scale, float(floor))
    return np.asarray(center, dtype=float), np.asarray(scale, dtype=float)


def build_training_target_prior(audit_csv_path: Path, expected_count=62):
    """Build spatial/size prior from audited HUMAN_GOLD only."""
    rows = read_csv(audit_csv_path)
    usable = []
    for row in rows:
        try:
            audit_ok = int(str(row.get("audit_ok", "0") or "0"))
        except ValueError:
            audit_ok = 0
        image_path = str(row.get("image_path", "")).strip()
        label_path = str(row.get("label_path", "")).strip()
        if audit_ok == 1 and image_path and label_path:
            usable.append(row)

    if expected_count > 0 and len(usable) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} audited HUMAN_GOLD rows for localization prior, found {len(usable)}"
        )

    centroids = []
    log_volume_fractions = []
    bbox_fractions = []
    for row in usable:
        image = sitk.ReadImage(str(row["image_path"]))
        label = sitk.ReadImage(str(row["label_path"]))
        if tuple(image.GetSize()) != tuple(label.GetSize()):
            raise RuntimeError(f"Audit prior geometry changed after audit: {row['case_id']}")
        mask = np.asarray(sitk.GetArrayFromImage(label)) > 0
        if mask.ndim != 3 or not mask.any():
            raise RuntimeError(f"Audit prior label is empty/non-3D: {row['case_id']}")
        zz, yy, xx = np.where(mask)
        sx, sy, sz = [float(x) for x in image.GetSize()]
        centroid_xyz = np.asarray(
            [(float(xx.mean()) + 0.5) / sx, (float(yy.mean()) + 0.5) / sy, (float(zz.mean()) + 0.5) / sz],
            dtype=float,
        )
        bbox_xyz = np.asarray(
            [
                (float(xx.max() - xx.min() + 1)) / sx,
                (float(yy.max() - yy.min() + 1)) / sy,
                (float(zz.max() - zz.min() + 1)) / sz,
            ],
            dtype=float,
        )
        volume_fraction = float(mask.sum()) / max(float(np.prod(image.GetSize())), 1.0)
        centroids.append(centroid_xyz)
        bbox_fractions.append(bbox_xyz)
        log_volume_fractions.append(math.log(max(volume_fraction, 1e-12)))

    centroid_center, centroid_scale = robust_center_scale(centroids, floor=0.06)
    logvol_center, logvol_scale = robust_center_scale(log_volume_fractions, floor=0.60)
    bbox_median = np.median(np.asarray(bbox_fractions, dtype=float), axis=0)
    bbox_q75 = np.quantile(np.asarray(bbox_fractions, dtype=float), 0.75, axis=0)

    return {
        "n_training_labels": len(usable),
        "centroid_center_xyz_norm": centroid_center,
        "centroid_scale_xyz_norm": centroid_scale,
        "log_volume_fraction_center": float(np.asarray(logvol_center).reshape(-1)[0]),
        "log_volume_fraction_scale": float(np.asarray(logvol_scale).reshape(-1)[0]),
        "bbox_fraction_median_xyz": np.asarray(bbox_median, dtype=float),
        "bbox_fraction_q75_xyz": np.asarray(bbox_q75, dtype=float),
    }


def component_candidates(prob_zyx, thresholds, prior, min_component_voxels=8):
    prob = np.asarray(prob_zyx, dtype=np.float32)
    if prob.ndim != 3:
        raise RuntimeError(f"Expected 3D probability map, got {prob.shape}")
    full_zyx = np.asarray(prob.shape, dtype=float)
    full_xyz = full_zyx[::-1]
    candidates = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    for threshold in thresholds:
        mask = prob >= float(threshold)
        labeled, n_components = ndimage.label(mask, structure=structure)
        if n_components <= 0:
            continue
        objects = ndimage.find_objects(labeled)
        for component_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            component = labeled[slc] == component_id
            voxels = int(component.sum())
            if voxels < int(min_component_voxels):
                continue

            z0, z1 = int(slc[0].start), int(slc[0].stop)
            y0, y1 = int(slc[1].start), int(slc[1].stop)
            x0, x1 = int(slc[2].start), int(slc[2].stop)
            local_prob = prob[slc][component]
            coords_local = np.argwhere(component)
            centroid_local_zyx = coords_local.mean(axis=0)
            centroid_zyx = centroid_local_zyx + np.asarray([z0, y0, x0], dtype=float)
            centroid_xyz_norm = (centroid_zyx[::-1] + 0.5) / full_xyz
            volume_fraction = voxels / max(float(prob.size), 1.0)

            centroid_z = (
                centroid_xyz_norm - prior["centroid_center_xyz_norm"]
            ) / prior["centroid_scale_xyz_norm"]
            centroid_prior = math.exp(-0.5 * float(np.mean(np.square(centroid_z))))
            logvol_z = (
                math.log(max(volume_fraction, 1e-12)) - prior["log_volume_fraction_center"]
            ) / prior["log_volume_fraction_scale"]
            volume_prior = math.exp(-0.5 * float(logvol_z * logvol_z))
            mean_probability = float(np.mean(local_prob))
            max_probability = float(np.max(local_prob))

            # Training-only ranking. No external GT term is permitted here.
            score = 0.45 * centroid_prior + 0.35 * volume_prior + 0.20 * mean_probability
            candidates.append({
                "source": "connected_component",
                "threshold": float(threshold),
                "component_id": int(component_id),
                "voxels": voxels,
                "volume_fraction": float(volume_fraction),
                "mean_probability": mean_probability,
                "max_probability": max_probability,
                "centroid_prior": float(centroid_prior),
                "volume_prior": float(volume_prior),
                "score": float(score),
                "bbox_start_xyz": (x0, y0, z0),
                "bbox_size_xyz": (x1 - x0, y1 - y0, z1 - z0),
                "centroid_xyz": tuple(float(x) for x in centroid_zyx[::-1]),
            })
    return candidates


def peak_fallback_candidate(prob_zyx, prior):
    prob = np.asarray(prob_zyx, dtype=np.float32)
    peak_zyx = np.asarray(np.unravel_index(int(np.argmax(prob)), prob.shape), dtype=float)
    full_xyz = np.asarray(prob.shape[::-1], dtype=float)
    center_xyz = peak_zyx[::-1]
    bbox_size = np.maximum(1, np.ceil(prior["bbox_fraction_median_xyz"] * full_xyz)).astype(int)
    start = np.floor(center_xyz - 0.5 * bbox_size).astype(int)
    full = full_xyz.astype(int)
    start = np.maximum(0, np.minimum(start, full - bbox_size))
    bbox_size = np.minimum(bbox_size, full - start)
    return {
        "source": "probability_peak_fallback",
        "threshold": float("nan"),
        "component_id": -1,
        "voxels": 1,
        "volume_fraction": 1.0 / max(float(prob.size), 1.0),
        "mean_probability": float(prob[tuple(peak_zyx.astype(int))]),
        "max_probability": float(prob[tuple(peak_zyx.astype(int))]),
        "centroid_prior": float("nan"),
        "volume_prior": float("nan"),
        "score": float("nan"),
        "bbox_start_xyz": tuple(int(x) for x in start),
        "bbox_size_xyz": tuple(int(x) for x in bbox_size),
        "centroid_xyz": tuple(float(x) for x in center_xyz),
    }


def choose_localization_candidate(prob_zyx, thresholds, prior, min_component_voxels):
    candidates = component_candidates(
        prob_zyx,
        thresholds=thresholds,
        prior=prior,
        min_component_voxels=min_component_voxels,
    )
    if not candidates:
        return peak_fallback_candidate(prob_zyx, prior), 0
    candidates.sort(
        key=lambda c: (
            -float(c["score"]),
            -float(c["mean_probability"]),
            -int(c["voxels"]),
            float(c["threshold"]),
        )
    )
    return candidates[0], len(candidates)


def expanded_roi_from_candidate(candidate, full_size_xyz, prior, margin_fraction=0.75, min_target_multiplier=2.0):
    full = np.asarray(full_size_xyz, dtype=int)
    bbox_start = np.asarray(candidate["bbox_start_xyz"], dtype=int)
    bbox_size = np.asarray(candidate["bbox_size_xyz"], dtype=int)
    center = bbox_start.astype(float) + 0.5 * np.maximum(bbox_size - 1, 0)

    margin_size = np.ceil(bbox_size.astype(float) * (1.0 + 2.0 * float(margin_fraction))).astype(int)
    training_min_size = np.ceil(
        prior["bbox_fraction_median_xyz"] * full.astype(float) * float(min_target_multiplier)
    ).astype(int)
    desired = np.maximum(margin_size, training_min_size)
    desired = np.maximum(desired, 1)
    desired = np.minimum(desired, full)

    start = np.floor(center - 0.5 * (desired - 1)).astype(int)
    start = np.maximum(start, 0)
    stop = start + desired
    overflow = np.maximum(stop - full, 0)
    start = np.maximum(start - overflow, 0)
    stop = np.minimum(start + desired, full)
    size = stop - start
    if np.any(size <= 0):
        raise RuntimeError(f"Invalid predicted ROI start={start.tolist()} size={size.tolist()}")
    return tuple(int(x) for x in start), tuple(int(x) for x in size)


def crop_gt_coverage(gt_zyx, start_xyz, size_xyz):
    gt = np.asarray(gt_zyx, dtype=bool)
    x0, y0, z0 = [int(x) for x in start_xyz]
    sx, sy, sz = [int(x) for x in size_xyz]
    gt_total = int(gt.sum())
    inside = int(gt[z0:z0 + sz, y0:y0 + sy, x0:x0 + sx].sum())
    return float(inside / max(gt_total, 1)), inside, gt_total


def oracle_case_rows(path: Path, expected_ids):
    rows = read_csv(path)
    by_id = {str(r["case_id"]): r for r in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("Duplicate case IDs in oracle ROI case CSV")
    if set(by_id) != set(expected_ids):
        raise RuntimeError("Oracle ROI case IDs do not exactly match predicted-ROI evaluation set")
    return by_id


def parse_thresholds(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise ValueError("candidate thresholds must be between 0 and 1")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("at least one candidate threshold is required")
    return values


def main():
    p = argparse.ArgumentParser(
        description="Evaluate automatic Final62 EMA predicted-ROI refinement on frozen external31"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--full-checkpoint", default=str(DEFAULT_FULL_CHECKPOINT))
    p.add_argument("--roi-checkpoint", default=str(DEFAULT_ROI_CHECKPOINT))
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--round2-audit-csv", default=str(DEFAULT_ROUND2_AUDIT_CSV))
    p.add_argument("--baseline-ema-csv", default=str(DEFAULT_BASELINE_CASES))
    p.add_argument("--oracle-case-csv", default=str(DEFAULT_ORACLE_CASES))
    p.add_argument("--oracle-summary", default=str(DEFAULT_ORACLE_SUMMARY))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--expected-training-labels", type=int, default=62)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seg-threshold", type=float, default=0.50)
    p.add_argument("--candidate-thresholds", default="0.30,0.50,0.70")
    p.add_argument("--min-component-voxels", type=int, default=8)
    p.add_argument("--roi-margin-fraction", type=float, default=0.75)
    p.add_argument("--min-target-multiplier", type=float, default=2.0)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if abs(float(args.seg_threshold) - 0.50) > 1e-8:
        p.error("Refined segmentation threshold is frozen at 0.50")
    if args.roi_margin_fraction < 0:
        p.error("--roi-margin-fraction must be >=0")
    if args.min_target_multiplier <= 0:
        p.error("--min-target-multiplier must be >0")
    if args.min_component_voxels < 1:
        p.error("--min-component-voxels must be >=1")
    candidate_thresholds = parse_thresholds(args.candidate_thresholds)

    full_checkpoint = Path(args.full_checkpoint)
    roi_checkpoint = Path(args.roi_checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    audit_csv_path = Path(args.round2_audit_csv)
    baseline_path = Path(args.baseline_ema_csv)
    oracle_case_path = Path(args.oracle_case_csv)
    oracle_summary_path = Path(args.oracle_summary)
    output_dir = Path(args.output_dir)
    for path in [
        full_checkpoint,
        roi_checkpoint,
        pool_manifest,
        audit_path,
        audit_csv_path,
        baseline_path,
        oracle_case_path,
        oracle_summary_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    crop_dir = output_dir / "predicted_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata is not marked passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(training_ids) != int(args.expected_training_labels):
        raise RuntimeError(
            f"Expected {args.expected_training_labels} Final62 training IDs, found {len(training_ids)}"
        )

    prior = build_training_target_prior(
        audit_csv_path, expected_count=int(args.expected_training_labels)
    )

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Predicted-ROI experiment requires prototype checkpoints with EMA teachers")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError("EXTERNAL/TRAINING LEAKAGE: " + ", ".join(overlap))

    baseline_by_id = baseline_ema_rows(baseline_path, cases_by_id)
    oracle_by_id = oracle_case_rows(oracle_case_path, cases_by_id)
    oracle_summary_payload = read_json(oracle_summary_path)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 124)
    print("PREDICTED-ROI REFINEMENT — FROZEN EXTERNAL31 DEVELOPMENT EXPERIMENT")
    print(f"Cases:                    {len(cases)}")
    print(f"Training prior labels:    {prior['n_training_labels']}")
    print(f"Candidate thresholds:     {candidate_thresholds}")
    print(f"ROI margin each side:     {args.roi_margin_fraction:.2f} x selected bbox extent")
    print(f"Minimum crop target size: {args.min_target_multiplier:.2f} x training median target bbox")
    print("ROI proposal uses full-volume EMA probabilities + training HUMAN_GOLD prior only.")
    print("External GT is evaluation-only and is not used to select the predicted ROI.")
    print("=" * 124)

    # ------------------------------------------------------------------
    # Phase 1: full-volume EMA proposes the crop. Save crops and metadata.
    # ------------------------------------------------------------------
    full_transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    full_inverse = build_invertd(
        keys=["pred"],
        transform=full_transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    full_items = [{"image": c["image"], "id": c["id"]} for c in cases]
    full_loader = DataLoader(
        Dataset(full_items, transform=full_transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    full_student, full_teacher = load_models(config, full_checkpoint, device)
    if full_teacher is None:
        raise RuntimeError("Final62 full-volume checkpoint has no EMA teacher")
    del full_student
    full_teacher.eval()

    crop_meta = {}
    with torch.no_grad():
        for index, batch in enumerate(full_loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, full_teacher))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, full_inverse, index=0)
            source_image, full_prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)

            # Sanity check: current run must reproduce the frozen full-volume EMA baseline.
            current_full = binary_metrics(full_prob_zyx > float(args.seg_threshold), gt)
            baseline_dice = float(baseline_by_id[case_id]["dice"])
            if abs(float(current_full["dice"]) - baseline_dice) > 5e-4:
                raise RuntimeError(
                    f"Full-volume EMA parity failed for {case_id}: current={current_full['dice']:.6f}, "
                    f"frozen={baseline_dice:.6f}"
                )

            candidate, n_candidates = choose_localization_candidate(
                full_prob_zyx,
                thresholds=candidate_thresholds,
                prior=prior,
                min_component_voxels=args.min_component_voxels,
            )
            start_xyz, size_xyz = expanded_roi_from_candidate(
                candidate,
                source_image.GetSize(),
                prior,
                margin_fraction=args.roi_margin_fraction,
                min_target_multiplier=args.min_target_multiplier,
            )
            coverage, gt_inside, gt_total = crop_gt_coverage(gt, start_xyz, size_xyz)
            crop_image = sitk.RegionOfInterest(source_image, size=size_xyz, index=start_xyz)
            crop_path = crop_dir / f"{case_id}{config.image_suffix}"
            sitk.WriteImage(crop_image, str(crop_path), useCompression=True)

            full_vox = max(int(np.prod(source_image.GetSize())), 1)
            crop_vox = int(np.prod(size_xyz))
            crop_meta[case_id] = {
                "crop_path": str(crop_path),
                "roi_start_x": int(start_xyz[0]),
                "roi_start_y": int(start_xyz[1]),
                "roi_start_z": int(start_xyz[2]),
                "roi_size_x": int(size_xyz[0]),
                "roi_size_y": int(size_xyz[1]),
                "roi_size_z": int(size_xyz[2]),
                "roi_volume_fraction_of_full": float(crop_vox / full_vox),
                "gt_coverage_by_predicted_roi": float(coverage),
                "gt_voxels_inside_predicted_roi": int(gt_inside),
                "gt_voxels_total": int(gt_total),
                "n_localization_candidates": int(n_candidates),
                "selected_candidate_source": candidate["source"],
                "selected_candidate_threshold": candidate["threshold"],
                "selected_candidate_score": candidate["score"],
                "selected_component_voxels": int(candidate["voxels"]),
                "selected_component_volume_fraction": float(candidate["volume_fraction"]),
                "selected_component_mean_probability": float(candidate["mean_probability"]),
                "selected_component_max_probability": float(candidate["max_probability"]),
                "selected_component_centroid_prior": candidate["centroid_prior"],
                "selected_component_volume_prior": candidate["volume_prior"],
            }
            print(
                f"[localize {index:2d}/{len(cases)}] {case_id} | "
                f"candidates={n_candidates:3d} | crop={crop_vox/full_vox:.3f} full | "
                f"GT coverage(eval only)={coverage:.3f}"
            )

    del full_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: ROI EMA refines each automatically proposed crop.
    # ------------------------------------------------------------------
    roi_transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    roi_inverse = build_invertd(
        keys=["pred"],
        transform=roi_transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    roi_items = [
        {"image": crop_meta[c["id"]]["crop_path"], "id": c["id"]}
        for c in cases
    ]
    roi_loader = DataLoader(
        Dataset(roi_items, transform=roi_transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    roi_student, roi_teacher = load_models(config, roi_checkpoint, device)
    if roi_teacher is None:
        raise RuntimeError("Oracle-ROI checkpoint has no EMA teacher")
    del roi_student
    roi_teacher.eval()

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(roi_loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            meta = crop_meta[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, roi_teacher))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, roi_inverse, index=0)
            _, crop_prob_zyx = normalize_native_probability(native_prob, meta["crop_path"])
            crop_pred = crop_prob_zyx > float(args.seg_threshold)

            full_source = sitk.ReadImage(str(case["image"]))
            gt = read_gt_binary(case["gt_path"], full_source)
            start_xyz = (meta["roi_start_x"], meta["roi_start_y"], meta["roi_start_z"])
            size_xyz = (meta["roi_size_x"], meta["roi_size_y"], meta["roi_size_z"])
            full_pred = paste_crop_into_full(crop_pred, gt.shape, start_xyz, size_xyz).astype(bool)
            metrics = binary_metrics(full_pred, gt)

            baseline = baseline_by_id[case_id]
            oracle = oracle_by_id[case_id]
            baseline_dice = float(baseline["dice"])
            oracle_dice = float(oracle["oracle_roi_ema_dice"])
            row = {
                "case_id": case_id,
                "baseline_full_volume_ema_dice": baseline_dice,
                "predicted_roi_refined_ema_dice": float(metrics["dice"]),
                "oracle_roi_ema_dice": oracle_dice,
                "predicted_minus_full_dice": float(metrics["dice"]) - baseline_dice,
                "oracle_minus_full_dice": oracle_dice - baseline_dice,
                "predicted_minus_oracle_dice": float(metrics["dice"]) - oracle_dice,
                **meta,
                **metrics,
            }
            rows.append(row)
            print(
                f"[refine   {index:2d}/{len(cases)}] {case_id} | "
                f"full={baseline_dice:.4f} -> predictedROI={metrics['dice']:.4f} | "
                f"oracle={oracle_dice:.4f} | coverage={meta['gt_coverage_by_predicted_roi']:.3f}"
            )

    rows.sort(key=lambda r: float(r["baseline_full_volume_ema_dice"]))
    predicted_summary = summarize(rows, args.failure_dice, args.high_quality_dice)

    baseline_summary_rows = []
    oracle_dices = []
    for case_id in sorted(cases_by_id):
        b = baseline_by_id[case_id]
        baseline_summary_rows.append({
            "dice": float(b["dice"]),
            "precision": float(b["precision"]),
            "recall": float(b["recall"]),
            "signed_rve_pct": float(b["signed_rve_pct"]),
        })
        oracle_dices.append(float(oracle_by_id[case_id]["oracle_roi_ema_dice"]))
    baseline_summary = summarize(baseline_summary_rows, args.failure_dice, args.high_quality_dice)

    oracle_summary = oracle_summary_payload.get("oracle_roi_ema", {})
    oracle_mean = float(oracle_summary.get("mean_dice", np.mean(oracle_dices)))
    oracle_median = float(oracle_summary.get("median_dice", np.median(oracle_dices)))
    oracle_fail = int(np.sum(np.asarray(oracle_dices) < float(args.failure_dice)))
    oracle_hq = int(np.sum(np.asarray(oracle_dices) >= float(args.high_quality_dice)))
    oracle_90 = int(np.sum(np.asarray(oracle_dices) >= 0.90))

    coverages = np.asarray([float(r["gt_coverage_by_predicted_roi"]) for r in rows], dtype=float)
    crop_fracs = np.asarray([float(r["roi_volume_fraction_of_full"]) for r in rows], dtype=float)
    baseline_gain_den = oracle_mean - float(baseline_summary["mean_dice"])
    recovered = (
        (float(predicted_summary["mean_dice"]) - float(baseline_summary["mean_dice"])) / baseline_gain_den
        if abs(baseline_gain_den) > 1e-12
        else float("nan")
    )

    case_csv = output_dir / "external31_predicted_roi_case_comparison.csv"
    write_csv(case_csv, rows)
    payload = {
        "version": "external31_predicted_roi_refinement_v1",
        "experiment_type": "automatic predicted ROI + offline ROI EMA refinement",
        "n_cases": len(rows),
        "full_checkpoint": str(full_checkpoint),
        "roi_checkpoint": str(roi_checkpoint),
        "candidate_thresholds": candidate_thresholds,
        "roi_margin_fraction_each_side": float(args.roi_margin_fraction),
        "min_target_multiplier": float(args.min_target_multiplier),
        "training_prior": {
            "n_training_labels": int(prior["n_training_labels"]),
            "centroid_center_xyz_norm": prior["centroid_center_xyz_norm"].tolist(),
            "centroid_scale_xyz_norm": prior["centroid_scale_xyz_norm"].tolist(),
            "log_volume_fraction_center": float(prior["log_volume_fraction_center"]),
            "log_volume_fraction_scale": float(prior["log_volume_fraction_scale"]),
            "bbox_fraction_median_xyz": prior["bbox_fraction_median_xyz"].tolist(),
            "bbox_fraction_q75_xyz": prior["bbox_fraction_q75_xyz"].tolist(),
        },
        "baseline_full_volume_ema": baseline_summary,
        "predicted_roi_refined_ema": predicted_summary,
        "oracle_roi_ema_reference": {
            "mean_dice": oracle_mean,
            "median_dice": oracle_median,
            "failures_dice_lt_070": oracle_fail,
            "high_quality_dice_gte_080": oracle_hq,
            "dice_gte_090": oracle_90,
        },
        "delta_mean_dice_predicted_vs_full": float(predicted_summary["mean_dice"]) - float(baseline_summary["mean_dice"]),
        "delta_mean_dice_oracle_vs_full": oracle_mean - float(baseline_summary["mean_dice"]),
        "fraction_of_oracle_mean_dice_gain_recovered": float(recovered),
        "predicted_roi_gt_coverage_median": float(np.median(coverages)),
        "predicted_roi_gt_coverage_mean": float(np.mean(coverages)),
        "predicted_roi_gt_coverage_lt_090": int(np.sum(coverages < 0.90)),
        "predicted_roi_gt_coverage_lt_050": int(np.sum(coverages < 0.50)),
        "predicted_roi_crop_fraction_median": float(np.median(crop_fracs)),
        "warning": (
            "External GT is evaluation-only. ROI proposal uses Final62 EMA probability maps plus priors derived "
            "only from the 62 audited HUMAN_GOLD training labels. This is an offline AL/refinement experiment; "
            "it does not change the single-stage production inference requirement."
        ),
    }
    summary_path = output_dir / "external31_predicted_roi_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("PREDICTED ROI REFINEMENT — EXTERNAL31 RESULT")
    print(f"Full-volume EMA mean Dice:      {baseline_summary['mean_dice']:.4f}")
    print(f"Predicted-ROI refined mean:     {predicted_summary['mean_dice']:.4f}")
    print(f"Oracle-ROI EMA mean Dice:       {oracle_mean:.4f}")
    print(f"Predicted vs full change:       {predicted_summary['mean_dice'] - baseline_summary['mean_dice']:+.4f}")
    print(f"Oracle vs full change:          {oracle_mean - baseline_summary['mean_dice']:+.4f}")
    print(f"Oracle gain recovered:          {recovered:.3f}")
    print(f"Median Dice full/pred/oracle:   {baseline_summary['median_dice']:.4f} / {predicted_summary['median_dice']:.4f} / {oracle_median:.4f}")
    print(f"Dice<.70 full/pred/oracle:      {baseline_summary['failures_dice_lt_070']} / {predicted_summary['failures_dice_lt_070']} / {oracle_fail}")
    print(f"Dice>=.80 full/pred/oracle:     {baseline_summary['high_quality_dice_gte_080']} / {predicted_summary['high_quality_dice_gte_080']} / {oracle_hq}")
    print(f"Dice>=.90 full/pred/oracle:     {baseline_summary['dice_gte_090']} / {predicted_summary['dice_gte_090']} / {oracle_90}")
    print(f"Median GT coverage by pred ROI: {np.median(coverages):.3f}")
    print(f"ROI GT coverage <.90:           {int(np.sum(coverages < 0.90))}/{len(rows)}")
    print(f"ROI GT coverage <.50:           {int(np.sum(coverages < 0.50))}/{len(rows)}")
    print(f"Median crop/full volume:        {np.median(crop_fracs):.3f}")
    print("\nPreviously low-Dice cases:")
    for row in rows:
        if float(row["baseline_full_volume_ema_dice"]) < float(args.failure_dice):
            print(
                f"  {row['case_id']} | full={float(row['baseline_full_volume_ema_dice']):.4f} "
                f"-> predROI={float(row['predicted_roi_refined_ema_dice']):.4f} "
                f"| oracle={float(row['oracle_roi_ema_dice']):.4f} "
                f"| coverage={float(row['gt_coverage_by_predicted_roi']):.3f}"
            )
    print(f"\nCases:   {case_csv}")
    print(f"Summary: {summary_path}")
    print("Decision rule: if predicted ROI recovers a useful fraction of oracle gain and coverage is high, integrate it into OFFLINE AL only.")
    print("=" * 124)


if __name__ == "__main__":
    main()
