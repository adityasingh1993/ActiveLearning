#!/usr/bin/env python3
"""Evaluate conservative 50/50 guided-ROI probability fusion on external31.

This is a single predeclared diagnostic. It DOES NOT retrain any model and DOES NOT sweep the
fusion weight.

Fusion rule
-----------
- outside the predicted ROI: keep the Final62 full-volume EMA probability unchanged;
- inside the predicted ROI:  0.50 * full-volume EMA probability + 0.50 * guided ROI probability;
- final segmentation threshold: 0.50.

The ROI proposal is identical to the existing guided-refiner external31 evaluator and uses only
deployable signals (Final62 EMA probability + the training-only localization prior). External GT
is used only after localization/fusion to calculate metrics.

External31 has been repeatedly examined and remains a development benchmark. Production remains
a single-stage segmentation model; this refiner/fusion path is offline-only.
"""

import os
import sys


def _consume_gpu_argument(argv):
    args = list(argv)
    gpu = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == "--gpu":
            if i + 1 >= len(args):
                raise SystemExit("--gpu requires 0 or 1")
            gpu = args[i + 1]
            i += 2
            continue
        if token.startswith("--gpu="):
            gpu = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if gpu is not None:
        if gpu not in {"0", "1"}:
            raise SystemExit(f"--gpu must be 0 or 1, got {gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu, cleaned


SELECTED_GPU, CLEAN_ARGV = _consume_gpu_argument(sys.argv)
sys.argv = CLEAN_ARGV

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("evaluate_guided_refiner_fusion50_external31.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.evaluate_guided_refiner_oof47_2ch_external31 import (
    DEFAULT_BASELINE_CASES,
    DEFAULT_FULL_CHECKPOINT,
    DEFAULT_GUIDED_CHECKPOINT,
    DEFAULT_ORACLE_CASES,
    DEFAULT_POOL_MANIFEST,
    DEFAULT_ROUND2_AUDIT,
    DEFAULT_ROUND2_AUDIT_CSV,
    build_guided_infer_transform,
    load_guided_teacher,
    read_json,
    write_csv,
)
from scripts.evaluate_oracle_roi_external31 import baseline_ema_rows, summarize
from scripts.evaluate_predicted_roi_refinement_external31 import (
    build_training_target_prior,
    choose_localization_candidate,
    crop_gt_coverage,
    expanded_roi_from_candidate,
    oracle_case_rows,
    parse_thresholds,
)
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

FUSION_WEIGHT_FULL = 0.50
FUSION_WEIGHT_GUIDED = 0.50
DEFAULT_OUTPUT_DIR = Path("experiments/external31_guided_refiner_fusion50_oof47_2ch_v1")


def _fuse_roi_probability(full_prob_zyx, guided_crop_prob_zyx, start_xyz, size_xyz):
    """Keep full probability outside ROI and average full/guided probability inside ROI."""
    full = np.asarray(full_prob_zyx, dtype=np.float32)
    guided = np.asarray(guided_crop_prob_zyx, dtype=np.float32)

    x0, y0, z0 = [int(v) for v in start_xyz]
    sx, sy, sz = [int(v) for v in size_xyz]
    expected_shape = (sz, sy, sx)
    if tuple(guided.shape) != expected_shape:
        raise RuntimeError(
            f"Guided crop shape mismatch: got={tuple(guided.shape)}, expected={expected_shape} "
            f"from ROI size XYZ={tuple(size_xyz)}"
        )

    z1, y1, x1 = z0 + sz, y0 + sy, x0 + sx
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 > full.shape[0] or y1 > full.shape[1] or x1 > full.shape[2]:
        raise RuntimeError(
            f"ROI outside full array: startXYZ={start_xyz}, sizeXYZ={size_xyz}, fullZYX={full.shape}"
        )

    fused = full.copy()
    full_roi = full[z0:z1, y0:y1, x0:x1]
    fused[z0:z1, y0:y1, x0:x1] = (
        FUSION_WEIGHT_FULL * full_roi + FUSION_WEIGHT_GUIDED * guided
    )
    return np.clip(fused, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description="Frozen 50/50 full-EMA + guided-ROI fusion on external31")
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--full-checkpoint", default=str(DEFAULT_FULL_CHECKPOINT))
    p.add_argument("--guided-checkpoint", default=str(DEFAULT_GUIDED_CHECKPOINT))
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--round2-audit-csv", default=str(DEFAULT_ROUND2_AUDIT_CSV))
    p.add_argument("--baseline-ema-csv", default=str(DEFAULT_BASELINE_CASES))
    p.add_argument("--oracle-case-csv", default=str(DEFAULT_ORACLE_CASES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
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
        p.error("Fusion50 evaluation is frozen at segmentation threshold 0.50")
    thresholds = parse_thresholds(args.candidate_thresholds)

    full_checkpoint = Path(args.full_checkpoint)
    guided_checkpoint = Path(args.guided_checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    audit_csv = Path(args.round2_audit_csv)
    baseline_path = Path(args.baseline_ema_csv)
    oracle_path = Path(args.oracle_case_csv)
    output_dir = Path(args.output_dir)

    for path in [full_checkpoint, guided_checkpoint, pool_manifest, audit_path, audit_csv, baseline_path, oracle_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")

    crop_image_dir = output_dir / "predicted_crops" / "images"
    crop_coarse_dir = output_dir / "predicted_crops" / "coarse"
    full_prob_dir = output_dir / "full_ema_probabilities"
    crop_image_dir.mkdir(parents=True, exist_ok=True)
    crop_coarse_dir.mkdir(parents=True, exist_ok=True)
    full_prob_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Fusion50 evaluation requires prototype DynUNet configuration")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    audit = read_json(audit_path)
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(training_ids) != 62:
        raise RuntimeError(f"Expected Final62 audit, found {len(training_ids)} IDs")
    prior = build_training_target_prior(audit_csv, expected_count=62)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError("EXTERNAL/TRAINING LEAKAGE: " + ", ".join(overlap))

    baseline_by_id = baseline_ema_rows(baseline_path, cases_by_id)
    oracle_by_id = oracle_case_rows(oracle_path, cases_by_id)

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 126)
    print("GUIDED ROI CONSERVATIVE FUSION50 — FROZEN EXTERNAL31 DEVELOPMENT EVALUATION")
    print(f"Cases:                  {len(cases)}")
    print("Fusion outside ROI:     keep Final62 EMA probability")
    print("Fusion inside ROI:      0.50 * Final62 EMA + 0.50 * guided ROI")
    print("Segmentation threshold: 0.50")
    print(f"Physical GPU:           {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("No fusion-weight sweep. External GT is evaluation-only and never selects/repairs the ROI.")
    print("=" * 126)

    full_transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    full_inverse = build_invertd(
        keys=["pred"], transform=full_transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    full_loader = DataLoader(
        Dataset([{"image": c["image"], "id": c["id"]} for c in cases], transform=full_transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    _, full_teacher = load_models(config, full_checkpoint, device)
    if full_teacher is None:
        raise RuntimeError("Final62 checkpoint has no EMA teacher")
    full_teacher.eval()

    crop_meta = {}
    with torch.no_grad():
        for index, batch in enumerate(full_loader, start=1):
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, full_teacher)))

            native_prob = invert_probability_exact(prob_t, batch, full_inverse, index=0)
            source_image, full_prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)

            current_full = binary_metrics(full_prob_zyx > float(args.seg_threshold), gt)
            frozen_full = float(baseline_by_id[case_id]["dice"])
            if abs(float(current_full["dice"]) - frozen_full) > 5e-4:
                raise RuntimeError(
                    f"Final62 EMA parity failed for {case_id}: current={current_full['dice']:.6f}, frozen={frozen_full:.6f}"
                )

            full_prob_image = sitk.GetImageFromArray(np.asarray(full_prob_zyx, dtype=np.float32))
            full_prob_image.CopyInformation(source_image)
            full_prob_path = full_prob_dir / f"{case_id}.ema_prob.mha"
            sitk.WriteImage(full_prob_image, str(full_prob_path), useCompression=True)

            candidate, n_candidates = choose_localization_candidate(
                full_prob_zyx,
                thresholds=thresholds,
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

            image_crop = sitk.RegionOfInterest(source_image, size=size_xyz, index=start_xyz)
            coarse_crop = sitk.RegionOfInterest(full_prob_image, size=size_xyz, index=start_xyz)
            image_crop_path = crop_image_dir / f"{case_id}.mha"
            coarse_crop_path = crop_coarse_dir / f"{case_id}.ema_prob.mha"
            sitk.WriteImage(image_crop, str(image_crop_path), useCompression=True)
            sitk.WriteImage(coarse_crop, str(coarse_crop_path), useCompression=True)

            full_vox = max(int(np.prod(source_image.GetSize())), 1)
            crop_vox = int(np.prod(size_xyz))
            crop_meta[case_id] = {
                "full_probability_path": str(full_prob_path),
                "image_crop_path": str(image_crop_path),
                "coarse_crop_path": str(coarse_crop_path),
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
                "selected_candidate_threshold": float(candidate["threshold"]),
                "selected_candidate_score": float(candidate["score"]),
            }
            print(
                f"[localize {index:2d}/{len(cases)}] {case_id} | crop/full={crop_vox/full_vox:.3f} | "
                f"GT coverage(eval)={coverage:.3f}"
            )

    del full_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    roi_transform = build_guided_infer_transform(config, args.resize_size)
    roi_inverse = build_invertd(
        keys=["pred"], transform=roi_transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    roi_items = [
        {
            "image": crop_meta[c["id"]]["image_crop_path"],
            "coarse": crop_meta[c["id"]]["coarse_crop_path"],
            "id": c["id"],
        }
        for c in cases
    ]
    roi_loader = DataLoader(Dataset(roi_items, transform=roi_transform), batch_size=1, shuffle=False, num_workers=0)
    guided_teacher = load_guided_teacher(config, guided_checkpoint, device)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(roi_loader, start=1):
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            case = cases_by_id[case_id]
            meta = crop_meta[case_id]
            guided_t = torch.cat([batch["image"], batch["coarse"]], dim=1).to(device)
            if guided_t.shape[1] != 2:
                raise RuntimeError(f"Expected 2 guided channels, got {guided_t.shape}")

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                prob_t = torch.sigmoid(cv.main_prediction(inferer(guided_t, guided_teacher)))

            native_prob = invert_probability_exact(prob_t, batch, roi_inverse, index=0)
            _, guided_crop_prob_zyx = normalize_native_probability(native_prob, meta["image_crop_path"])

            full_prob_img = sitk.ReadImage(meta["full_probability_path"])
            full_prob_zyx = sitk.GetArrayFromImage(full_prob_img).astype(np.float32, copy=False)
            full_source = sitk.ReadImage(str(case["image"]))
            gt = read_gt_binary(case["gt_path"], full_source)

            start_xyz = (meta["roi_start_x"], meta["roi_start_y"], meta["roi_start_z"])
            size_xyz = (meta["roi_size_x"], meta["roi_size_y"], meta["roi_size_z"])
            fused_prob_zyx = _fuse_roi_probability(
                full_prob_zyx,
                guided_crop_prob_zyx,
                start_xyz=start_xyz,
                size_xyz=size_xyz,
            )
            fused_pred = fused_prob_zyx > float(args.seg_threshold)
            metrics = binary_metrics(fused_pred, gt)

            full_dice = float(baseline_by_id[case_id]["dice"])
            oracle_dice = float(oracle_by_id[case_id]["oracle_roi_ema_dice"])
            rows.append({
                "case_id": case_id,
                "baseline_full_volume_ema_dice": full_dice,
                "fusion50_guided_roi_dice": float(metrics["dice"]),
                "oracle_roi_ema_dice": oracle_dice,
                "fusion50_minus_full_dice": float(metrics["dice"]) - full_dice,
                "fusion50_minus_oracle_dice": float(metrics["dice"]) - oracle_dice,
                "fusion_weight_full": FUSION_WEIGHT_FULL,
                "fusion_weight_guided": FUSION_WEIGHT_GUIDED,
                **meta,
                **metrics,
            })
            print(
                f"[fusion   {index:2d}/{len(cases)}] {case_id} | full={full_dice:.4f} -> "
                f"fusion50={metrics['dice']:.4f} | oracle={oracle_dice:.4f} | "
                f"coverage={meta['gt_coverage_by_predicted_roi']:.3f}"
            )

    del guided_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: float(r["baseline_full_volume_ema_dice"]))
    fusion_summary = summarize(rows, args.failure_dice, args.high_quality_dice)

    baseline_rows = []
    oracle_dices = []
    for case_id in sorted(cases_by_id):
        b = baseline_by_id[case_id]
        baseline_rows.append({
            "dice": float(b["dice"]),
            "precision": float(b["precision"]),
            "recall": float(b["recall"]),
            "signed_rve_pct": float(b["signed_rve_pct"]),
        })
        oracle_dices.append(float(oracle_by_id[case_id]["oracle_roi_ema_dice"]))

    baseline_summary = summarize(baseline_rows, args.failure_dice, args.high_quality_dice)
    oracle_dices = np.asarray(oracle_dices, dtype=float)
    coverages = np.asarray([float(r["gt_coverage_by_predicted_roi"]) for r in rows], dtype=float)
    fusion_dices = np.asarray([float(r["fusion50_guided_roi_dice"]) for r in rows], dtype=float)
    full_dices = np.asarray([float(r["baseline_full_volume_ema_dice"]) for r in rows], dtype=float)

    oracle_mean = float(np.mean(oracle_dices))
    full_mean = float(baseline_summary["mean_dice"])
    fusion_mean = float(fusion_summary["mean_dice"])
    available = oracle_mean - full_mean
    recovered = (fusion_mean - full_mean) / available if abs(available) > 1e-12 else float("nan")

    case_csv = output_dir / "external31_guided_fusion50_case_comparison.csv"
    write_csv(case_csv, rows)
    payload = {
        "version": "external31_guided_refiner_fusion50_oof47_2ch_v1",
        "n_cases": len(rows),
        "full_checkpoint": str(full_checkpoint),
        "guided_checkpoint": str(guided_checkpoint),
        "fusion_rule": {
            "outside_roi": "keep Final62 EMA probability",
            "inside_roi_full_weight": FUSION_WEIGHT_FULL,
            "inside_roi_guided_weight": FUSION_WEIGHT_GUIDED,
            "segmentation_threshold": float(args.seg_threshold),
            "weight_sweep_performed": False,
        },
        "baseline_full_volume_ema": baseline_summary,
        "fusion50_guided_roi": fusion_summary,
        "oracle_roi_reference_mean_dice": oracle_mean,
        "delta_mean_dice_fusion50_vs_full": fusion_mean - full_mean,
        "fraction_of_oracle_mean_gain_recovered": float(recovered),
        "fusion50_wins_cases": int(np.sum(fusion_dices > full_dices + 1e-8)),
        "fusion50_wins_ge_0p02": int(np.sum((fusion_dices - full_dices) >= 0.02)),
        "fusion50_wins_ge_0p05": int(np.sum((fusion_dices - full_dices) >= 0.05)),
        "fusion50_losses_le_minus0p02": int(np.sum((fusion_dices - full_dices) <= -0.02)),
        "fusion50_losses_le_minus0p05": int(np.sum((fusion_dices - full_dices) <= -0.05)),
        "predicted_roi_gt_coverage_median": float(np.median(coverages)),
        "predicted_roi_gt_coverage_lt_090": int(np.sum(coverages < 0.90)),
        "predicted_roi_gt_coverage_lt_050": int(np.sum(coverages < 0.50)),
        "warning": (
            "External31 has been repeatedly examined and is a development benchmark. This is one predeclared "
            "50/50 fusion diagnostic only; no weight sweep was performed. Production remains single-stage."
        ),
    }
    summary_path = output_dir / "external31_guided_fusion50_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 126)
    print("GUIDED ROI CONSERVATIVE FUSION50 — EXTERNAL31 RESULT")
    print(f"Full-volume EMA mean Dice:      {full_mean:.4f}")
    print(f"Fusion50 mean Dice:             {fusion_mean:.4f}")
    print(f"Oracle-ROI EMA mean Dice:       {oracle_mean:.4f}")
    print(f"Fusion50 vs full change:        {fusion_mean - full_mean:+.4f}")
    print(f"Oracle gain recovered:          {recovered:.3f}")
    print(f"Median Dice full/fusion/oracle: {np.median(full_dices):.4f} / {np.median(fusion_dices):.4f} / {np.median(oracle_dices):.4f}")
    print(f"Dice<.70 full -> fusion50:      {int(np.sum(full_dices < 0.70))} -> {int(np.sum(fusion_dices < 0.70))}")
    print(f"Dice>=.80 full -> fusion50:     {int(np.sum(full_dices >= 0.80))} -> {int(np.sum(fusion_dices >= 0.80))}")
    print(f"Fusion50 wins cases:            {int(np.sum(fusion_dices > full_dices + 1e-8))}/{len(rows)}")
    print(f"Fusion50 wins >=+.02:           {int(np.sum((fusion_dices-full_dices) >= 0.02))}/{len(rows)}")
    print(f"Fusion50 wins >=+.05:           {int(np.sum((fusion_dices-full_dices) >= 0.05))}/{len(rows)}")
    print(f"Fusion50 losses <=-.02:         {int(np.sum((fusion_dices-full_dices) <= -0.02))}/{len(rows)}")
    print(f"Fusion50 losses <=-.05:         {int(np.sum((fusion_dices-full_dices) <= -0.05))}/{len(rows)}")
    print(f"Median GT coverage by pred ROI: {np.median(coverages):.3f}")
    print(f"ROI GT coverage <.90:           {int(np.sum(coverages < 0.90))}/{len(rows)}")
    print("\nPreviously low-Dice cases:")
    for row in rows:
        if float(row["baseline_full_volume_ema_dice"]) < args.failure_dice:
            print(
                f"  {row['case_id']} | full={float(row['baseline_full_volume_ema_dice']):.4f} "
                f"-> fusion50={float(row['fusion50_guided_roi_dice']):.4f} | "
                f"oracle={float(row['oracle_roi_ema_dice']):.4f} | "
                f"coverage={float(row['gt_coverage_by_predicted_roi']):.3f}"
            )
    print(f"\nCases:   {case_csv}")
    print(f"Summary: {summary_path}")
    print("=" * 126)


if __name__ == "__main__":
    main()
