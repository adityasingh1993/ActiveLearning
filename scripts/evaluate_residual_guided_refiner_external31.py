#!/usr/bin/env python3
"""Evaluate the final residual guided-ROI experiment on external31.

This evaluator is frozen to the already selected predicted-ROI proposal and threshold 0.50.
It does not tune residual strength, fusion weight, ROI thresholds, or segmentation threshold.

Inference
---------
1. Final62 EMA produces a native full-volume probability map.
2. The existing training-only prior + frozen candidate rule proposes the ROI.
3. The residual refiner receives [ultrasound ROI, Final62 EMA probability ROI].
4. The refiner internally predicts:
       corrected_logit = coarse_logit + delta_logit
5. Outside the ROI, Final62 EMA probability remains unchanged.
6. Inside the ROI, the corrected residual-refiner probability replaces the coarse probability.
7. Threshold is fixed at 0.50.

External GT is evaluation-only. Production remains single-stage; this is an offline AL/refinement
experiment and is the final planned ROI-model variant before returning to the base model.
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
    raise ImportError("evaluate_residual_guided_refiner_external31.py requires SimpleITK") from exc

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
    DEFAULT_ORACLE_CASES,
    DEFAULT_POOL_MANIFEST,
    DEFAULT_ROUND2_AUDIT,
    DEFAULT_ROUND2_AUDIT_CSV,
    build_guided_infer_transform,
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
from scripts.residual_guided_refiner_utils import load_residual_teacher
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)


DEFAULT_RESIDUAL_CHECKPOINT = Path(
    "experiments/residual_guided_refiner_oof47_2ch_v1/checkpoints/final_checkpoint.pth"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_residual_guided_refiner_oof47_2ch_v1")


def replace_roi_probability(full_prob_zyx, corrected_crop_prob_zyx, start_xyz, size_xyz):
    """Keep full EMA outside ROI; replace only the ROI with corrected residual probability."""
    full = np.asarray(full_prob_zyx, dtype=np.float32)
    corrected = np.asarray(corrected_crop_prob_zyx, dtype=np.float32)
    x0, y0, z0 = [int(v) for v in start_xyz]
    sx, sy, sz = [int(v) for v in size_xyz]
    expected = (sz, sy, sx)
    if tuple(corrected.shape) != expected:
        raise RuntimeError(
            f"Corrected crop shape mismatch: got={tuple(corrected.shape)}, expected={expected}, sizeXYZ={size_xyz}"
        )
    z1, y1, x1 = z0 + sz, y0 + sy, x0 + sx
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 > full.shape[0] or y1 > full.shape[1] or x1 > full.shape[2]:
        raise RuntimeError(
            f"ROI outside full probability: startXYZ={start_xyz}, sizeXYZ={size_xyz}, fullZYX={full.shape}"
        )
    out = full.copy()
    out[z0:z1, y0:y1, x0:x1] = corrected
    return np.clip(out, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description="Frozen residual guided ROI evaluation on external31")
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--full-checkpoint", default=str(DEFAULT_FULL_CHECKPOINT))
    p.add_argument("--residual-checkpoint", default=str(DEFAULT_RESIDUAL_CHECKPOINT))
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
        p.error("Residual guided evaluation is frozen at segmentation threshold 0.50")
    thresholds = parse_thresholds(args.candidate_thresholds)

    full_checkpoint = Path(args.full_checkpoint)
    residual_checkpoint = Path(args.residual_checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    audit_csv = Path(args.round2_audit_csv)
    baseline_path = Path(args.baseline_ema_csv)
    oracle_path = Path(args.oracle_case_csv)
    output_dir = Path(args.output_dir)

    for path in [full_checkpoint, residual_checkpoint, pool_manifest, audit_path, audit_csv, baseline_path, oracle_path]:
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
        raise RuntimeError("Residual guided evaluation requires prototype DynUNet configuration")
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
    print("RESIDUAL 2-CHANNEL GUIDED ROI REFINER — FROZEN EXTERNAL31 DEVELOPMENT EVALUATION")
    print(f"Cases:                {len(cases)}")
    print("Coarse source:        Final62 full-volume EMA")
    print("Residual rule:        corrected_logit = coarse_logit + learned delta_logit")
    print("Outside predicted ROI: Final62 EMA probability unchanged")
    print("Inside predicted ROI:  residual-corrected probability replaces coarse probability")
    print("Seg threshold:        0.50 fixed")
    print(f"Physical GPU:         {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("External GT is evaluation-only and never selects/repairs the ROI.")
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

            full_prob_image = sitk.GetImageFromArray(np.asarray(full_prob_zyx, dtype=np.float32))
            full_prob_image.CopyInformation(source_image)
            full_prob_path = full_prob_dir / f"{case_id}.ema_prob.mha"
            sitk.WriteImage(full_prob_image, str(full_prob_path), useCompression=True)

            image_crop = sitk.RegionOfInterest(source_image, size=size_xyz, index=start_xyz)
            coarse_crop = sitk.RegionOfInterest(full_prob_image, size=size_xyz, index=start_xyz)
            image_crop_path = crop_image_dir / f"{case_id}.mha"
            coarse_crop_path = crop_coarse_dir / f"{case_id}.ema_prob.mha"
            sitk.WriteImage(image_crop, str(image_crop_path), useCompression=True)
            sitk.WriteImage(coarse_crop, str(coarse_crop_path), useCompression=True)

            full_vox = max(int(np.prod(source_image.GetSize())), 1)
            crop_vox = int(np.prod(size_xyz))
            crop_meta[case_id] = {
                "image_crop_path": str(image_crop_path),
                "coarse_crop_path": str(coarse_crop_path),
                "full_probability_path": str(full_prob_path),
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
    residual_teacher = load_residual_teacher(config, residual_checkpoint, device)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(roi_loader, start=1):
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            case = cases_by_id[case_id]
            meta = crop_meta[case_id]
            guided_t = torch.cat([batch["image"], batch["coarse"]], dim=1).to(device)
            if guided_t.shape[1] != 2:
                raise RuntimeError(f"Expected 2 residual-guided channels, got {guided_t.shape}")

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                corrected_prob_t = torch.sigmoid(cv.main_prediction(inferer(guided_t, residual_teacher)))

            native_corrected = invert_probability_exact(corrected_prob_t, batch, roi_inverse, index=0)
            _, corrected_crop_prob_zyx = normalize_native_probability(native_corrected, meta["image_crop_path"])

            full_prob_img = sitk.ReadImage(meta["full_probability_path"])
            full_prob_zyx = sitk.GetArrayFromImage(full_prob_img).astype(np.float32, copy=False)
            full_source = sitk.ReadImage(str(case["image"]))
            gt = read_gt_binary(case["gt_path"], full_source)

            start_xyz = (meta["roi_start_x"], meta["roi_start_y"], meta["roi_start_z"])
            size_xyz = (meta["roi_size_x"], meta["roi_size_y"], meta["roi_size_z"])
            final_prob = replace_roi_probability(
                full_prob_zyx,
                corrected_crop_prob_zyx,
                start_xyz,
                size_xyz,
            )
            final_pred = final_prob > float(args.seg_threshold)
            metrics = binary_metrics(final_pred, gt)

            full_dice = float(baseline_by_id[case_id]["dice"])
            oracle_dice = float(oracle_by_id[case_id]["oracle_roi_ema_dice"])
            rows.append({
                "case_id": case_id,
                "baseline_full_volume_ema_dice": full_dice,
                "residual_guided_roi_ema_dice": float(metrics["dice"]),
                "oracle_roi_ema_dice": oracle_dice,
                "residual_minus_full_dice": float(metrics["dice"]) - full_dice,
                "residual_minus_oracle_dice": float(metrics["dice"]) - oracle_dice,
                **meta,
                **metrics,
            })
            print(
                f"[correct  {index:2d}/{len(cases)}] {case_id} | full={full_dice:.4f} -> "
                f"residual={metrics['dice']:.4f} | oracle={oracle_dice:.4f} | "
                f"coverage={meta['gt_coverage_by_predicted_roi']:.3f}"
            )

    rows.sort(key=lambda r: float(r["baseline_full_volume_ema_dice"]))
    residual_summary = summarize(rows, args.failure_dice, args.high_quality_dice)

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
    residual_dices = np.asarray([float(r["residual_guided_roi_ema_dice"]) for r in rows], dtype=float)
    full_dices = np.asarray([float(r["baseline_full_volume_ema_dice"]) for r in rows], dtype=float)
    deltas = residual_dices - full_dices

    oracle_mean = float(np.mean(oracle_dices))
    full_mean = float(baseline_summary["mean_dice"])
    residual_mean = float(residual_summary["mean_dice"])
    available = oracle_mean - full_mean
    recovered = (residual_mean - full_mean) / available if abs(available) > 1e-12 else float("nan")

    case_csv = output_dir / "external31_residual_guided_case_comparison.csv"
    write_csv(case_csv, rows)
    payload = {
        "version": "external31_residual_guided_refiner_oof47_2ch_v1",
        "n_cases": len(rows),
        "full_checkpoint": str(full_checkpoint),
        "residual_checkpoint": str(residual_checkpoint),
        "baseline_full_volume_ema": baseline_summary,
        "residual_guided_roi_ema": residual_summary,
        "oracle_roi_reference_mean_dice": oracle_mean,
        "delta_mean_dice_residual_vs_full": residual_mean - full_mean,
        "fraction_of_oracle_mean_gain_recovered": float(recovered),
        "residual_wins_cases": int(np.sum(deltas > 1e-8)),
        "residual_wins_ge_0p02": int(np.sum(deltas >= 0.02)),
        "residual_wins_ge_0p05": int(np.sum(deltas >= 0.05)),
        "residual_losses_le_minus_0p02": int(np.sum(deltas <= -0.02)),
        "residual_losses_le_minus_0p05": int(np.sum(deltas <= -0.05)),
        "predicted_roi_gt_coverage_median": float(np.median(coverages)),
        "predicted_roi_gt_coverage_lt_090": int(np.sum(coverages < 0.90)),
        "predicted_roi_gt_coverage_lt_050": int(np.sum(coverages < 0.50)),
        "predeclared_stop_rule": {
            "mean_dice_lt_0p750": "stop ROI work",
            "mean_dice_0p750_to_lt_0p755": "small positive; document, do not pursue",
            "mean_dice_ge_0p755_and_dice_lt_0p70_le_6": "useful offline AL/refinement result",
            "mean_dice_ge_0p770": "strong ROI result",
            "desired_losses_le_minus_0p05": 0,
        },
        "warning": (
            "External31 has been repeatedly examined and is a development benchmark. No hyperparameter "
            "selection should be performed from this result. Production remains single-stage."
        ),
    }
    summary_path = output_dir / "external31_residual_guided_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 126)
    print("RESIDUAL GUIDED ROI REFINER — EXTERNAL31 RESULT")
    print(f"Full-volume EMA mean Dice:        {full_mean:.4f}")
    print(f"Residual guided ROI mean Dice:    {residual_mean:.4f}")
    print(f"Oracle-ROI EMA mean Dice:         {oracle_mean:.4f}")
    print(f"Residual vs full change:          {residual_mean - full_mean:+.4f}")
    print(f"Oracle gain recovered:            {recovered:.3f}")
    print(f"Median Dice full/residual/oracle: {np.median(full_dices):.4f} / {np.median(residual_dices):.4f} / {np.median(oracle_dices):.4f}")
    print(f"Dice<.70 full -> residual:        {int(np.sum(full_dices < 0.70))} -> {int(np.sum(residual_dices < 0.70))}")
    print(f"Dice>=.80 full -> residual:       {int(np.sum(full_dices >= 0.80))} -> {int(np.sum(residual_dices >= 0.80))}")
    print(f"Residual wins cases:              {int(np.sum(deltas > 1e-8))}/{len(rows)}")
    print(f"Residual wins >=+.02:             {int(np.sum(deltas >= 0.02))}/{len(rows)}")
    print(f"Residual wins >=+.05:             {int(np.sum(deltas >= 0.05))}/{len(rows)}")
    print(f"Residual losses <=-.02:           {int(np.sum(deltas <= -0.02))}/{len(rows)}")
    print(f"Residual losses <=-.05:           {int(np.sum(deltas <= -0.05))}/{len(rows)}")
    print(f"Median GT coverage by pred ROI:   {np.median(coverages):.3f}")
    print(f"ROI GT coverage <.90:             {int(np.sum(coverages < 0.90))}/{len(rows)}")
    print("\nPreviously low-Dice cases:")
    for row in rows:
        if float(row["baseline_full_volume_ema_dice"]) < args.failure_dice:
            print(
                f"  {row['case_id']} | full={float(row['baseline_full_volume_ema_dice']):.4f} "
                f"-> residual={float(row['residual_guided_roi_ema_dice']):.4f} | "
                f"oracle={float(row['oracle_roi_ema_dice']):.4f} | "
                f"coverage={float(row['gt_coverage_by_predicted_roi']):.3f}"
            )
    print(f"\nCases:   {case_csv}")
    print(f"Summary: {summary_path}")
    print("=" * 126)


if __name__ == "__main__":
    main()
