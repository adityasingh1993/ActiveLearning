#!/usr/bin/env python3
"""Evaluate the 2-channel OOF47 guided ROI refiner on frozen external31 development data.

Deployable inputs to the OFFLINE refiner:
- Final62 full-volume EMA probability map,
- ROI proposed from that probability map + the training-only localization prior,
- ultrasound ROI + coarse-probability ROI as two channels.

External GT is used only after ROI proposal for evaluation. Production remains single-stage.
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
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("evaluate_guided_refiner_oof47_2ch_external31.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import ORIENTATIOND_RAS_LABELS, build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.evaluate_oracle_roi_external31 import baseline_ema_rows, summarize
from scripts.evaluate_predicted_roi_refinement_external31 import (
    build_training_target_prior,
    choose_localization_candidate,
    crop_gt_coverage,
    expanded_roi_from_candidate,
    oracle_case_rows,
    parse_thresholds,
)
from scripts.oracle_roi_utils import paste_crop_into_full
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
DEFAULT_GUIDED_CHECKPOINT = Path(
    "experiments/guided_refiner_oof47_2ch_v1/checkpoints/final_checkpoint.pth"
)
DEFAULT_POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
DEFAULT_ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_ROUND2_AUDIT_CSV = Path("experiments/round2_supervised_62_translation12/round2_label_audit.csv")
DEFAULT_BASELINE_CASES = Path(
    "experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv"
)
DEFAULT_ORACLE_CASES = Path(
    "experiments/external31_oracle_roi_final62_v1/external31_oracle_roi_case_comparison.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_guided_refiner_oof47_2ch_v1")


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
            writer.writerow({k: row.get(k, "") for k in fields})


def build_guided_dynunet(num_classes, dropout):
    return DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=num_classes,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 4,
        filters=[16, 32, 64, 128, 256],
        dropout=dropout,
        norm_name="instance",
        deep_supervision=True,
    )


def load_guided_teacher(config, checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "teacher" not in state:
        raise RuntimeError(f"Guided checkpoint has no EMA teacher: {checkpoint}")
    teacher = build_guided_dynunet(config.num_classes, config.dropout).to(device)
    teacher.load_state_dict(state["teacher"])
    teacher.eval()
    return teacher


def build_guided_infer_transform(config, resize_size):
    keys = ["image", "coarse"]
    steps = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=ORIENTATIOND_RAS_LABELS),
    ]
    if config.spacing and all(float(x) >= 0.1 for x in config.spacing):
        steps.append(Spacingd(keys=keys, pixdim=config.spacing, mode=("bilinear", "bilinear")))
    steps.extend([
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True, channel_wise=True
        ),
        Resized(
            keys=keys,
            spatial_size=(int(resize_size),) * 3,
            mode=("trilinear", "trilinear"),
        ),
    ])
    return Compose(steps)


def main():
    p = argparse.ArgumentParser(description="Evaluate OOF47 2-channel guided refiner on external31")
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
        p.error("Guided feasibility evaluation is frozen at threshold 0.50")
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
    crop_image_dir.mkdir(parents=True, exist_ok=True)
    crop_coarse_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Guided evaluation requires prototype DynUNet configuration")
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

    print("=" * 124)
    print("2-CHANNEL GUIDED ROI REFINER — FROZEN EXTERNAL31 DEVELOPMENT EVALUATION")
    print(f"Cases:                {len(cases)}")
    print("Coarse source:        Final62 full-volume EMA")
    print("Guided channels:      ultrasound ROI + Final62 EMA probability ROI")
    print(f"Physical GPU:         {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("External GT is evaluation-only and never selects/repairs the ROI.")
    print("=" * 124)

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
            _, crop_prob_zyx = normalize_native_probability(native_prob, meta["image_crop_path"])
            crop_pred = crop_prob_zyx > float(args.seg_threshold)

            full_source = sitk.ReadImage(str(case["image"]))
            gt = read_gt_binary(case["gt_path"], full_source)
            start_xyz = (meta["roi_start_x"], meta["roi_start_y"], meta["roi_start_z"])
            size_xyz = (meta["roi_size_x"], meta["roi_size_y"], meta["roi_size_z"])
            full_pred = paste_crop_into_full(crop_pred, gt.shape, start_xyz, size_xyz).astype(bool)
            metrics = binary_metrics(full_pred, gt)

            full_dice = float(baseline_by_id[case_id]["dice"])
            oracle_dice = float(oracle_by_id[case_id]["oracle_roi_ema_dice"])
            rows.append({
                "case_id": case_id,
                "baseline_full_volume_ema_dice": full_dice,
                "guided_roi_2ch_ema_dice": float(metrics["dice"]),
                "oracle_roi_ema_dice": oracle_dice,
                "guided_minus_full_dice": float(metrics["dice"]) - full_dice,
                "guided_minus_oracle_dice": float(metrics["dice"]) - oracle_dice,
                **meta,
                **metrics,
            })
            print(
                f"[refine   {index:2d}/{len(cases)}] {case_id} | full={full_dice:.4f} -> "
                f"guided={metrics['dice']:.4f} | oracle={oracle_dice:.4f} | "
                f"coverage={meta['gt_coverage_by_predicted_roi']:.3f}"
            )

    rows.sort(key=lambda r: float(r["baseline_full_volume_ema_dice"]))
    guided_summary = summarize(rows, args.failure_dice, args.high_quality_dice)

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
    guided_dices = np.asarray([float(r["guided_roi_2ch_ema_dice"]) for r in rows], dtype=float)
    full_dices = np.asarray([float(r["baseline_full_volume_ema_dice"]) for r in rows], dtype=float)

    oracle_mean = float(np.mean(oracle_dices))
    full_mean = float(baseline_summary["mean_dice"])
    guided_mean = float(guided_summary["mean_dice"])
    available = oracle_mean - full_mean
    recovered = (guided_mean - full_mean) / available if abs(available) > 1e-12 else float("nan")

    case_csv = output_dir / "external31_guided_refiner_case_comparison.csv"
    write_csv(case_csv, rows)
    payload = {
        "version": "external31_guided_refiner_oof47_2ch_v1",
        "n_cases": len(rows),
        "full_checkpoint": str(full_checkpoint),
        "guided_checkpoint": str(guided_checkpoint),
        "baseline_full_volume_ema": baseline_summary,
        "guided_roi_2ch_ema": guided_summary,
        "oracle_roi_reference_mean_dice": oracle_mean,
        "delta_mean_dice_guided_vs_full": guided_mean - full_mean,
        "fraction_of_oracle_mean_gain_recovered": float(recovered),
        "guided_wins_cases": int(np.sum(guided_dices > full_dices + 1e-8)),
        "guided_wins_ge_0p02": int(np.sum((guided_dices - full_dices) >= 0.02)),
        "guided_wins_ge_0p05": int(np.sum((guided_dices - full_dices) >= 0.05)),
        "predicted_roi_gt_coverage_median": float(np.median(coverages)),
        "predicted_roi_gt_coverage_lt_090": int(np.sum(coverages < 0.90)),
        "predicted_roi_gt_coverage_lt_050": int(np.sum(coverages < 0.50)),
        "warning": (
            "External31 has been repeatedly examined and is a development benchmark. The 2-channel refiner is "
            "offline-only; production remains a single-stage model."
        ),
    }
    summary_path = output_dir / "external31_guided_refiner_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("GUIDED ROI REFINER — EXTERNAL31 RESULT")
    print(f"Full-volume EMA mean Dice:      {full_mean:.4f}")
    print(f"Guided 2ch ROI mean Dice:       {guided_mean:.4f}")
    print(f"Oracle-ROI EMA mean Dice:       {oracle_mean:.4f}")
    print(f"Guided vs full change:          {guided_mean - full_mean:+.4f}")
    print(f"Oracle gain recovered:          {recovered:.3f}")
    print(f"Median Dice full/guided/oracle: {np.median(full_dices):.4f} / {np.median(guided_dices):.4f} / {np.median(oracle_dices):.4f}")
    print(f"Dice<.70 full -> guided:        {int(np.sum(full_dices < 0.70))} -> {int(np.sum(guided_dices < 0.70))}")
    print(f"Dice>=.80 full -> guided:       {int(np.sum(full_dices >= 0.80))} -> {int(np.sum(guided_dices >= 0.80))}")
    print(f"Guided wins cases:              {int(np.sum(guided_dices > full_dices + 1e-8))}/{len(rows)}")
    print(f"Guided wins >=+.02:             {int(np.sum((guided_dices-full_dices) >= 0.02))}/{len(rows)}")
    print(f"Guided wins >=+.05:             {int(np.sum((guided_dices-full_dices) >= 0.05))}/{len(rows)}")
    print(f"Median GT coverage by pred ROI: {np.median(coverages):.3f}")
    print(f"ROI GT coverage <.90:           {int(np.sum(coverages < 0.90))}/{len(rows)}")
    print("\nPreviously low-Dice cases:")
    for row in rows:
        if float(row["baseline_full_volume_ema_dice"]) < args.failure_dice:
            print(
                f"  {row['case_id']} | full={float(row['baseline_full_volume_ema_dice']):.4f} "
                f"-> guided={float(row['guided_roi_2ch_ema_dice']):.4f} | "
                f"oracle={float(row['oracle_roi_ema_dice']):.4f} | "
                f"coverage={float(row['gt_coverage_by_predicted_roi']):.3f}"
            )
    print(f"\nCases:   {case_csv}")
    print(f"Summary: {summary_path}")
    print("=" * 124)


if __name__ == "__main__":
    main()
