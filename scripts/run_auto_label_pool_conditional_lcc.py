#!/usr/bin/env python3
"""Run conditional-LCC segmentation + QC v2 over the unlabeled image pool.

This preserves the original run_auto_label_pool.py / QC v1 experiment unchanged.
For every unlabeled case:

1. run the final 47-label student + EMA teacher,
2. form the same 50/50 ensemble used during OOF development,
3. threshold at 0.50 to obtain the raw model-space mask,
4. apply conditional LCC only when raw components > 1 and the largest component
   contains at least the configured fraction of raw foreground (default 0.65),
5. extract the exact QC-v2 deployable schema, retaining raw fragmentation features
   while final-mask features describe the delivered post-processed mask,
6. score predicted Dice and P(failure) using the recalibrated QC-v2 bundle/policy,
7. invert the FINAL binary mask to native source geometry and write .seg.nrrd.

High-confidence outputs remain pseudo-label candidates only; this is development calibration.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import load_models
from scripts.build_oof_qc_dataset_conditional_lcc import (
    DEFAULT_DOMINANCE_THRESHOLD,
    extract_conditional_lcc_features,
)
from scripts.run_auto_label_pool import (
    BUCKET_DIR,
    assign_bucket,
    collect_pool_cases,
    invert_prediction_exact,
    load_policy,
    load_qc_bundle,
    main_prediction,
    qc_predict,
    read_json,
    verify_native_mask_before_write,
    verify_saved_geometry,
    write_csv,
)


DEFAULT_FINAL_CHECKPOINT = Path(
    "experiments/final_supervised_47_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_QC_DIR = Path("experiments/auto_label_qc_lcc_v2")
DEFAULT_QC_BUNDLE = DEFAULT_QC_DIR / "qc_model_bundle.pkl"
DEFAULT_POLICY = DEFAULT_QC_DIR / "qc_policy_development.json"
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_pool_lcc_v2")


def main():
    parser = argparse.ArgumentParser(
        description="Run conditional-LCC final segmentation + QC-v2 triage over unlabeled images"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None, help="Default: config.data_dir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_FINAL_CHECKPOINT))
    parser.add_argument("--qc-bundle", default=str(DEFAULT_QC_BUNDLE))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=DEFAULT_DOMINANCE_THRESHOLD,
        help="Apply LCC iff raw components >1 and largest raw component fraction >= this value",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: process first N cases")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if abs(float(args.seg_threshold) - 0.50) > 1e-8:
        parser.error("QC v2 was developed at segmentation threshold 0.50; do not change it")
    if not 0.0 <= args.dominance_threshold <= 1.0:
        parser.error("--dominance-threshold must be in [0,1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >=1")

    checkpoint = Path(args.checkpoint)
    qc_bundle_path = Path(args.qc_bundle)
    policy_path = Path(args.policy)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite only when intentionally replacing this entire v2 pool run."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError(
            "QC development used prototype student + EMA teacher; use compute_mode='prototype'."
        )
    cv.apply_baseline(config, args.resize_size, epochs=1)

    source_manifest = read_json(source_manifest_path)
    source_labeled_ids = set(source_manifest.get("all_case_ids", []))
    if len(source_labeled_ids) != 47:
        raise RuntimeError(
            f"Expected frozen source manifest with 47 labeled IDs, found {len(source_labeled_ids)}"
        )

    bundle = load_qc_bundle(qc_bundle_path)
    policy_payload, accept_p, accept_dice, active_p, active_dice = load_policy(policy_path)

    required_v2_features = {
        "raw_component_count",
        "raw_largest_component_fraction",
        "raw_second_component_fraction",
        "conditional_lcc_applied",
        "conditional_lcc_removed_fraction",
    }
    missing_v2 = required_v2_features - set(bundle["feature_columns"])
    if missing_v2:
        raise RuntimeError(
            "QC bundle is not the conditional-LCC v2 bundle. Missing features: "
            + str(sorted(missing_v2))
        )

    input_dir = Path(args.input_dir) if args.input_dir else Path(config.data_dir)
    cases, explicit_human = collect_pool_cases(
        config, input_dir, source_labeled_ids, limit=args.limit
    )

    # One concrete forward Compose instance is shared by Dataset and Invertd.
    # MONAI 1.5+ inversion depends on the matching MetaTensor transform trace.
    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=True,
        to_tensor=True,
    )
    loader = DataLoader(
        Dataset(cases, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError(
            "Final checkpoint has no EMA teacher; QC-v2 disagreement features require it."
        )
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    for dirname in BUCKET_DIR.values():
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    feature_rows = []
    case_path_by_id = {case["id"]: case["image"] for case in cases}

    print("=" * 118)
    print("UNLABELED AUTO-LABEL + CONDITIONAL LCC + QC V2")
    print(f"Input directory:       {input_dir}")
    print(f"Unlabeled cases:       {len(cases)}")
    print(f"Excluded source IDs:   {len(source_labeled_ids)}")
    print(f"Explicit human labels: {len(explicit_human)}")
    print(f"Checkpoint:            {checkpoint}")
    print(f"QC bundle:             {qc_bundle_path}")
    print(f"Policy:                {policy_path}")
    print(f"Conditional LCC gate:  components>1 AND largest fraction>={args.dominance_threshold:.2f}")
    print(f"Device:                {device}")
    print(
        f"High-confidence gate: P(failure)<={accept_p:.4f} AND predicted Dice>={accept_dice:.4f}"
    )
    print(
        f"Active priority gate: P(failure)>={active_p:.4f} OR predicted Dice<={active_dice:.4f}"
    )
    print("NOTE: high-confidence means pseudo-label candidate, not production auto-accept.")
    print("=" * 118)

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            image_t = batch["image"].to(device)
            case_raw = batch.get("id")
            case_id = case_raw[0] if isinstance(case_raw, (list, tuple)) else str(case_raw)
            source_path = case_path_by_id[case_id]

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(main_prediction(inferer(image_t, student)))
                teacher_prob_t = torch.sigmoid(main_prediction(inferer(image_t, teacher)))
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = teacher_prob_t[0, 0].detach().float().cpu().numpy()
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()

            features, raw_pred_np, final_pred_np, lcc_applied = extract_conditional_lcc_features(
                image_np,
                student_prob,
                teacher_prob,
                ensemble_prob,
                threshold=args.seg_threshold,
                dominance_threshold=args.dominance_threshold,
            )
            predicted_dice, failure_probability = qc_predict(bundle, features)
            bucket = assign_bucket(
                predicted_dice,
                failure_probability,
                accept_p,
                accept_dice,
                active_p,
                active_dice,
            )

            # IMPORTANT: do not recreate the final mask with torch.from_numpy(). That produces
            # a plain Tensor and discards MONAI MetaTensor.applied_operations, so Invertd cannot
            # undo Resize/Spacing/Orientation. Clone the already traced ensemble MetaTensor and
            # replace only its voxel values with the conditional-LCC binary mask.
            final_pred_t = ensemble_prob_t.clone()
            final_values_t = torch.as_tensor(
                final_pred_np[None, None],
                dtype=final_pred_t.dtype,
                device=final_pred_t.device,
            )
            final_pred_t.copy_(final_values_t)

            native_pred = invert_prediction_exact(
                final_pred_t,
                batch,
                inverse_transform,
                index=0,
            )
            reference_image, native_pred_sitk = verify_native_mask_before_write(
                native_pred, source_path
            )
            bucket_dir = output_dir / BUCKET_DIR[bucket]
            seg_path = bucket_dir / f"{case_id}_pred{config.label_suffix}"
            write_mask_with_spatial_geometry(
                str(seg_path), native_pred_sitk, reference_image_path=source_path
            )
            verify_saved_geometry(seg_path, reference_image)

            probability_path = ""
            if args.save_probabilities:
                probability_path = str(probability_dir / f"{case_id}.npz")
                np.savez_compressed(
                    probability_path,
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=teacher_prob.astype(np.float32),
                    ensemble_prob=ensemble_prob.astype(np.float32),
                    raw_prediction_model_space=raw_pred_np.astype(np.uint8),
                    final_prediction_model_space=final_pred_np.astype(np.uint8),
                    conditional_lcc_applied=np.asarray([int(lcc_applied)], dtype=np.uint8),
                )

            row = {
                "case_id": case_id,
                "image_path": str(source_path),
                "segmentation_path": str(seg_path),
                "prediction_source": "student_teacher_50_50_ensemble",
                "segmentation_threshold": float(args.seg_threshold),
                "conditional_lcc_dominance_threshold": float(args.dominance_threshold),
                "conditional_lcc_applied": int(lcc_applied),
                "raw_component_count": int(features["raw_component_count"]),
                "raw_largest_component_fraction": float(features["raw_largest_component_fraction"]),
                "raw_pred_vox_model_space": int(features["raw_pred_vox"]),
                "final_pred_vox_model_space": int(features["pred_vox"]),
                "conditional_lcc_removed_fraction": float(features["conditional_lcc_removed_fraction"]),
                "predicted_dice": predicted_dice,
                "predicted_failure_probability": failure_probability,
                "qc_bucket": bucket,
                "student_teacher_mae": float(features["student_teacher_mae"]),
                "final_component_count": int(features["component_count"]),
                "probability_npz": probability_path,
            }
            manifest_rows.append(row)

            frow = {"case_id": case_id, "image_path": str(source_path)}
            frow.update(features)
            frow["predicted_dice"] = predicted_dice
            frow["predicted_failure_probability"] = failure_probability
            frow["qc_bucket"] = bucket
            feature_rows.append(frow)

            lcc_flag = "LCC" if lcc_applied else "RAW"
            print(
                f"[{index:4d}/{len(cases)}] {case_id} | {lcc_flag:3s} | "
                f"raw comps={int(features['raw_component_count']):4d} | "
                f"largest={float(features['raw_largest_component_fraction']):.1%} | "
                f"pred Dice={predicted_dice:.4f} | P(fail)={failure_probability:.4f} | {bucket}"
            )

    active_rows = [r for r in manifest_rows if r["qc_bucket"] == "ACTIVE_LEARN_PRIORITY"]
    active_rows.sort(
        key=lambda r: (
            -float(r["predicted_failure_probability"]),
            float(r["predicted_dice"]),
            str(r["case_id"]),
        )
    )
    active_rank = {r["case_id"]: i + 1 for i, r in enumerate(active_rows)}
    for row in manifest_rows:
        row["active_learning_rank"] = active_rank.get(row["case_id"], "")

    bucket_order = {
        "ACTIVE_LEARN_PRIORITY": 0,
        "REVIEW": 1,
        "HIGH_CONFIDENCE_PSEUDO_LABEL": 2,
    }
    manifest_rows.sort(
        key=lambda r: (
            bucket_order[r["qc_bucket"]],
            int(r["active_learning_rank"]) if r["active_learning_rank"] != "" else 10**9,
            -float(r["predicted_failure_probability"]),
            str(r["case_id"]),
        )
    )
    feature_rows.sort(key=lambda r: str(r["case_id"]))

    manifest_path = output_dir / "auto_label_manifest.csv"
    feature_path = output_dir / "unlabeled_qc_features.csv"
    write_csv(manifest_path, manifest_rows)
    feature_fields = ["case_id", "image_path"] + list(bundle["feature_columns"]) + [
        "predicted_dice", "predicted_failure_probability", "qc_bucket"
    ]
    write_csv(feature_path, feature_rows, feature_fields)

    counts = {
        bucket: int(sum(r["qc_bucket"] == bucket for r in manifest_rows))
        for bucket in BUCKET_DIR
    }
    lcc_count = int(sum(int(r["conditional_lcc_applied"]) for r in manifest_rows))
    dice_values = np.asarray([float(r["predicted_dice"]) for r in manifest_rows])
    failure_values = np.asarray(
        [float(r["predicted_failure_probability"]) for r in manifest_rows]
    )

    metadata = {
        "version": "auto_label_pool_conditional_lcc_v2",
        "input_dir": str(input_dir),
        "n_processed": len(manifest_rows),
        "n_source_labeled_excluded": len(source_labeled_ids),
        "n_explicit_human_labeled_excluded_in_input_scan": len(explicit_human),
        "checkpoint": str(checkpoint),
        "qc_bundle": str(qc_bundle_path),
        "policy": str(policy_path),
        "source_manifest": str(source_manifest_path),
        "feature_schema": list(bundle["feature_columns"]),
        "segmentation_threshold": float(args.seg_threshold),
        "conditional_lcc": {
            "dominance_threshold": float(args.dominance_threshold),
            "connectivity": "6-connected foreground",
            "rule": "apply iff raw component_count >1 and raw largest component fraction >= threshold",
            "n_applied": lcc_count,
        },
        "prediction_source": "student_teacher_50_50_ensemble",
        "saved_mask": "final conditional-LCC mask",
        "native_geometry_validation": (
            "exact forward Compose + traced MetaTensor carrier for Invertd; MONAI XYZ to "
            "SimpleITK ZYX axis-order normalization without resampling; strict saved geometry check"
        ),
        "policy_thresholds": {
            "high_confidence_failure_probability_lte": accept_p,
            "high_confidence_predicted_dice_gte": accept_dice,
            "active_learning_failure_probability_gte": active_p,
            "active_learning_predicted_dice_lte": active_dice,
        },
        "bucket_counts": counts,
        "pool_predicted_dice_mean": float(np.mean(dice_values)),
        "pool_failure_probability_mean": float(np.mean(failure_values)),
        "arrays_saved": bool(args.save_probabilities),
        "development_policy_warning": policy_payload.get(
            "warning",
            "Development calibration only; validate on a future locked set before production auto-accept.",
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("CONDITIONAL-LCC AUTO-LABEL POOL COMPLETE")
    print(f"Processed:                     {len(manifest_rows)}")
    print(f"Conditional LCC applied:       {lcc_count}/{len(manifest_rows)}")
    print(f"HIGH_CONFIDENCE_PSEUDO_LABEL: {counts['HIGH_CONFIDENCE_PSEUDO_LABEL']}")
    print(f"REVIEW:                        {counts['REVIEW']}")
    print(f"ACTIVE_LEARN_PRIORITY:         {counts['ACTIVE_LEARN_PRIORITY']}")
    print(f"Mean predicted Dice:           {np.mean(dice_values):.4f}")
    print(f"Mean P(failure):               {np.mean(failure_values):.4f}")
    print(f"Manifest:                      {manifest_path}")
    print(f"QC features:                   {feature_path}")
    print(f"Metadata:                      {metadata_path}")
    if active_rows:
        print("\nTop ACTIVE_LEARN_PRIORITY cases:")
        for row in active_rows[:10]:
            print(
                f"  #{active_rank[row['case_id']]:02d} {row['case_id']} | "
                f"pred Dice={float(row['predicted_dice']):.4f} | "
                f"P(fail)={float(row['predicted_failure_probability']):.4f} | "
                f"LCC={bool(row['conditional_lcc_applied'])}"
            )
    print("\nCompare this bucket distribution against raw-mask v1: 21 high-confidence / 17 review / 18 active.")
    print("Do not reuse the previous 10-case annotation batch; re-select after this v2 run.")
    print("=" * 118)


if __name__ == "__main__":
    main()
