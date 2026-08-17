#!/usr/bin/env python3
"""Build leak-safe OOF QC data for the conditional-LCC post-processing candidate.

This is a separate v2 development dataset. The original raw-mask QC v1 remains unchanged.

Prediction path
---------------
1. Frozen OOF student + EMA-teacher 50/50 ensemble.
2. Threshold probability at 0.50 to obtain the RAW mask.
3. If RAW has >1 component and its largest component contains at least 65% of RAW
   foreground voxels, keep that largest component; otherwise retain RAW unchanged.
4. QC targets (Dice/failure/high-quality) are calculated from the FINAL delivered mask.

Feature semantics
-----------------
The original deployment-only feature schema is retained, but mask-dependent fields such as
pred_vox, pred_fg_fraction, component/geometry/boundary/predicted-region intensity fields
are recalculated on the FINAL mask. Additional raw_* features preserve fragmentation
information from the network output before post-processing.

GT information remains target_* only and is never a deployable input.

Default output:
  experiments/auto_label_qc_lcc_v2/
    oof_qc_dataset.csv
    oof_qc_metadata.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import (
    _boundary_features,
    _component_features,
    _finite_stats,
    _geometry_features,
    _intensity_features,
    extract_features,
    load_models,
)
from scripts.evaluate_lcc_oof import keep_largest_component


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_qc_lcc_v2")
DEFAULT_DOMINANCE_THRESHOLD = 0.65


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_conditional_lcc_features(
    image,
    student_prob,
    teacher_prob,
    ensemble_prob,
    threshold=0.50,
    dominance_threshold=DEFAULT_DOMINANCE_THRESHOLD,
):
    """Return deployable QC features plus raw/final masks for conditional LCC.

    The base feature extractor is first run on the raw thresholded mask. We then preserve
    explicit raw fragmentation fields and overwrite all final-mask-dependent fields with
    statistics from the conditional-LCC result.
    """
    features, raw_pred = extract_features(
        image,
        student_prob,
        teacher_prob,
        ensemble_prob,
        threshold,
    )
    raw_pred = np.asarray(raw_pred).astype(np.uint8, copy=False)

    lcc_pred, raw_component_count, raw_largest_vox, raw_largest_fraction = (
        keep_largest_component(raw_pred)
    )
    apply_lcc = bool(
        raw_component_count > 1
        and float(raw_largest_fraction) >= float(dominance_threshold)
    )
    final_pred = lcc_pred if apply_lcc else raw_pred
    final_pred = np.asarray(final_pred).astype(np.uint8, copy=False)

    raw_vox = int(raw_pred.sum())
    final_vox = int(final_pred.sum())
    raw_component_features = _component_features(raw_pred)

    # Preserve pre-postprocess fragmentation information. These are fully deployable.
    features.update({
        "raw_pred_vox": raw_vox,
        "raw_pred_fg_fraction": float(raw_vox / raw_pred.size),
        "raw_component_count": int(raw_component_features["component_count"]),
        "raw_components_ge_100": int(raw_component_features["components_ge_100"]),
        "raw_components_ge_1000": int(raw_component_features["components_ge_1000"]),
        "raw_largest_component_vox": int(raw_component_features["largest_component_vox"]),
        "raw_largest_component_fraction": float(raw_component_features["largest_component_fraction"]),
        "raw_second_component_fraction": float(raw_component_features["second_component_fraction"]),
        "conditional_lcc_applied": int(apply_lcc),
        "conditional_lcc_removed_fraction": (
            float((raw_vox - final_vox) / raw_vox) if raw_vox > 0 else 0.0
        ),
    })

    # Existing mask-dependent feature names now describe the FINAL delivered mask.
    features["pred_vox"] = final_vox
    features["pred_fg_fraction"] = float(final_vox / final_pred.size)
    if final_vox > 0:
        features.update(_finite_stats(ensemble_prob[final_pred.astype(bool)], "pred_region_prob"))
    else:
        features.update(_finite_stats([], "pred_region_prob"))
    features.update(_component_features(final_pred))
    features.update(_geometry_features(final_pred))
    features.update(_boundary_features(final_pred, ensemble_prob))
    features.update(_intensity_features(image, final_pred))

    return features, raw_pred, final_pred, apply_lcc


def verify_raw_against_frozen(rows, result_path, tolerance):
    """Verify the replayed RAW predictions still match the frozen Round-0 OOF results."""
    frozen_rows = read_csv(result_path)
    frozen = {row["case_id"]: row for row in frozen_rows}
    if len(frozen) != len(rows):
        raise RuntimeError(
            f"Frozen results contain {len(frozen)} cases; regenerated v2 dataset has {len(rows)}"
        )

    failures = []
    for row in rows:
        case_id = row["case_id"]
        expected = frozen.get(case_id)
        if expected is None:
            failures.append(f"{case_id}: missing from frozen results")
            continue
        if expected.get("source") != "ensemble":
            failures.append(f"{case_id}: frozen source={expected.get('source')} (expected ensemble)")
            continue
        try:
            expected_threshold = float(expected["threshold"])
            expected_dice = float(expected["dice"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{case_id}: malformed frozen row")
            continue
        if abs(expected_threshold - float(row["threshold"])) > 1e-8:
            failures.append(
                f"{case_id}: threshold frozen={expected_threshold} regenerated={row['threshold']}"
            )
        if abs(expected_dice - float(row["target_raw_dice"])) > tolerance:
            failures.append(
                f"{case_id}: raw Dice frozen={expected_dice:.6f} "
                f"regenerated={float(row['target_raw_dice']):.6f}"
            )

    if failures:
        raise RuntimeError(
            "RAW OOF provenance verification failed; do not train QC v2.\n"
            + "\n".join(failures[:10])
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build OOF QC v2 using conditional-LCC final segmentation masks"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=DEFAULT_DOMINANCE_THRESHOLD,
        help="Apply LCC only when largest raw component fraction is at least this value",
    )
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--high-quality-dice", type=float, default=0.80)
    parser.add_argument("--verification-tolerance", type=float, default=1e-4)
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0,1)")
    if not 0.0 <= args.dominance_threshold <= 1.0:
        parser.error("--dominance-threshold must be in [0,1]")
    if not 0.0 <= args.failure_dice <= 1.0 or not 0.0 <= args.high_quality_dice <= 1.0:
        parser.error("Dice thresholds must be in [0,1]")
    if args.failure_dice >= args.high_quality_dice:
        parser.error("--failure-dice must be lower than --high-quality-dice")

    experiment_dir = Path(args.experiment_dir)
    split_manifest = (
        Path(args.split_manifest)
        if args.split_manifest
        else experiment_dir / "cv_splits.json"
    )
    results_csv = (
        Path(args.results_csv)
        if args.results_csv
        else experiment_dir / "cv_results.csv"
    )
    output_dir = Path(args.output_dir)
    if not split_manifest.exists():
        raise FileNotFoundError(split_manifest)
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, args.resize_size, epochs=100)
    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, args.folds, split_manifest)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    rows = []
    feature_columns = None
    array_dir = output_dir / "arrays"
    if args.save_arrays:
        array_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("OOF QC DATASET BUILDER — CONDITIONAL LCC V2")
    print(f"Experiment:          {experiment_dir}")
    print(f"Cases:               {len(cases)}")
    print(f"Segmentation cutoff: {args.threshold:.2f}")
    print(f"LCC dominance gate:  {args.dominance_threshold:.2f}")
    print(f"Output:              {output_dir}")
    print("Targets describe the FINAL mask; raw fragmentation remains deployable QC input.")
    print("=" * 112)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        fold_cases = [by_id[case_id] for case_id in fold_spec["val_ids"]]
        loader = DataLoader(
            Dataset(fold_cases, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint, device)

        for batch in loader:
            image_t = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                if teacher is not None:
                    teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)
                else:
                    teacher_prob_t = None
                    ensemble_prob_t = student_prob_t

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = (
                teacher_prob_t[0, 0].detach().float().cpu().numpy()
                if teacher_prob_t is not None
                else None
            )
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()

            features, raw_pred, final_pred, lcc_applied = extract_conditional_lcc_features(
                image_np,
                student_prob,
                teacher_prob,
                ensemble_prob,
                threshold=args.threshold,
                dominance_threshold=args.dominance_threshold,
            )

            spacing = cv.transformed_spacing(image_t, config)
            raw_t = torch.from_numpy(raw_pred[None, None].astype(np.float32)).to(device)
            final_t = torch.from_numpy(final_pred[None, None].astype(np.float32)).to(device)
            raw_metrics = cv.case_metrics(raw_t, target_t, spacing)
            final_metrics = cv.case_metrics(final_t, target_t, spacing)

            row = {
                "fold": fold_idx,
                "case_id": case_id,
                "source": "ensemble" if teacher is not None else "student",
                "threshold": float(args.threshold),
            }
            row.update(features)
            row.update({
                "target_dice": float(final_metrics["dice"]),
                "target_precision": float(final_metrics["precision"]),
                "target_recall": float(final_metrics["recall"]),
                "target_rve": float(final_metrics["rve"]),
                "target_hd95": float(final_metrics["hd95"]),
                "target_gt_vox": float(final_metrics["gt_vox"]),
                "target_gt_fg_fraction": float(final_metrics["gt_fg"]),
                "target_failure": int(float(final_metrics["dice"]) < args.failure_dice),
                "target_high_quality": int(float(final_metrics["dice"]) >= args.high_quality_dice),
                "target_raw_dice": float(raw_metrics["dice"]),
                "target_raw_precision": float(raw_metrics["precision"]),
                "target_raw_recall": float(raw_metrics["recall"]),
                "target_raw_rve": float(raw_metrics["rve"]),
                "target_raw_hd95": float(raw_metrics["hd95"]),
            })
            rows.append(row)

            current_features = [
                key
                for key in row
                if key not in {"fold", "case_id", "source", "threshold"}
                and not key.startswith("target_")
            ]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise RuntimeError("Feature schema changed between v2 cases")

            if args.save_arrays:
                target_np = target_t[0, 0].detach().float().cpu().numpy().astype(np.uint8)
                np.savez_compressed(
                    array_dir / f"{case_id}.npz",
                    image=image_np.astype(np.float32),
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=(
                        teacher_prob.astype(np.float32)
                        if teacher_prob is not None
                        else np.asarray([], dtype=np.float32)
                    ),
                    ensemble_prob=ensemble_prob.astype(np.float32),
                    raw_prediction=raw_pred.astype(np.uint8),
                    final_prediction=final_pred.astype(np.uint8),
                    label=target_np,
                    lcc_applied=np.asarray([int(lcc_applied)], dtype=np.uint8),
                    fold=np.asarray([fold_idx], dtype=np.int16),
                )

            print(
                f"[fold {fold_idx}] {case_id} | raw={raw_metrics['dice']:.4f} | "
                f"final={final_metrics['dice']:.4f} | comps={int(features['raw_component_count'])} | "
                f"largest={float(features['raw_largest_component_fraction']):.3f} | "
                f"LCC={'Y' if lcc_applied else 'N'}"
            )

        del student
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    if sorted(row["case_id"] for row in rows) != sorted(case["id"] for case in cases):
        raise RuntimeError("OOF v2 rows do not cover every frozen case exactly once")

    verify_raw_against_frozen(rows, results_csv, args.verification_tolerance)
    print("\nFrozen RAW OOF provenance verification: PASS")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "oof_qc_dataset.csv"
    write_csv(dataset_path, rows)

    target_columns = [key for key in rows[0] if key.startswith("target_")]
    metadata = {
        "version": "oof_qc_conditional_lcc_v2",
        "experiment_dir": str(experiment_dir),
        "split_manifest": str(split_manifest),
        "frozen_results_csv": str(results_csv),
        "n_cases": len(rows),
        "n_folds": args.folds,
        "prediction_source": "student_teacher_50_50_ensemble when teacher exists",
        "prediction_threshold": float(args.threshold),
        "postprocess": {
            "name": "conditional_largest_connected_component",
            "connectivity": 6,
            "dominance_threshold": float(args.dominance_threshold),
            "rule": "apply LCC iff raw component_count > 1 and raw largest-component fraction >= dominance threshold",
        },
        "failure_dice_threshold": float(args.failure_dice),
        "high_quality_dice_threshold": float(args.high_quality_dice),
        "feature_columns": feature_columns or [],
        "target_columns": target_columns,
        "id_columns": ["fold", "case_id", "source", "threshold"],
        "feature_semantics": {
            "raw_prefix": "pre-postprocess fragmentation/volume signals",
            "unprefixed_mask_features": "final delivered conditional-LCC mask",
            "probability_and_student_teacher_features": "unchanged ensemble/student/teacher probability signals",
        },
        "leakage_rule": (
            "Only feature_columns may be used as deployable QC inputs. All target_* columns use ground truth."
        ),
        "raw_result_verification": "passed",
        "arrays_saved": bool(args.save_arrays),
        "development_warning": (
            "The 0.65 postprocess threshold was selected on these same 47 OOF development cases. "
            "QC/policy results remain development estimates until future locked validation."
        ),
    }
    metadata_path = output_dir / "oof_qc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    raw_dice = np.asarray([float(row["target_raw_dice"]) for row in rows], dtype=float)
    final_dice = np.asarray([float(row["target_dice"]) for row in rows], dtype=float)
    raw_fail = int(np.sum(raw_dice < args.failure_dice))
    final_fail = int(np.sum(final_dice < args.failure_dice))
    raw_hq = int(np.sum(raw_dice >= args.high_quality_dice))
    final_hq = int(np.sum(final_dice >= args.high_quality_dice))
    applied = int(sum(int(row["conditional_lcc_applied"]) for row in rows))

    print("\n" + "=" * 112)
    print("CONDITIONAL-LCC QC V2 DATASET COMPLETE")
    print(f"Cases:                  {len(rows)}")
    print(f"Features/case:          {len(feature_columns or [])}")
    print(f"Conditional LCC applied:{applied}/{len(rows)}")
    print(f"Mean Dice RAW -> FINAL: {np.mean(raw_dice):.4f} -> {np.mean(final_dice):.4f}")
    print(f"Failure <{args.failure_dice:.2f}:          {raw_fail} -> {final_fail}")
    print(f"High quality >={args.high_quality_dice:.2f}:    {raw_hq} -> {final_hq}")
    print(f"Dataset:                {dataset_path}")
    print(f"Metadata:               {metadata_path}")
    print("Next: run train_qc_oof.py against this v2 dataset, then recalibrate policy in the same directory.")
    print("=" * 112)


if __name__ == "__main__":
    main()
