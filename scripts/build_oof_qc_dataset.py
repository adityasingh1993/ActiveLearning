#!/usr/bin/env python3
"""Build a leak-safe QC dataset from frozen out-of-fold segmentation predictions.

This is the first stage of the auto-label / active-learning pipeline. It replays the
frozen 5-fold held-out inference and produces one row per OOF case containing:

Deployment-available QC features only
-------------------------------------
- ensemble probability/confidence/entropy statistics
- student/teacher disagreement
- predicted foreground fraction and volume
- connected-component statistics
- predicted centroid, bounding box, occupancy, and boundary confidence
- whole-image and predicted-region intensity statistics

Ground-truth-derived training targets only
------------------------------------------
- Dice, precision, recall, RVE, HD95, GT voxels/fraction
- configurable failure/high-quality labels derived from Dice

No GT-dependent statistic is allowed into the feature list. Metadata explicitly records
feature_columns and target_columns so downstream QC training can enforce this boundary.

By default, regenerated OOF Dice is checked against the frozen cv_results.csv. A mismatch
fails fast because it usually means the wrong checkpoint, split manifest, threshold, or
experiment directory is being used.

Expected baseline layout:
  experiments/cv5_supervised_47_translation12/
    cv_splits.json
    cv_results.csv
    checkpoints/fold_0/best_checkpoint.pth
    ...

Default outputs:
  experiments/auto_label_qc_v1/oof_qc_dataset.csv
  experiments/auto_label_qc_v1/oof_qc_metadata.json

This script does not train a QC model and does not alter segmentation checkpoints.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError("build_oof_qc_dataset.py requires scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_qc_v1")


def _finite_stats(values, prefix):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
    }


def _binary_entropy(prob):
    p = np.clip(prob.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy / math.log(2.0)


def _component_features(pred):
    pred = pred.astype(bool, copy=False)
    pred_vox = int(pred.sum())
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(pred, structure=structure)

    if count == 0:
        return {
            "component_count": 0,
            "components_ge_100": 0,
            "components_ge_1000": 0,
            "largest_component_vox": 0,
            "largest_component_fraction": 0.0,
            "second_component_fraction": 0.0,
        }

    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    ordered = np.sort(sizes)[::-1]
    largest = int(ordered[0])
    second = int(ordered[1]) if ordered.size > 1 else 0
    denom = max(pred_vox, 1)
    return {
        "component_count": int(count),
        "components_ge_100": int(np.sum(sizes >= 100)),
        "components_ge_1000": int(np.sum(sizes >= 1000)),
        "largest_component_vox": largest,
        "largest_component_fraction": float(largest / denom),
        "second_component_fraction": float(second / denom),
    }


def _geometry_features(pred):
    pred = pred.astype(bool, copy=False)
    shape = np.asarray(pred.shape, dtype=np.float32)
    coords = np.argwhere(pred)
    if coords.size == 0:
        return {
            "centroid_d_norm": float("nan"),
            "centroid_h_norm": float("nan"),
            "centroid_w_norm": float("nan"),
            "bbox_d_fraction": 0.0,
            "bbox_h_fraction": 0.0,
            "bbox_w_fraction": 0.0,
            "bbox_volume_fraction": 0.0,
            "bbox_occupancy": 0.0,
        }

    centroid = coords.mean(axis=0)
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    bbox_size = maximum - minimum + 1
    bbox_vox = int(np.prod(bbox_size))
    pred_vox = int(pred.sum())

    denom = np.maximum(shape - 1.0, 1.0)
    return {
        "centroid_d_norm": float(centroid[0] / denom[0]),
        "centroid_h_norm": float(centroid[1] / denom[1]),
        "centroid_w_norm": float(centroid[2] / denom[2]),
        "bbox_d_fraction": float(bbox_size[0] / shape[0]),
        "bbox_h_fraction": float(bbox_size[1] / shape[1]),
        "bbox_w_fraction": float(bbox_size[2] / shape[2]),
        "bbox_volume_fraction": float(bbox_vox / pred.size),
        "bbox_occupancy": float(pred_vox / max(bbox_vox, 1)),
    }


def _boundary_features(pred, prob):
    pred = pred.astype(bool, copy=False)
    if not pred.any():
        return {
            "boundary_vox_fraction": 0.0,
            "boundary_confidence_mean": float("nan"),
            "boundary_confidence_p10": float("nan"),
            "boundary_entropy_mean": float("nan"),
        }

    eroded = ndimage.binary_erosion(pred, structure=ndimage.generate_binary_structure(3, 1))
    boundary = np.logical_and(pred, np.logical_not(eroded))
    values = prob[boundary]
    confidence = 2.0 * np.abs(values - 0.5)
    entropy = _binary_entropy(values)
    return {
        "boundary_vox_fraction": float(boundary.sum() / pred.size),
        "boundary_confidence_mean": float(np.mean(confidence)) if values.size else float("nan"),
        "boundary_confidence_p10": float(np.percentile(confidence, 10)) if values.size else float("nan"),
        "boundary_entropy_mean": float(np.mean(entropy)) if values.size else float("nan"),
    }


def _intensity_features(image, pred):
    image = image.astype(np.float32, copy=False)
    pred = pred.astype(bool, copy=False)
    result = {}
    result.update(_finite_stats(image, "image"))

    fg = image[pred]
    bg = image[~pred]
    fg_mean = float(np.mean(fg)) if fg.size else float("nan")
    bg_mean = float(np.mean(bg)) if bg.size else float("nan")
    result.update(_finite_stats(fg, "pred_region_image"))
    result.update(_finite_stats(bg, "pred_background_image"))
    result["pred_region_minus_background_mean"] = (
        fg_mean - bg_mean if math.isfinite(fg_mean) and math.isfinite(bg_mean) else float("nan")
    )
    return result


def extract_features(image, student_prob, teacher_prob, ensemble_prob, threshold):
    """Extract only signals available when GT is absent."""
    pred = ensemble_prob > float(threshold)
    pred_vox = int(pred.sum())
    pred_fg = float(pred_vox / pred.size)

    confidence = np.maximum(ensemble_prob, 1.0 - ensemble_prob)
    entropy = _binary_entropy(ensemble_prob)
    margin = np.abs(ensemble_prob - 0.5)

    features = {
        "pred_vox": pred_vox,
        "pred_fg_fraction": pred_fg,
        "confidence_mean": float(np.mean(confidence)),
        "confidence_p05": float(np.percentile(confidence, 5)),
        "confidence_p10": float(np.percentile(confidence, 10)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p90": float(np.percentile(entropy, 90)),
        "entropy_p95": float(np.percentile(entropy, 95)),
        "uncertain_fraction_margin_0p05": float(np.mean(margin < 0.05)),
        "uncertain_fraction_margin_0p10": float(np.mean(margin < 0.10)),
        "uncertain_fraction_margin_0p20": float(np.mean(margin < 0.20)),
    }
    features.update(_finite_stats(ensemble_prob, "ensemble_prob"))

    if pred_vox > 0:
        features.update(_finite_stats(ensemble_prob[pred], "pred_region_prob"))
    else:
        features.update(_finite_stats([], "pred_region_prob"))

    if teacher_prob is not None:
        disagreement = np.abs(student_prob - teacher_prob)
        s_pred = student_prob > float(threshold)
        t_pred = teacher_prob > float(threshold)
        features.update({
            "student_teacher_mae": float(np.mean(disagreement)),
            "student_teacher_p95_absdiff": float(np.percentile(disagreement, 95)),
            "student_teacher_p99_absdiff": float(np.percentile(disagreement, 99)),
            "student_teacher_hard_disagreement_fraction": float(np.mean(s_pred != t_pred)),
            "student_pred_fg_fraction": float(np.mean(s_pred)),
            "teacher_pred_fg_fraction": float(np.mean(t_pred)),
            "student_teacher_fg_fraction_absdiff": float(abs(np.mean(s_pred) - np.mean(t_pred))),
        })
    else:
        features.update({
            "student_teacher_mae": float("nan"),
            "student_teacher_p95_absdiff": float("nan"),
            "student_teacher_p99_absdiff": float("nan"),
            "student_teacher_hard_disagreement_fraction": float("nan"),
            "student_pred_fg_fraction": float(np.mean(student_prob > float(threshold))),
            "teacher_pred_fg_fraction": float("nan"),
            "student_teacher_fg_fraction_absdiff": float("nan"),
        })

    features.update(_component_features(pred))
    features.update(_geometry_features(pred))
    features.update(_boundary_features(pred, ensemble_prob))
    features.update(_intensity_features(image, pred))
    return features, pred


def load_models(config, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "net_A" not in state:
        raise RuntimeError(f"Invalid HASSL checkpoint: {checkpoint}")

    student = cv.build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = cv.build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()
    return student, teacher


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_against_frozen_results(rows, result_path, tolerance):
    frozen_rows = read_csv(result_path)
    frozen = {row["case_id"]: row for row in frozen_rows}
    if len(frozen) != len(rows):
        raise RuntimeError(
            f"Frozen results contain {len(frozen)} unique cases, regenerated QC dataset contains {len(rows)}."
        )

    failures = []
    for row in rows:
        case_id = row["case_id"]
        if case_id not in frozen:
            failures.append(f"{case_id}: missing from frozen results")
            continue
        expected = frozen[case_id]
        if expected.get("source") != "ensemble":
            failures.append(f"{case_id}: frozen source={expected.get('source')} (expected ensemble)")
            continue
        try:
            expected_threshold = float(expected["threshold"])
            expected_dice = float(expected["dice"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{case_id}: malformed frozen result row")
            continue
        if abs(expected_threshold - float(row["threshold"])) > 1e-8:
            failures.append(
                f"{case_id}: threshold mismatch frozen={expected_threshold} regenerated={row['threshold']}"
            )
        if abs(expected_dice - float(row["target_dice"])) > tolerance:
            failures.append(
                f"{case_id}: Dice mismatch frozen={expected_dice:.6f} "
                f"regenerated={float(row['target_dice']):.6f}"
            )

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            "OOF provenance verification failed. Do not build QC training data from mismatched predictions.\n"
            + preview
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build leak-safe OOF QC features/targets from frozen 47-case CV checkpoints"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--high-quality-dice", type=float, default=0.80)
    parser.add_argument("--verification-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--skip-result-verification",
        action="store_true",
        help="Skip frozen cv_results.csv comparison. Not recommended for the baseline QC dataset.",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Also save compressed per-case image/probability/prediction/label NPZ files.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >=2")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be in (0,1)")
    if not 0 <= args.failure_dice <= 1 or not 0 <= args.high_quality_dice <= 1:
        parser.error("Dice label thresholds must be in [0,1]")
    if args.failure_dice >= args.high_quality_dice:
        parser.error("--failure-dice must be lower than --high-quality-dice")

    experiment_dir = Path(args.experiment_dir)
    split_manifest = Path(args.split_manifest) if args.split_manifest else experiment_dir / "cv_splits.json"
    results_csv = Path(args.results_csv) if args.results_csv else experiment_dir / "cv_results.csv"
    output_dir = Path(args.output_dir)

    if not split_manifest.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest}")
    if not args.skip_result_verification and not results_csv.exists():
        raise FileNotFoundError(f"Missing frozen CV results: {results_csv}")

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

    print("=" * 100)
    print("OOF QC DATASET BUILDER")
    print(f"Experiment:      {experiment_dir}")
    print(f"Split manifest:  {split_manifest}")
    print(f"Frozen results:  {results_csv}")
    print(f"Threshold:       {args.threshold:.2f}")
    print(f"Output:          {output_dir}")
    print("QC features are deployment-only; GT information is written only as target_* columns.")
    print("=" * 100)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")

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
            features, pred_np = extract_features(
                image_np, student_prob, teacher_prob, ensemble_prob, args.threshold
            )

            pred_t = torch.from_numpy(pred_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            metrics = cv.case_metrics(pred_t, target_t, spacing)

            row = {
                "fold": fold_idx,
                "case_id": case_id,
                "source": "ensemble" if teacher is not None else "student",
                "threshold": float(args.threshold),
            }
            row.update(features)
            row.update({
                "target_dice": float(metrics["dice"]),
                "target_precision": float(metrics["precision"]),
                "target_recall": float(metrics["recall"]),
                "target_rve": float(metrics["rve"]),
                "target_hd95": float(metrics["hd95"]),
                "target_gt_vox": float(metrics["gt_vox"]),
                "target_gt_fg_fraction": float(metrics["gt_fg"]),
                "target_failure": int(float(metrics["dice"]) < args.failure_dice),
                "target_high_quality": int(float(metrics["dice"]) >= args.high_quality_dice),
            })
            rows.append(row)

            current_features = [
                key for key in row
                if key not in {"fold", "case_id", "source", "threshold"}
                and not key.startswith("target_")
            ]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise RuntimeError("Feature schema changed between cases; refusing inconsistent QC dataset")

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
                    prediction=pred_np.astype(np.uint8),
                    label=target_np,
                    fold=np.asarray([fold_idx], dtype=np.int16),
                )

            print(
                f"[fold {fold_idx}] {case_id} | Dice={metrics['dice']:.4f} | "
                f"pred_fg={features['pred_fg_fraction']:.5f} | "
                f"components={features['component_count']} | "
                f"S/T MAE={features['student_teacher_mae']:.5f}"
            )

        del student
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    expected_ids = sorted(case["id"] for case in cases)
    regenerated_ids = sorted(row["case_id"] for row in rows)
    if regenerated_ids != expected_ids:
        raise RuntimeError("OOF QC rows do not cover every labeled case exactly once")

    if not args.skip_result_verification:
        verify_against_frozen_results(rows, results_csv, args.verification_tolerance)
        print("\nFrozen OOF verification: PASS")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "oof_qc_dataset.csv"
    write_csv(dataset_path, rows)

    target_columns = [key for key in rows[0].keys() if key.startswith("target_")]
    metadata = {
        "version": "oof_qc_v1",
        "experiment_dir": str(experiment_dir),
        "split_manifest": str(split_manifest),
        "frozen_results_csv": str(results_csv),
        "n_cases": len(rows),
        "n_folds": args.folds,
        "prediction_source": "ensemble when teacher exists, otherwise student",
        "prediction_threshold": args.threshold,
        "failure_dice_threshold": args.failure_dice,
        "high_quality_dice_threshold": args.high_quality_dice,
        "feature_columns": feature_columns or [],
        "target_columns": target_columns,
        "id_columns": ["fold", "case_id", "source", "threshold"],
        "leakage_rule": (
            "Only feature_columns may be used as deployable QC inputs. All target_* columns "
            "use ground truth and are forbidden at inference time."
        ),
        "result_verification": "skipped" if args.skip_result_verification else "passed",
        "arrays_saved": bool(args.save_arrays),
    }
    metadata_path = output_dir / "oof_qc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dice = np.asarray([row["target_dice"] for row in rows], dtype=float)
    failures = np.asarray([row["target_failure"] for row in rows], dtype=int)
    high_quality = np.asarray([row["target_high_quality"] for row in rows], dtype=int)

    print("\n" + "=" * 100)
    print("QC DATASET COMPLETE")
    print(f"Cases:              {len(rows)}")
    print(f"Features/case:      {len(feature_columns or [])}")
    print(f"OOF Dice:           {np.mean(dice):.4f} +/- {np.std(dice):.4f}")
    print(f"Failure (<{args.failure_dice:.2f}):      {int(failures.sum())}/{len(rows)}")
    print(f"High quality (>={args.high_quality_dice:.2f}): {int(high_quality.sum())}/{len(rows)}")
    print(f"Dataset:            {dataset_path}")
    print(f"Metadata:           {metadata_path}")
    if args.save_arrays:
        print(f"Arrays:             {array_dir}")
    print("Next: train/evaluate QC using patient/fold-aware validation; never fit and score on the same OOF rows.")
    print("=" * 100)


if __name__ == "__main__":
    main()
