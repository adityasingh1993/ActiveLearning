#!/usr/bin/env python3
"""Build the Round-1 leak-safe RAW-mask QC dataset.

Round-1 segmentation training uses 55 human-labeled cases (47 frozen source labels +
8 newly annotated active-learning labels), but only the original 47 source cases have
true held-out predictions in the controlled Round-1 CV experiment. The 8 new labels
were appended to every fold's training set and therefore MUST NOT be treated as OOF QC
rows.

This script replays the five Round-1 best checkpoints on the exact original held-out
47 cases, extracts the original 72 deployment-available QC features from the RAW
student/EMA-teacher 50/50 ensemble at threshold 0.50, and verifies regenerated Dice
case-by-case against experiments/round1_cv_55_translation12/cv_results.csv.

Conditional LCC is deliberately NOT applied. Cross-fitted Round-1 evaluation showed
only a marginal +0.0022 mean-Dice benefit and no change in failure/high-quality counts,
so RAW masks are the promoted Round-1 segmentation definition.

Default outputs:
  experiments/auto_label_qc_round1_raw_v1/oof_qc_dataset.csv
  experiments/auto_label_qc_round1_raw_v1/oof_qc_metadata.json
"""

import argparse
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
    extract_features,
    load_models,
    verify_against_frozen_results,
    write_csv,
)


DEFAULT_EXPERIMENT_DIR = Path("experiments/round1_cv_55_translation12")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_AUDIT_METADATA = Path(
    "experiments/round1_supervised_55_translation12/round1_label_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_qc_round1_raw_v1")
EXPECTED_SOURCE_CASES = 47
EXPECTED_NEW_LABELS = 8
EXPECTED_TOTAL_LABELS = EXPECTED_SOURCE_CASES + EXPECTED_NEW_LABELS


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Build Round-1 RAW OOF QC data from the original 47 held-out cases"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT_METADATA))
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--high-quality-dice", type=float, default=0.80)
    parser.add_argument("--verification-tolerance", type=float, default=1e-4)
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        parser.error("Round-1 QC provenance is frozen at segmentation threshold 0.50")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        parser.error("Require 0 <= failure-dice < high-quality-dice <= 1")

    experiment_dir = Path(args.experiment_dir)
    source_manifest_path = Path(args.source_manifest)
    audit_metadata_path = Path(args.audit_metadata)
    results_csv = Path(args.results_csv) if args.results_csv else experiment_dir / "cv_results.csv"
    output_dir = Path(args.output_dir)

    source_manifest = read_json(source_manifest_path)
    source_ids = [str(x) for x in source_manifest.get("all_case_ids", [])]
    if len(source_ids) != EXPECTED_SOURCE_CASES or len(set(source_ids)) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Expected frozen source manifest with {EXPECTED_SOURCE_CASES} unique IDs"
        )
    source_set = set(source_ids)

    audit = read_json(audit_metadata_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-1 audit metadata does not record a passing label audit")
    new_ids = sorted(str(x) for x in audit.get("new_human_label_ids", []))
    if len(new_ids) != EXPECTED_NEW_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_LABELS} audited Round-1 labels, found {len(new_ids)}: {new_ids}"
        )
    if source_set & set(new_ids):
        raise RuntimeError("Round-1 new-label IDs overlap the frozen source IDs")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Round-1 CV used prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=100)

    all_cases = cv.collect_cases(config)
    by_id = {str(case["id"]): case for case in all_cases}
    if len(by_id) != len(all_cases):
        raise RuntimeError("Duplicate labeled case IDs detected")
    expected_visible = source_set | set(new_ids)
    if set(by_id) != expected_visible:
        missing = sorted(expected_visible - set(by_id))
        extra = sorted(set(by_id) - expected_visible)
        raise RuntimeError(
            "Current label folder differs from audited Round-1 provenance. "
            f"Missing={missing} Extra={extra}"
        )
    if len(by_id) != EXPECTED_TOTAL_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_LABELS} current human labels, found {len(by_id)}"
        )

    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    # Validate the original fold partition without calling cv.validate_manifest(), because
    # that helper correctly expects the manifest to cover *all* currently visible labels.
    # Here the intended OOF scope is explicitly only the original 47 cases.
    folds = source_manifest.get("folds", [])
    if len(folds) != 5:
        raise RuntimeError(f"Expected 5 frozen folds, found {len(folds)}")
    held_out = [str(case_id) for fold in folds for case_id in fold.get("val_ids", [])]
    if len(held_out) != EXPECTED_SOURCE_CASES or set(held_out) != source_set:
        raise RuntimeError("Frozen fold val_ids do not cover the original 47 cases exactly once")
    if len(held_out) != len(set(held_out)):
        raise RuntimeError("A frozen source case appears in more than one held-out fold")

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

    print("=" * 108)
    print("ROUND-1 RAW OOF QC DATASET")
    print(f"Segmentation experiment: {experiment_dir}")
    print(f"Current human labels:    {len(by_id)} = 47 frozen + {len(new_ids)} new")
    print(f"Leak-safe QC OOF rows:   {EXPECTED_SOURCE_CASES} original held-out cases only")
    print(f"New AL labels excluded:  {', '.join(new_ids)}")
    print("Mask definition:         RAW 50/50 student+EMA ensemble > 0.50; no LCC")
    print(f"Round-1 CV results:      {results_csv}")
    print(f"Output:                   {output_dir}")
    print("=" * 108)

    for fold_spec in folds:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        val_ids = [str(x) for x in fold_spec.get("val_ids", [])]
        if set(val_ids) & set(new_ids):
            raise RuntimeError(f"Fold {fold_idx}: new AL label leaked into OOF validation")
        fold_cases = [by_id[case_id] for case_id in val_ids]
        loader = DataLoader(
            Dataset(fold_cases, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"Fold {fold_idx} checkpoint has no EMA teacher")

        for batch in loader:
            image_t = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = teacher_prob_t[0, 0].detach().float().cpu().numpy()
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()
            features, pred_np = extract_features(
                image_np,
                student_prob,
                teacher_prob,
                ensemble_prob,
                threshold=args.threshold,
            )

            pred_t = torch.from_numpy(pred_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            metrics = cv.case_metrics(pred_t, target_t, spacing)

            row = {
                "fold": fold_idx,
                "case_id": case_id,
                "source": "ensemble",
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
                key
                for key in row
                if key not in {"fold", "case_id", "source", "threshold"}
                and not key.startswith("target_")
            ]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise RuntimeError("Feature schema changed between Round-1 QC cases")

            if args.save_arrays:
                target_np = target_t[0, 0].detach().float().cpu().numpy().astype(np.uint8)
                np.savez_compressed(
                    array_dir / f"{case_id}.npz",
                    image=image_np.astype(np.float32),
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=teacher_prob.astype(np.float32),
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

        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    regenerated_ids = [str(row["case_id"]) for row in rows]
    if len(rows) != EXPECTED_SOURCE_CASES or set(regenerated_ids) != source_set:
        raise RuntimeError("Round-1 QC rows do not cover the original 47 cases exactly once")
    if len(regenerated_ids) != len(set(regenerated_ids)):
        raise RuntimeError("Duplicate Round-1 OOF QC row detected")

    verify_against_frozen_results(rows, results_csv, args.verification_tolerance)
    print("\nRound-1 OOF provenance verification: PASS")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "oof_qc_dataset.csv"
    write_csv(dataset_path, rows)

    target_columns = [key for key in rows[0] if key.startswith("target_")]
    metadata = {
        "version": "round1_oof_qc_raw_v1",
        "segmentation_experiment_dir": str(experiment_dir),
        "source_manifest": str(source_manifest_path),
        "audit_metadata": str(audit_metadata_path),
        "round1_cv_results": str(results_csv),
        "n_segmentation_training_labels": len(by_id),
        "n_frozen_source_labels": len(source_ids),
        "n_new_round1_training_labels": len(new_ids),
        "new_round1_training_label_ids": new_ids,
        "n_qc_oof_rows": len(rows),
        "qc_oof_scope": "original 47 source cases only; eight new AL labels were training-only in every segmentation fold",
        "n_folds": 5,
        "prediction_source": "student_teacher_50_50_ensemble",
        "prediction_threshold": float(args.threshold),
        "postprocessing": "RAW mask; no connected-component post-processing",
        "lcc_decision": (
            "Not promoted after Round-1 cross-fitted conditional-LCC evaluation: mean Dice +0.0022, "
            "no change in Dice<0.70, Dice<0.50, or Dice>=0.80 counts."
        ),
        "failure_dice_threshold": float(args.failure_dice),
        "high_quality_dice_threshold": float(args.high_quality_dice),
        "feature_columns": feature_columns or [],
        "target_columns": target_columns,
        "id_columns": ["fold", "case_id", "source", "threshold"],
        "leakage_rule": (
            "Only feature_columns may be used as deployable QC inputs. All target_* columns "
            "use ground truth and are forbidden at inference time."
        ),
        "result_verification": "passed_against_round1_cv_results",
        "arrays_saved": bool(args.save_arrays),
        "warning": (
            "QC remains a 47-row development dataset because the eight new active-learning labels "
            "were included in every Round-1 segmentation training fold and are therefore not OOF."
        ),
    }
    metadata_path = output_dir / "oof_qc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dice = np.asarray([float(row["target_dice"]) for row in rows], dtype=float)
    failures = np.asarray([int(row["target_failure"]) for row in rows], dtype=int)
    high_quality = np.asarray([int(row["target_high_quality"]) for row in rows], dtype=int)

    print("\n" + "=" * 108)
    print("ROUND-1 RAW QC DATASET COMPLETE")
    print(f"QC OOF cases:        {len(rows)}")
    print(f"Features/case:       {len(feature_columns or [])}")
    print(f"Mean OOF Dice:       {np.mean(dice):.4f}")
    print(f"Failure (<{args.failure_dice:.2f}):    {int(failures.sum())}/{len(rows)}")
    print(f"High quality (>={args.high_quality_dice:.2f}): {int(high_quality.sum())}/{len(rows)}")
    print(f"Dataset:             {dataset_path}")
    print(f"Metadata:            {metadata_path}")
    print("Next: train_qc_oof.py on this dataset, then recalibrate the development policy.")
    print("=" * 108)


if __name__ == "__main__":
    main()
