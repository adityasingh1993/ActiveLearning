#!/usr/bin/env python3
"""Build the Round-2 leak-safe RAW-mask QC dataset.

Only the original 47 source cases are valid OOF QC rows. Round-1 and Round-2 human labels
were appended to every segmentation fold's training set, so they are deliberately excluded
from QC target rows.

The builder replays the completed Round-2 CV best checkpoints on the exact original held-out
47 cases, extracts the same deployment-only QC features used by earlier rounds, and adds
explicit signed-volume failure targets:

- target_failure: Dice < 0.70
- target_high_quality: Dice >= 0.80
- target_signed_rve: signed relative volume error (%)
- target_overseg_gt_20pct
- target_overseg_gt_50pct
- target_underseg_lt_minus20pct
- target_tp_vox / target_fp_vox / target_fn_vox

Positive signed RVE means over-segmentation. The segmentation definition is frozen as the
RAW 50/50 student + EMA teacher ensemble at threshold 0.50; no LCC is applied.
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

DEFAULT_EXPERIMENT_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_AUDIT_METADATA = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_qc_round2_raw_v1")
EXPECTED_SOURCE_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def binary_counts(pred_np, target_np):
    pred = np.asarray(pred_np) > 0
    gt = np.asarray(target_np) > 0
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: pred={pred.shape} gt={gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, fn


def main():
    p = argparse.ArgumentParser(
        description="Build Round-2 RAW OOF QC data from the exact original 47 held-out cases"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT_METADATA))
    p.add_argument("--results-csv", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    p.add_argument("--overseg-threshold", type=float, default=20.0)
    p.add_argument("--severe-overseg-threshold", type=float, default=50.0)
    p.add_argument("--underseg-threshold", type=float, default=-20.0)
    p.add_argument("--verification-tolerance", type=float, default=1e-4)
    p.add_argument("--save-arrays", action="store_true")
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Round-2 QC segmentation provenance is frozen at threshold 0.50")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        p.error("Require 0 <= failure-dice < high-quality-dice <= 1")
    if args.severe_overseg_threshold <= args.overseg_threshold:
        p.error("--severe-overseg-threshold must be greater than --overseg-threshold")
    if args.underseg_threshold >= 0:
        p.error("--underseg-threshold must be negative")

    experiment_dir = Path(args.experiment_dir)
    source_manifest_path = Path(args.source_manifest)
    audit_metadata_path = Path(args.audit_metadata)
    results_csv = Path(args.results_csv) if args.results_csv else experiment_dir / "cv_results.csv"
    output_dir = Path(args.output_dir)

    source_manifest = read_json(source_manifest_path)
    source_ids = [str(x) for x in source_manifest.get("all_case_ids", [])]
    source_set = set(source_ids)
    if len(source_ids) != EXPECTED_SOURCE_CASES or len(source_set) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Expected frozen source manifest with {EXPECTED_SOURCE_CASES} unique IDs")

    audit = read_json(audit_metadata_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata does not record a passing label audit")
    if not audit.get("selection_provenance_enforced", False):
        raise RuntimeError("Round-2 audit did not enforce acquisition-selection provenance")

    round1_ids = sorted(str(x) for x in audit.get("round1_human_label_ids", []))
    round2_ids = sorted(str(x) for x in audit.get("round2_new_human_label_ids", []))
    if set(round1_ids) & source_set or set(round2_ids) & source_set:
        raise RuntimeError("Round-1/Round-2 added human labels overlap frozen source IDs")
    if set(round1_ids) & set(round2_ids):
        raise RuntimeError("Round-1 and Round-2 human-label ID sets overlap")

    expected_current = source_set | set(round1_ids) | set(round2_ids)
    reported_current = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    if reported_current and reported_current != expected_current:
        raise RuntimeError("Round-2 audit all_current_human_label_ids is internally inconsistent")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Controlled Round-2 CV used prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=100)

    all_cases = cv.collect_cases(config)
    by_id = {str(case["id"]): case for case in all_cases}
    if len(by_id) != len(all_cases):
        raise RuntimeError("Duplicate labeled case IDs detected")
    current_set = set(by_id)
    if current_set != expected_current:
        raise RuntimeError(
            "Current label folder differs from the passing Round-2 audit. "
            f"Missing={sorted(expected_current-current_set)} Extra={sorted(current_set-expected_current)}"
        )

    folds = source_manifest.get("folds", [])
    if len(folds) != 5:
        raise RuntimeError(f"Expected 5 frozen folds, found {len(folds)}")
    held_out = [str(cid) for fold in folds for cid in fold.get("val_ids", [])]
    if len(held_out) != EXPECTED_SOURCE_CASES or set(held_out) != source_set:
        raise RuntimeError("Frozen fold val_ids do not cover the original 47 cases exactly once")
    if len(held_out) != len(set(held_out)):
        raise RuntimeError("A frozen source case appears in more than one held-out fold")
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
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
    print("ROUND-2 RAW OOF QC DATASET")
    print(f"Segmentation experiment:   {experiment_dir}")
    print(f"Current human labels:      {len(by_id)} = 47 source + {len(round1_ids)} Round-1 + {len(round2_ids)} Round-2")
    print(f"Leak-safe QC OOF rows:     {EXPECTED_SOURCE_CASES} original held-out cases only")
    print(f"Excluded added labels:     {len(round1_ids) + len(round2_ids)} training-only human cases")
    print("Mask definition:           RAW 50/50 student+EMA ensemble > 0.50; no LCC")
    print(f"Round-2 CV results:        {results_csv}")
    print(f"Output:                    {output_dir}")
    print("=" * 112)

    for fold_spec in folds:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        val_ids = [str(x) for x in fold_spec.get("val_ids", [])]
        if (set(val_ids) & set(round1_ids)) or (set(val_ids) & set(round2_ids)):
            raise RuntimeError(f"Fold {fold_idx}: added human label leaked into OOF validation")

        fold_cases = [by_id[cid] for cid in val_ids]
        loader = DataLoader(Dataset(fold_cases, transform=transform), batch_size=1, shuffle=False, num_workers=0)
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
                image_np, student_prob, teacher_prob, ensemble_prob, threshold=args.threshold
            )

            pred_t = torch.from_numpy(pred_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            metrics = cv.case_metrics(pred_t, target_t, spacing)
            target_np = target_t[0, 0].detach().float().cpu().numpy() > 0.5
            tp, fp, fn = binary_counts(pred_np, target_np)
            pred_vox = int(np.asarray(pred_np).sum())
            gt_vox = int(target_np.sum())
            signed_rve = 100.0 * (pred_vox - gt_vox) / (gt_vox + 1e-8)

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
                "target_signed_rve": float(signed_rve),
                "target_hd95": float(metrics["hd95"]),
                "target_pred_vox": int(pred_vox),
                "target_gt_vox": int(gt_vox),
                "target_tp_vox": int(tp),
                "target_fp_vox": int(fp),
                "target_fn_vox": int(fn),
                "target_gt_fg_fraction": float(metrics["gt_fg"]),
                "target_failure": int(float(metrics["dice"]) < args.failure_dice),
                "target_high_quality": int(float(metrics["dice"]) >= args.high_quality_dice),
                "target_overseg_gt_20pct": int(signed_rve > args.overseg_threshold),
                "target_overseg_gt_50pct": int(signed_rve > args.severe_overseg_threshold),
                "target_underseg_lt_minus20pct": int(signed_rve < args.underseg_threshold),
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
                raise RuntimeError("Feature schema changed between Round-2 QC cases")

            if args.save_arrays:
                np.savez_compressed(
                    array_dir / f"{case_id}.npz",
                    image=image_np.astype(np.float32),
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=teacher_prob.astype(np.float32),
                    ensemble_prob=ensemble_prob.astype(np.float32),
                    prediction=np.asarray(pred_np).astype(np.uint8),
                    label=target_np.astype(np.uint8),
                    fold=np.asarray([fold_idx], dtype=np.int16),
                )

            print(
                f"[fold {fold_idx}] {case_id} | Dice={metrics['dice']:.4f} | "
                f"signedRVE={signed_rve:+.1f}% | FP={fp} | FN={fn} | "
                f"components={features['component_count']}"
            )

        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    ids = [str(r["case_id"]) for r in rows]
    if len(rows) != EXPECTED_SOURCE_CASES or set(ids) != source_set or len(ids) != len(set(ids)):
        raise RuntimeError("Round-2 QC rows do not cover the original 47 cases exactly once")

    verify_against_frozen_results(rows, results_csv, args.verification_tolerance)
    print("\nRound-2 OOF provenance verification: PASS")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "oof_qc_dataset.csv"
    write_csv(dataset_path, rows)

    target_columns = [key for key in rows[0] if key.startswith("target_")]
    metadata = {
        "version": "round2_oof_qc_raw_overseg_v1",
        "segmentation_experiment_dir": str(experiment_dir),
        "source_manifest": str(source_manifest_path),
        "audit_metadata": str(audit_metadata_path),
        "round2_cv_results": str(results_csv),
        "n_segmentation_training_labels": len(by_id),
        "n_frozen_source_labels": len(source_ids),
        "n_round1_training_labels": len(round1_ids),
        "n_round2_training_labels": len(round2_ids),
        "round1_training_label_ids": round1_ids,
        "round2_training_label_ids": round2_ids,
        "n_qc_oof_rows": len(rows),
        "qc_oof_scope": "original 47 source cases only; all added AL labels were training-only in every segmentation fold",
        "n_folds": 5,
        "prediction_source": "student_teacher_50_50_ensemble",
        "prediction_threshold": float(args.threshold),
        "postprocessing": "RAW mask; no connected-component post-processing",
        "failure_dice_threshold": float(args.failure_dice),
        "high_quality_dice_threshold": float(args.high_quality_dice),
        "oversegmentation_threshold_pct": float(args.overseg_threshold),
        "severe_oversegmentation_threshold_pct": float(args.severe_overseg_threshold),
        "undersegmentation_threshold_pct": float(args.underseg_threshold),
        "feature_columns": feature_columns or [],
        "target_columns": target_columns,
        "id_columns": ["fold", "case_id", "source", "threshold"],
        "leakage_rule": "Only feature_columns are deployable QC inputs; every target_* column uses GT and is forbidden at inference time.",
        "result_verification": "passed_against_round2_cv_results",
        "arrays_saved": bool(args.save_arrays),
        "external31_role": "excluded_from_QC_training_and_policy_calibration",
    }
    metadata_path = output_dir / "oof_qc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dice = np.asarray([float(r["target_dice"]) for r in rows])
    signed = np.asarray([float(r["target_signed_rve"]) for r in rows])
    print("\nROUND-2 OOF QC TARGET SUMMARY")
    print(f"Cases:                  {len(rows)}")
    print(f"Features:               {len(feature_columns or [])}")
    print(f"Mean Dice:              {float(np.mean(dice)):.4f}")
    print(f"Dice <0.70:             {int(np.sum(dice < args.failure_dice))}")
    print(f"Dice >=0.80:            {int(np.sum(dice >= args.high_quality_dice))}")
    print(f"Median signed RVE:      {float(np.median(signed)):+.2f}%")
    print(f"Overseg >+20%:          {int(np.sum(signed > args.overseg_threshold))}")
    print(f"Severe overseg >+50%:   {int(np.sum(signed > args.severe_overseg_threshold))}")
    print(f"Underseg <-20%:         {int(np.sum(signed < args.underseg_threshold))}")
    print(f"Dataset:                {dataset_path}")
    print(f"Metadata:               {metadata_path}")


if __name__ == "__main__":
    main()
