#!/usr/bin/env python3
"""Train/evaluate Round-2 QC with explicit over-segmentation risk heads.

Inputs come from build_oof_qc_dataset_round2_raw.py. Evaluation is leave-one-original-
segmentation-fold-out, preserving one QC prediction per original 47 case without fitting
that QC head on the case being evaluated.

Heads
-----
1. Dice regression: ExtraTreesRegressor
2. Failure probability: LogisticRegression for Dice < 0.70
3. Over-segmentation probability: LogisticRegression for signed RVE > +20%
4. Severe over-segmentation probability: LogisticRegression for signed RVE > +50%

All heads use only metadata-declared deployment feature columns. target_* columns are never
included in X. The full-fit bundle is created only after fold-aware OOF evaluation.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from scripts.train_qc_oof import (
    classification_metrics,
    failure_model,
    load_matrix,
    regression_metrics,
    regression_model,
    top_failure_features,
    top_regression_features,
)

DEFAULT_DIR = Path("experiments/auto_label_qc_round2_raw_v1")


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_classifier_oof(X, y, folds, seed, name, decision_threshold):
    unique_folds = sorted(np.unique(folds).tolist())
    prob = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    for fold in unique_folds:
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        if len(np.unique(y[train_idx])) < 2:
            raise RuntimeError(
                f"{name} QC train partition for fold {fold} has one class only: "
                f"positives={int(y[train_idx].sum())}/{len(train_idx)}"
            )
        model = failure_model(seed + fold)
        model.fit(X[train_idx], y[train_idx])
        prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        fold_metrics.append({
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_positive": int(y[train_idx].sum()),
            "n_test_positive": int(y[test_idx].sum()),
            "metrics": classification_metrics(y[test_idx], prob[test_idx], decision_threshold),
        })
    if not np.isfinite(prob).all():
        raise RuntimeError(f"Some cases did not receive OOF predictions for {name}")
    overall = classification_metrics(y, prob, decision_threshold)
    full = failure_model(seed)
    full.fit(X, y)
    return prob, fold_metrics, overall, full


def main():
    p = argparse.ArgumentParser(description="Round-2 fold-aware multi-head QC training/evaluation")
    p.add_argument("--dataset", default=str(DEFAULT_DIR / "oof_qc_dataset.csv"))
    p.add_argument("--metadata", default=str(DEFAULT_DIR / "oof_qc_metadata.json"))
    p.add_argument("--output-dir", default=str(DEFAULT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--failure-prob-threshold", type=float, default=0.50)
    p.add_argument("--overseg20-prob-threshold", type=float, default=0.50)
    p.add_argument("--overseg50-prob-threshold", type=float, default=0.50)
    args = p.parse_args()

    for name, value in {
        "failure-prob-threshold": args.failure_prob_threshold,
        "overseg20-prob-threshold": args.overseg20_prob_threshold,
        "overseg50-prob-threshold": args.overseg50_prob_threshold,
    }.items():
        if not 0 < value < 1:
            p.error(f"--{name} must be in (0,1)")

    dataset_path = Path(args.dataset)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    rows = read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns", []))
    target_columns = set(metadata.get("target_columns", []))
    if not rows or not feature_columns:
        raise RuntimeError("Round-2 QC dataset/feature metadata is empty")
    if any(c.startswith("target_") for c in feature_columns):
        raise RuntimeError("GT leakage detected: target_* column present in feature_columns")
    if set(feature_columns) & target_columns:
        raise RuntimeError("GT leakage detected: feature/target overlap")

    required_targets = {
        "target_dice",
        "target_failure",
        "target_high_quality",
        "target_signed_rve",
        "target_overseg_gt_20pct",
        "target_overseg_gt_50pct",
    }
    missing = required_targets - set(rows[0])
    if missing:
        raise RuntimeError(f"Round-2 QC dataset missing targets: {sorted(missing)}")

    folds = np.asarray([int(r["fold"]) for r in rows], dtype=int)
    case_ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    unique_folds = sorted(np.unique(folds).tolist())
    expected_folds = int(metadata.get("n_folds", len(unique_folds)))
    if unique_folds != list(range(expected_folds)):
        raise RuntimeError(f"Expected fold IDs 0..{expected_folds - 1}, found {unique_folds}")

    X = load_matrix(rows, feature_columns)
    y_dice = np.asarray([float(r["target_dice"]) for r in rows], dtype=float)
    y_failure = np.asarray([int(float(r["target_failure"])) for r in rows], dtype=int)
    y_hq = np.asarray([int(float(r["target_high_quality"])) for r in rows], dtype=int)
    y_signed = np.asarray([float(r["target_signed_rve"]) for r in rows], dtype=float)
    y_over20 = np.asarray([int(float(r["target_overseg_gt_20pct"])) for r in rows], dtype=int)
    y_over50 = np.asarray([int(float(r["target_overseg_gt_50pct"])) for r in rows], dtype=int)

    for name, y in {"failure": y_failure, "overseg20": y_over20, "overseg50": y_over50}.items():
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"Need both classes for {name}; positives={int(y.sum())}/{len(y)}")

    dice_pred = np.full(len(rows), np.nan, dtype=float)
    reg_fold_metrics = []
    print("=" * 108)
    print("ROUND-2 FOLD-AWARE QC EVALUATION")
    print(f"Cases: {len(rows)} | deployment features: {len(feature_columns)} | folds: {unique_folds}")
    print(f"Targets: failures={int(y_failure.sum())}, overseg>20={int(y_over20.sum())}, overseg>50={int(y_over50.sum())}")
    print("=" * 108)

    for fold in unique_folds:
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        reg = regression_model(args.seed + fold)
        reg.fit(X[train_idx], y_dice[train_idx])
        dice_pred[test_idx] = np.clip(reg.predict(X[test_idx]), 0, 1)
        reg_m = regression_metrics(y_dice[test_idx], dice_pred[test_idx])
        reg_fold_metrics.append({
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "metrics": reg_m,
        })

    if not np.isfinite(dice_pred).all():
        raise RuntimeError("Some cases did not receive OOF Dice predictions")
    overall_reg = regression_metrics(y_dice, dice_pred)
    final_reg = regression_model(args.seed)
    final_reg.fit(X, y_dice)

    failure_prob, failure_folds, failure_metrics, final_failure = fit_classifier_oof(
        X, y_failure, folds, args.seed + 1000, "failure", args.failure_prob_threshold
    )
    over20_prob, over20_folds, over20_metrics, final_over20 = fit_classifier_oof(
        X, y_over20, folds, args.seed + 2000, "overseg20", args.overseg20_prob_threshold
    )
    over50_prob, over50_folds, over50_metrics, final_over50 = fit_classifier_oof(
        X, y_over50, folds, args.seed + 3000, "overseg50", args.overseg50_prob_threshold
    )

    prediction_rows = []
    for i in range(len(rows)):
        prediction_rows.append({
            "fold": int(folds[i]),
            "case_id": str(case_ids[i]),
            "actual_dice": float(y_dice[i]),
            "predicted_dice_oof": float(dice_pred[i]),
            "actual_failure": int(y_failure[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "predicted_failure_oof": int(failure_prob[i] >= args.failure_prob_threshold),
            "actual_high_quality": int(y_hq[i]),
            "actual_signed_rve": float(y_signed[i]),
            "actual_overseg_gt_20pct": int(y_over20[i]),
            "predicted_overseg20_probability_oof": float(over20_prob[i]),
            "predicted_overseg20_oof": int(over20_prob[i] >= args.overseg20_prob_threshold),
            "actual_overseg_gt_50pct": int(y_over50[i]),
            "predicted_overseg50_probability_oof": float(over50_prob[i]),
            "predicted_overseg50_oof": int(over50_prob[i] >= args.overseg50_prob_threshold),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "qc_oof_predictions.csv"
    write_csv(prediction_path, prediction_rows)

    metrics = {
        "version": "qc_tabular_round2_overseg_v1",
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "n_cases": int(len(rows)),
        "n_features": int(len(feature_columns)),
        "folds": unique_folds,
        "validation": "leave-one-original-segmentation-fold-out",
        "failure_dice_threshold": float(metadata.get("failure_dice_threshold", 0.70)),
        "high_quality_dice_threshold": float(metadata.get("high_quality_dice_threshold", 0.80)),
        "oversegmentation_threshold_pct": float(metadata.get("oversegmentation_threshold_pct", 20.0)),
        "severe_oversegmentation_threshold_pct": float(metadata.get("severe_oversegmentation_threshold_pct", 50.0)),
        "class_prevalence": {
            "failure": float(np.mean(y_failure)),
            "overseg_gt_20pct": float(np.mean(y_over20)),
            "overseg_gt_50pct": float(np.mean(y_over50)),
        },
        "oof_regression": overall_reg,
        "oof_failure": failure_metrics,
        "oof_overseg20": over20_metrics,
        "oof_overseg50": over50_metrics,
        "per_fold": {
            "regression": reg_fold_metrics,
            "failure": failure_folds,
            "overseg20": over20_folds,
            "overseg50": over50_folds,
        },
        "full_fit_top_regression_features": top_regression_features(final_reg, feature_columns),
        "full_fit_top_failure_features": top_failure_features(final_failure, feature_columns),
        "full_fit_top_overseg20_features": top_failure_features(final_over20, feature_columns),
        "full_fit_top_overseg50_features": top_failure_features(final_over50, feature_columns),
        "warning": "Development OOF QC metrics on 47 original cases only. External31 remains excluded from fitting and policy calibration.",
    }
    metrics_path = output_dir / "qc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "version": "qc_tabular_round2_overseg_v1",
        "feature_columns": feature_columns,
        "regression_model": final_reg,
        "failure_model": final_failure,
        "overseg20_model": final_over20,
        "overseg50_model": final_over50,
        "failure_probability_decision_threshold": float(args.failure_prob_threshold),
        "overseg20_probability_decision_threshold": float(args.overseg20_prob_threshold),
        "overseg50_probability_decision_threshold": float(args.overseg50_prob_threshold),
        "failure_dice_threshold": float(metadata.get("failure_dice_threshold", 0.70)),
        "high_quality_dice_threshold": float(metadata.get("high_quality_dice_threshold", 0.80)),
        "oversegmentation_threshold_pct": float(metadata.get("oversegmentation_threshold_pct", 20.0)),
        "severe_oversegmentation_threshold_pct": float(metadata.get("severe_oversegmentation_threshold_pct", 50.0)),
        "training_cases": [str(x) for x in case_ids.tolist()],
        "note": "Full-fit development bundle created only after original-fold OOF evaluation; external31 excluded.",
    }
    bundle_path = output_dir / "qc_model_bundle.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 108)
    print("ROUND-2 QC OOF RESULTS")
    print(
        f"Dice regression: MAE={overall_reg['mae']:.4f} | RMSE={overall_reg['rmse']:.4f} | "
        f"R2={overall_reg['r2']:.4f} | Spearman={overall_reg['spearman']:.4f}"
    )
    for label, result in [
        ("Failure Dice<.70", failure_metrics),
        ("Overseg >+20%", over20_metrics),
        ("Severe overseg >+50%", over50_metrics),
    ]:
        print(
            f"{label:<22} AUROC={result['auroc']:.4f} | AUPRC={result['auprc']:.4f} | "
            f"Brier={result['brier']:.4f} | Recall={result['recall']:.4f} | Precision={result['precision']:.4f}"
        )
    print(f"OOF predictions: {prediction_path}")
    print(f"Metrics:         {metrics_path}")
    print(f"Full-fit bundle: {bundle_path}")
    print("Next: recalibrate a conservative four-signal AUTO_ACCEPT / REVIEW / ACTIVE policy.")
    print("=" * 108)


if __name__ == "__main__":
    main()
