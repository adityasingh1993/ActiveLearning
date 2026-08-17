#!/usr/bin/env python3
"""Train and evaluate a leak-safe tabular QC model on OOF segmentation features.

The input must come from scripts/build_oof_qc_dataset.py. Only the feature_columns listed
in oof_qc_metadata.json are used as model inputs. All target_* columns are treated as
training/evaluation targets and are never included in X.

Evaluation is leave-one-segmentation-fold-out: for each original segmentation CV fold,
QC is trained on the other four OOF folds and evaluated on the held-out fold. This gives
one QC prediction per labeled case without fitting the QC model on that case.

Two deliberately simple models are used for the first QC baseline:
- Dice regression: strongly regularized ExtraTreesRegressor
- Failure probability (Dice < configured failure threshold): class-balanced logistic regression

The goal is not to maximize development-set performance with aggressive model search. With
47 cases and 72 features, a small/regularized baseline is safer and easier to interpret.

Outputs:
  experiments/auto_label_qc_v1/qc_oof_predictions.csv
  experiments/auto_label_qc_v1/qc_metrics.json
  experiments/auto_label_qc_v1/qc_model_bundle.pkl

The final bundle is fit on all 47 development rows only after fold-aware OOF evaluation.
Its metrics remain development estimates; policy thresholds still need a future locked set.
"""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError("train_qc_oof.py requires scipy and scikit-learn") from exc

DEFAULT_DIR = Path("experiments/auto_label_qc_v1")


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_matrix(rows, feature_columns):
    return np.asarray([[to_float(row.get(c)) for c in feature_columns] for row in rows], dtype=float)


def regression_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(n_estimators=500, max_depth=3, min_samples_leaf=3,
                                      max_features=0.5, random_state=seed, n_jobs=-1)),
    ])


def failure_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000,
                                     solver="liblinear", random_state=seed)),
    ])


def safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    try:
        result = fn(y_true, y_pred)
        value = result.statistic if hasattr(result, "statistic") else result[0]
        return float(value)
    except Exception:
        return float("nan")


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "pearson": safe_corr(pearsonr, y_true, y_pred),
        "spearman": safe_corr(spearmanr, y_true, y_pred),
    }


def classification_metrics(y_true, prob, decision_threshold=0.5):
    pred = (prob >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, prob))
        result["auprc"] = float(average_precision_score(y_true, prob))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


def top_regression_features(model, feature_columns, limit=15):
    values = np.asarray(model.named_steps["model"].feature_importances_, dtype=float)
    order = np.argsort(-values)[:limit]
    return [{"feature": feature_columns[int(i)], "importance": float(values[int(i)])} for i in order]


def top_failure_features(model, feature_columns, limit=15):
    values = np.asarray(model.named_steps["model"].coef_[0], dtype=float)
    order = np.argsort(-np.abs(values))[:limit]
    return [{"feature": feature_columns[int(i)], "standardized_coefficient": float(values[int(i)])} for i in order]


def main():
    parser = argparse.ArgumentParser(description="Leave-one-segmentation-fold-out QC training/evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DIR / "oof_qc_dataset.csv"))
    parser.add_argument("--metadata", default=str(DEFAULT_DIR / "oof_qc_metadata.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-prob-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 < args.failure_prob_threshold < 1:
        parser.error("--failure-prob-threshold must be in (0,1)")

    dataset_path, metadata_path, output_dir = Path(args.dataset), Path(args.metadata), Path(args.output_dir)
    if not dataset_path.exists(): raise FileNotFoundError(dataset_path)
    if not metadata_path.exists(): raise FileNotFoundError(metadata_path)

    rows = read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns", []))
    target_columns = set(metadata.get("target_columns", []))
    if not rows or not feature_columns: raise RuntimeError("QC dataset/feature metadata is empty")
    if any(c.startswith("target_") for c in feature_columns):
        raise RuntimeError("GT leakage detected: target_* column present in feature_columns")
    if set(feature_columns) & target_columns:
        raise RuntimeError("GT leakage detected: feature/target column overlap")
    required_targets = {"target_dice", "target_failure", "target_high_quality"}
    missing_targets = required_targets - set(rows[0])
    if missing_targets: raise RuntimeError(f"QC dataset missing targets: {sorted(missing_targets)}")

    folds = np.asarray([int(r["fold"]) for r in rows], dtype=int)
    case_ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    unique_folds = sorted(np.unique(folds).tolist())
    expected_folds = int(metadata.get("n_folds", len(unique_folds)))
    if unique_folds != list(range(expected_folds)):
        raise RuntimeError(f"Expected fold IDs 0..{expected_folds - 1}, found {unique_folds}")

    X = load_matrix(rows, feature_columns)
    y_dice = np.asarray([float(r["target_dice"]) for r in rows])
    y_failure = np.asarray([int(float(r["target_failure"])) for r in rows])
    y_high_quality = np.asarray([int(float(r["target_high_quality"])) for r in rows])
    if len(np.unique(y_failure)) < 2:
        raise RuntimeError("Need both failure and non-failure examples to train QC classifier")

    dice_pred = np.full(len(rows), np.nan)
    failure_prob = np.full(len(rows), np.nan)
    fold_metrics = []
    print("=" * 100)
    print("FOLD-AWARE QC EVALUATION")
    print(f"Cases: {len(rows)} | features: {len(feature_columns)} | folds: {unique_folds}")
    print("Validation: leave one ORIGINAL SEGMENTATION fold out; no QC model sees its test rows during fit")
    print("=" * 100)

    for fold in unique_folds:
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        if len(np.unique(y_failure[train_idx])) < 2:
            raise RuntimeError(f"QC train partition for fold {fold} has only one failure class")
        reg, clf = regression_model(args.seed + fold), failure_model(args.seed + fold)
        reg.fit(X[train_idx], y_dice[train_idx])
        clf.fit(X[train_idx], y_failure[train_idx])
        dice_pred[test_idx] = np.clip(reg.predict(X[test_idx]), 0, 1)
        failure_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        reg_m = regression_metrics(y_dice[test_idx], dice_pred[test_idx])
        cls_m = classification_metrics(y_failure[test_idx], failure_prob[test_idx], args.failure_prob_threshold)
        fold_metrics.append({"fold": int(fold), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                             "n_test_failures": int(y_failure[test_idx].sum()), "regression": reg_m, "failure": cls_m})
        print(f"Fold {fold}: n={len(test_idx)} | Dice MAE={reg_m['mae']:.4f} | failure AUROC={cls_m['auroc']:.4f} | AUPRC={cls_m['auprc']:.4f} | recall={cls_m['recall']:.3f}")

    if not np.isfinite(dice_pred).all() or not np.isfinite(failure_prob).all():
        raise RuntimeError("Some cases did not receive fold-aware QC predictions")
    overall_reg = regression_metrics(y_dice, dice_pred)
    overall_failure = classification_metrics(y_failure, failure_prob, args.failure_prob_threshold)

    final_reg, final_failure = regression_model(args.seed), failure_model(args.seed)
    final_reg.fit(X, y_dice); final_failure.fit(X, y_failure)

    prediction_rows = [{
        "fold": int(folds[i]), "case_id": str(case_ids[i]), "actual_dice": float(y_dice[i]),
        "predicted_dice_oof": float(dice_pred[i]), "actual_failure": int(y_failure[i]),
        "predicted_failure_probability_oof": float(failure_prob[i]),
        "predicted_failure_oof": int(failure_prob[i] >= args.failure_prob_threshold),
        "actual_high_quality": int(y_high_quality[i]),
    } for i in range(len(rows))]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "qc_oof_predictions.csv"
    write_csv(prediction_path, prediction_rows, ["fold", "case_id", "actual_dice", "predicted_dice_oof",
        "actual_failure", "predicted_failure_probability_oof", "predicted_failure_oof", "actual_high_quality"])

    metrics = {
        "version": "qc_tabular_v1", "dataset": str(dataset_path), "metadata": str(metadata_path),
        "n_cases": int(len(rows)), "n_features": int(len(feature_columns)), "folds": unique_folds,
        "validation": "leave-one-original-segmentation-fold-out",
        "failure_dice_threshold": float(metadata.get("failure_dice_threshold", 0.70)),
        "high_quality_dice_threshold": float(metadata.get("high_quality_dice_threshold", 0.80)),
        "failure_probability_decision_threshold": float(args.failure_prob_threshold),
        "regression_model": {"type": "ExtraTreesRegressor", "n_estimators": 500, "max_depth": 3,
                             "min_samples_leaf": 3, "max_features": 0.5},
        "failure_model": {"type": "LogisticRegression", "C": 0.1, "class_weight": "balanced",
                          "scaling": "median imputation + StandardScaler"},
        "oof_regression": overall_reg, "oof_failure": overall_failure, "per_fold": fold_metrics,
        "full_fit_top_regression_features": top_regression_features(final_reg, feature_columns),
        "full_fit_top_failure_features": top_failure_features(final_failure, feature_columns),
        "warning": "These are development OOF QC metrics on the 47-case segmentation development set. Validate policy thresholds on a future locked set.",
    }
    metrics_path = output_dir / "qc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {"version": "qc_tabular_v1", "feature_columns": feature_columns,
        "regression_model": final_reg, "failure_model": final_failure,
        "failure_probability_decision_threshold": float(args.failure_prob_threshold),
        "failure_dice_threshold": float(metadata.get("failure_dice_threshold", 0.70)),
        "high_quality_dice_threshold": float(metadata.get("high_quality_dice_threshold", 0.80)),
        "training_cases": [str(x) for x in case_ids.tolist()],
        "note": "Fit on all 47 development rows after OOF evaluation. Auto-accept thresholds require future locked validation."}
    bundle_path = output_dir / "qc_model_bundle.pkl"
    with bundle_path.open("wb") as handle: pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 100)
    print("QC OOF RESULTS")
    print(f"Dice regression: MAE={overall_reg['mae']:.4f} | RMSE={overall_reg['rmse']:.4f} | R2={overall_reg['r2']:.4f} | Spearman={overall_reg['spearman']:.4f}")
    print(f"Failure classifier: AUROC={overall_failure['auroc']:.4f} | AUPRC={overall_failure['auprc']:.4f} | Brier={overall_failure['brier']:.4f} | Recall={overall_failure['recall']:.4f} | Precision={overall_failure['precision']:.4f}")
    print(f"Confusion @ {args.failure_prob_threshold:.2f}: TN={overall_failure['tn']} FP={overall_failure['fp']} FN={overall_failure['fn']} TP={overall_failure['tp']}")
    print(f"OOF predictions: {prediction_path}")
    print(f"Metrics:         {metrics_path}")
    print(f"Full-fit bundle: {bundle_path}")
    print("Next: inspect OOF failure ranking and design a conservative auto-accept policy.")
    print("=" * 100)


if __name__ == "__main__": main()
