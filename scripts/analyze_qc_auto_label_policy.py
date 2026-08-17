#!/usr/bin/env python3
"""Analyze development OOF QC predictions and propose a conservative auto-label policy.

This script consumes qc_oof_predictions.csv from scripts/train_qc_oof.py. It does NOT
retrain QC. Instead it evaluates selective risk/coverage and searches simple deployment-
compatible gates of the form:

    AUTO_ACCEPT if
        predicted_failure_probability <= P_accept
        AND predicted_dice >= D_accept

Rows not auto-accepted can then be split into REVIEW / ACTIVE_LEARN_PRIORITY using a
separate conservative rule.

Important: policy thresholds are selected on the same 47-case development OOF predictions
used to evaluate QC. Therefore all policy metrics are DEVELOPMENT CALIBRATION ONLY, not
final unbiased auto-label performance. A future locked set is required before production.

Outputs:
  experiments/auto_label_qc_v1/policy_risk_coverage.csv
  experiments/auto_label_qc_v1/policy_candidates.csv
  experiments/auto_label_qc_v1/policy_case_decisions.csv
  experiments/auto_label_qc_v1/qc_policy_development.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path("experiments/auto_label_qc_v1")


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


def wilson_upper_bound(k, n, z=1.959963984540054):
    """Two-sided 95% Wilson interval upper bound for a binomial proportion."""
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return min(1.0, center + half)


def subset_summary(indices, actual_dice, actual_failure, actual_high_quality):
    idx = np.asarray(indices, dtype=int)
    n = int(idx.size)
    if n == 0:
        return {
            "n": 0,
            "mean_actual_dice": float("nan"),
            "median_actual_dice": float("nan"),
            "observed_failures": 0,
            "observed_failure_rate": float("nan"),
            "failure_rate_wilson95_upper": float("nan"),
            "high_quality_count": 0,
            "high_quality_rate": float("nan"),
        }
    failures = int(actual_failure[idx].sum())
    high_quality = int(actual_high_quality[idx].sum())
    return {
        "n": n,
        "mean_actual_dice": float(np.mean(actual_dice[idx])),
        "median_actual_dice": float(np.median(actual_dice[idx])),
        "observed_failures": failures,
        "observed_failure_rate": float(failures / n),
        "failure_rate_wilson95_upper": float(wilson_upper_bound(failures, n)),
        "high_quality_count": high_quality,
        "high_quality_rate": float(high_quality / n),
    }


def risk_coverage_rows(actual_dice, actual_failure, actual_high_quality, failure_prob):
    """Accept increasingly many lowest-predicted-risk cases."""
    order = np.argsort(failure_prob, kind="stable")
    rows = []
    total = len(order)
    for retained in range(1, total + 1):
        idx = order[:retained]
        summary = subset_summary(idx, actual_dice, actual_failure, actual_high_quality)
        rows.append({
            "retained_n": retained,
            "coverage": retained / total,
            "max_failure_probability_in_accepted": float(np.max(failure_prob[idx])),
            **summary,
        })
    return rows


def threshold_values(values, lower_first=False):
    unique = sorted(set(float(x) for x in values))
    if lower_first:
        return [0.0] + unique + [1.0]
    return [0.0] + unique + [1.0]


def search_acceptance_candidates(
    actual_dice,
    actual_failure,
    actual_high_quality,
    predicted_dice,
    failure_prob,
    folds,
    max_failure_rate,
    min_mean_dice,
    min_high_quality_rate,
    min_accepted,
):
    n_total = len(actual_dice)
    p_thresholds = threshold_values(failure_prob)
    d_thresholds = threshold_values(predicted_dice)
    candidates = []

    for p_thr in p_thresholds:
        for d_thr in d_thresholds:
            mask = (failure_prob <= p_thr) & (predicted_dice >= d_thr)
            idx = np.where(mask)[0]
            if len(idx) < min_accepted:
                continue
            summary = subset_summary(idx, actual_dice, actual_failure, actual_high_quality)
            if summary["observed_failure_rate"] > max_failure_rate + 1e-12:
                continue
            if summary["mean_actual_dice"] < min_mean_dice:
                continue
            if summary["high_quality_rate"] < min_high_quality_rate:
                continue

            fold_stats = {}
            for fold in sorted(set(int(x) for x in folds)):
                fold_idx = idx[folds[idx] == fold]
                fold_stats[str(fold)] = subset_summary(
                    fold_idx, actual_dice, actual_failure, actual_high_quality
                )

            candidates.append({
                "accept_failure_probability_max": float(p_thr),
                "accept_predicted_dice_min": float(d_thr),
                "accepted_n": int(len(idx)),
                "coverage": float(len(idx) / n_total),
                **{k: v for k, v in summary.items() if k != "n"},
                "accepted_fold_counts": json.dumps(
                    {fold: stats["n"] for fold, stats in fold_stats.items()}, sort_keys=True
                ),
                "accepted_fold_failures": json.dumps(
                    {fold: stats["observed_failures"] for fold, stats in fold_stats.items()},
                    sort_keys=True,
                ),
            })

    # Highest coverage first. For ties, prefer stricter failure probability then higher Dice.
    candidates.sort(
        key=lambda row: (
            -row["coverage"],
            row["accept_failure_probability_max"],
            -row["accept_predicted_dice_min"],
        )
    )
    return candidates


def assign_three_way_policy(
    predicted_dice,
    failure_prob,
    accept_p,
    accept_dice,
    active_p,
    active_dice,
):
    decisions = []
    for pdice, pfail in zip(predicted_dice, failure_prob):
        if pfail <= accept_p and pdice >= accept_dice:
            decisions.append("AUTO_ACCEPT")
        elif pfail >= active_p or pdice <= active_dice:
            decisions.append("ACTIVE_LEARN_PRIORITY")
        else:
            decisions.append("REVIEW")
    return np.asarray(decisions, dtype=object)


def main():
    parser = argparse.ArgumentParser(
        description="Development risk/coverage analysis for QC-gated auto-labeling"
    )
    parser.add_argument(
        "--predictions", default=str(DEFAULT_DIR / "qc_oof_predictions.csv")
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--max-accepted-failure-rate", type=float, default=0.0)
    parser.add_argument("--min-accepted-mean-dice", type=float, default=0.80)
    parser.add_argument("--min-accepted-high-quality-rate", type=float, default=0.80)
    parser.add_argument("--min-accepted", type=int, default=5)
    parser.add_argument(
        "--active-failure-prob",
        type=float,
        default=0.50,
        help="Development heuristic for ACTIVE_LEARN_PRIORITY among non-accepted rows.",
    )
    parser.add_argument(
        "--active-predicted-dice",
        type=float,
        default=0.70,
        help="Development heuristic for ACTIVE_LEARN_PRIORITY among non-accepted rows.",
    )
    args = parser.parse_args()

    for name, value in [
        ("--max-accepted-failure-rate", args.max_accepted_failure_rate),
        ("--min-accepted-mean-dice", args.min_accepted_mean_dice),
        ("--min-accepted-high-quality-rate", args.min_accepted_high_quality_rate),
        ("--active-failure-prob", args.active_failure_prob),
        ("--active-predicted-dice", args.active_predicted_dice),
    ]:
        if not 0 <= value <= 1:
            parser.error(f"{name} must be in [0,1]")
    if args.min_accepted < 1:
        parser.error("--min-accepted must be >=1")

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    rows = read_csv(predictions_path)
    if not rows:
        raise RuntimeError("QC OOF predictions are empty")

    required = {
        "fold",
        "case_id",
        "actual_dice",
        "predicted_dice_oof",
        "actual_failure",
        "predicted_failure_probability_oof",
        "actual_high_quality",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    case_ids = np.asarray([row["case_id"] for row in rows], dtype=object)
    actual_dice = np.asarray([float(row["actual_dice"]) for row in rows], dtype=float)
    predicted_dice = np.asarray(
        [float(row["predicted_dice_oof"]) for row in rows], dtype=float
    )
    actual_failure = np.asarray([int(row["actual_failure"]) for row in rows], dtype=int)
    failure_prob = np.asarray(
        [float(row["predicted_failure_probability_oof"]) for row in rows], dtype=float
    )
    actual_high_quality = np.asarray(
        [int(row["actual_high_quality"]) for row in rows], dtype=int
    )

    if not (
        np.isfinite(actual_dice).all()
        and np.isfinite(predicted_dice).all()
        and np.isfinite(failure_prob).all()
    ):
        raise RuntimeError("Predictions contain non-finite values")

    risk_rows = risk_coverage_rows(
        actual_dice, actual_failure, actual_high_quality, failure_prob
    )
    candidates = search_acceptance_candidates(
        actual_dice=actual_dice,
        actual_failure=actual_failure,
        actual_high_quality=actual_high_quality,
        predicted_dice=predicted_dice,
        failure_prob=failure_prob,
        folds=folds,
        max_failure_rate=args.max_accepted_failure_rate,
        min_mean_dice=args.min_accepted_mean_dice,
        min_high_quality_rate=args.min_accepted_high_quality_rate,
        min_accepted=args.min_accepted,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_path = output_dir / "policy_risk_coverage.csv"
    candidate_path = output_dir / "policy_candidates.csv"
    write_csv(risk_path, risk_rows)
    write_csv(candidate_path, candidates)

    if candidates:
        chosen = candidates[0]
        accept_p = float(chosen["accept_failure_probability_max"])
        accept_dice = float(chosen["accept_predicted_dice_min"])
        policy_source = "max-coverage candidate satisfying development constraints"
    else:
        # Fail closed: no auto-accept if development data cannot support the requested constraints.
        accept_p = -1.0
        accept_dice = 1.1
        policy_source = "no candidate satisfied constraints; auto-accept disabled"

    decisions = assign_three_way_policy(
        predicted_dice=predicted_dice,
        failure_prob=failure_prob,
        accept_p=accept_p,
        accept_dice=accept_dice,
        active_p=args.active_failure_prob,
        active_dice=args.active_predicted_dice,
    )

    decision_rows = []
    for i, row in enumerate(rows):
        decision_rows.append({
            "fold": int(folds[i]),
            "case_id": str(case_ids[i]),
            "actual_dice": float(actual_dice[i]),
            "predicted_dice_oof": float(predicted_dice[i]),
            "actual_failure": int(actual_failure[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "actual_high_quality": int(actual_high_quality[i]),
            "development_policy_decision": str(decisions[i]),
        })

    decision_path = output_dir / "policy_case_decisions.csv"
    write_csv(decision_path, decision_rows)

    bucket_metrics = {}
    for bucket in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]:
        idx = np.where(decisions == bucket)[0]
        bucket_metrics[bucket] = subset_summary(
            idx, actual_dice, actual_failure, actual_high_quality
        )

    # Useful ranking diagnostics: most dangerous missed failures and safest accepted cases.
    failure_indices = np.where(actual_failure == 1)[0]
    missed_failure_ranking = [
        {
            "case_id": str(case_ids[i]),
            "fold": int(folds[i]),
            "actual_dice": float(actual_dice[i]),
            "predicted_dice_oof": float(predicted_dice[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "development_policy_decision": str(decisions[i]),
        }
        for i in sorted(failure_indices, key=lambda j: failure_prob[j])
    ]

    metadata = {
        "version": "qc_auto_label_policy_dev_v1",
        "predictions": str(predictions_path),
        "n_cases": int(len(rows)),
        "warning": (
            "Policy thresholds and metrics are selected/evaluated on the same 47-case development "
            "OOF QC predictions. They are for workflow design only and require validation on a "
            "future locked set before production auto-accept."
        ),
        "acceptance_constraints": {
            "max_observed_failure_rate": args.max_accepted_failure_rate,
            "min_mean_actual_dice": args.min_accepted_mean_dice,
            "min_actual_high_quality_rate": args.min_accepted_high_quality_rate,
            "min_accepted_cases": args.min_accepted,
        },
        "selected_development_policy": {
            "source": policy_source,
            "auto_accept_if": {
                "predicted_failure_probability_lte": accept_p,
                "predicted_dice_gte": accept_dice,
            },
            "active_learn_priority_if_not_accepted": {
                "predicted_failure_probability_gte": args.active_failure_prob,
                "OR_predicted_dice_lte": args.active_predicted_dice,
            },
            "otherwise": "REVIEW",
        },
        "bucket_metrics": bucket_metrics,
        "n_candidate_acceptance_rules": int(len(candidates)),
        "top_candidate": candidates[0] if candidates else None,
        "actual_failures_ranked_from_lowest_predicted_failure_probability": missed_failure_ranking,
    }
    metadata_path = output_dir / "qc_policy_development.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 100)
    print("QC AUTO-LABEL DEVELOPMENT POLICY")
    print(f"Cases: {len(rows)} | actual failures: {int(actual_failure.sum())}")
    print(
        "Acceptance search constraints: "
        f"failure_rate<={args.max_accepted_failure_rate:.3f}, "
        f"mean Dice>={args.min_accepted_mean_dice:.3f}, "
        f"high-quality rate>={args.min_accepted_high_quality_rate:.3f}"
    )
    print("=" * 100)

    if candidates:
        top = candidates[0]
        print("\nSELECTED DEVELOPMENT AUTO-ACCEPT GATE")
        print(
            f"P(failure) <= {float(top['accept_failure_probability_max']):.4f} AND "
            f"predicted Dice >= {float(top['accept_predicted_dice_min']):.4f}"
        )
        print(
            f"Coverage={float(top['coverage']):.3f} ({int(top['accepted_n'])}/{len(rows)}) | "
            f"observed failures={int(top['observed_failures'])} | "
            f"mean actual Dice={float(top['mean_actual_dice']):.4f} | "
            f"actual high-quality rate={float(top['high_quality_rate']):.3f}"
        )
        print(
            f"95% Wilson upper bound on accepted failure rate: "
            f"{float(top['failure_rate_wilson95_upper']):.3f}"
        )
    else:
        print("\nNo acceptance gate satisfied the requested constraints. AUTO_ACCEPT is disabled.")

    print("\nTHREE-WAY DEVELOPMENT BUCKETS")
    for bucket in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]:
        stats = bucket_metrics[bucket]
        print(
            f"{bucket:<22} n={stats['n']:2d} | mean Dice={stats['mean_actual_dice']:.4f} | "
            f"failures={stats['observed_failures']} | HQ rate={stats['high_quality_rate']:.3f}"
        )

    print("\nLowest QC-risk ACTUAL failures (the cases most dangerous to auto-accept):")
    for row in missed_failure_ranking[:5]:
        print(
            f"  fold {row['fold']} {row['case_id']} | actual Dice={row['actual_dice']:.4f} | "
            f"pred Dice={row['predicted_dice_oof']:.4f} | P(fail)={row['predicted_failure_probability_oof']:.4f} | "
            f"{row['development_policy_decision']}"
        )

    print("\nOutputs:")
    print(f"  {risk_path}")
    print(f"  {candidate_path}")
    print(f"  {decision_path}")
    print(f"  {metadata_path}")
    print(
        "\nNOTE: do not call the selected thresholds production-safe until they pass a future locked set."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
