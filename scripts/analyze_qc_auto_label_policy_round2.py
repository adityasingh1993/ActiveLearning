#!/usr/bin/env python3
"""Calibrate a Round-2 over-segmentation-aware development QC policy.

Consumes qc_oof_predictions.csv from train_qc_oof_round2.py and searches a conservative
four-signal auto-accept gate over OOF predictions:

  P(failure) <= A
  predicted Dice >= B
  P(overseg >+20%) <= C
  P(overseg >+50%) <= D

The search is performed only on the original 47 segmentation OOF development cases. The
external 31-case benchmark is never used for policy fitting or threshold selection.

ACTIVE_LEARN_PRIORITY is assigned to non-accepted cases when any deployable risk signal is
high: failure probability, low predicted Dice, overseg20 probability, or overseg50 probability.
Everything else is REVIEW.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

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


def wilson_upper_bound(k, n, z=1.959963984540054):
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return min(1.0, center + half)


def subset_summary(idx, actual_dice, actual_failure, actual_hq, signed_rve, over20, over50):
    idx = np.asarray(idx, dtype=int)
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
            "median_signed_rve_pct": float("nan"),
            "overseg20_count": 0,
            "overseg20_rate": float("nan"),
            "overseg50_count": 0,
            "overseg50_rate": float("nan"),
        }
    failures = int(actual_failure[idx].sum())
    hq = int(actual_hq[idx].sum())
    o20 = int(over20[idx].sum())
    o50 = int(over50[idx].sum())
    return {
        "n": n,
        "mean_actual_dice": float(np.mean(actual_dice[idx])),
        "median_actual_dice": float(np.median(actual_dice[idx])),
        "observed_failures": failures,
        "observed_failure_rate": failures / n,
        "failure_rate_wilson95_upper": float(wilson_upper_bound(failures, n)),
        "high_quality_count": hq,
        "high_quality_rate": hq / n,
        "median_signed_rve_pct": float(np.median(signed_rve[idx])),
        "overseg20_count": o20,
        "overseg20_rate": o20 / n,
        "overseg50_count": o50,
        "overseg50_rate": o50 / n,
    }


def quantile_grid(values, n=11, include_bounds=True):
    values = np.asarray(values, dtype=float)
    q = np.linspace(0.0, 1.0, n)
    out = sorted(set(float(x) for x in np.quantile(values, q)))
    if include_bounds:
        out = sorted(set([0.0, 1.0] + out))
    return out


def search_candidates(
    actual_dice,
    actual_failure,
    actual_hq,
    signed_rve,
    actual_over20,
    actual_over50,
    predicted_dice,
    failure_prob,
    over20_prob,
    over50_prob,
    folds,
    max_failure_rate,
    max_over20_rate,
    max_over50_rate,
    min_mean_dice,
    min_hq_rate,
    min_accepted,
    grid_size,
):
    pf_grid = quantile_grid(failure_prob, grid_size)
    pd_grid = quantile_grid(predicted_dice, grid_size)
    o20_grid = quantile_grid(over20_prob, grid_size)
    o50_grid = quantile_grid(over50_prob, grid_size)
    candidates = []
    n_total = len(actual_dice)

    for pf_thr in pf_grid:
        for pd_thr in pd_grid:
            base = (failure_prob <= pf_thr) & (predicted_dice >= pd_thr)
            if int(base.sum()) < min_accepted:
                continue
            for o20_thr in o20_grid:
                partial = base & (over20_prob <= o20_thr)
                if int(partial.sum()) < min_accepted:
                    continue
                for o50_thr in o50_grid:
                    idx = np.where(partial & (over50_prob <= o50_thr))[0]
                    if len(idx) < min_accepted:
                        continue
                    s = subset_summary(
                        idx, actual_dice, actual_failure, actual_hq,
                        signed_rve, actual_over20, actual_over50
                    )
                    if s["observed_failure_rate"] > max_failure_rate + 1e-12:
                        continue
                    if s["overseg20_rate"] > max_over20_rate + 1e-12:
                        continue
                    if s["overseg50_rate"] > max_over50_rate + 1e-12:
                        continue
                    if s["mean_actual_dice"] < min_mean_dice:
                        continue
                    if s["high_quality_rate"] < min_hq_rate:
                        continue

                    fold_counts = {}
                    fold_fail = {}
                    fold_over50 = {}
                    for fold in sorted(set(int(x) for x in folds)):
                        fi = idx[folds[idx] == fold]
                        fs = subset_summary(
                            fi, actual_dice, actual_failure, actual_hq,
                            signed_rve, actual_over20, actual_over50
                        )
                        fold_counts[str(fold)] = fs["n"]
                        fold_fail[str(fold)] = fs["observed_failures"]
                        fold_over50[str(fold)] = fs["overseg50_count"]

                    candidates.append({
                        "accept_failure_probability_max": float(pf_thr),
                        "accept_predicted_dice_min": float(pd_thr),
                        "accept_overseg20_probability_max": float(o20_thr),
                        "accept_overseg50_probability_max": float(o50_thr),
                        "accepted_n": int(len(idx)),
                        "coverage": len(idx) / n_total,
                        **{k: v for k, v in s.items() if k != "n"},
                        "accepted_fold_counts": json.dumps(fold_counts, sort_keys=True),
                        "accepted_fold_failures": json.dumps(fold_fail, sort_keys=True),
                        "accepted_fold_overseg50": json.dumps(fold_over50, sort_keys=True),
                    })

    candidates.sort(key=lambda r: (
        -r["coverage"],
        r["observed_failure_rate"],
        r["overseg50_rate"],
        r["overseg20_rate"],
        -r["mean_actual_dice"],
        r["accept_failure_probability_max"],
        r["accept_overseg50_probability_max"],
        r["accept_overseg20_probability_max"],
        -r["accept_predicted_dice_min"],
    ))
    return candidates


def assign_policy(
    predicted_dice,
    failure_prob,
    over20_prob,
    over50_prob,
    accept_p,
    accept_dice,
    accept_o20,
    accept_o50,
    active_p,
    active_dice,
    active_o20,
    active_o50,
):
    out = []
    for pd, pf, p20, p50 in zip(predicted_dice, failure_prob, over20_prob, over50_prob):
        if pf <= accept_p and pd >= accept_dice and p20 <= accept_o20 and p50 <= accept_o50:
            out.append("AUTO_ACCEPT")
        elif pf >= active_p or pd <= active_dice or p20 >= active_o20 or p50 >= active_o50:
            out.append("ACTIVE_LEARN_PRIORITY")
        else:
            out.append("REVIEW")
    return np.asarray(out, dtype=object)


def main():
    p = argparse.ArgumentParser(description="Round-2 four-signal QC development policy calibration")
    p.add_argument("--predictions", default=str(DEFAULT_DIR / "qc_oof_predictions.csv"))
    p.add_argument("--output-dir", default=str(DEFAULT_DIR))
    p.add_argument("--max-accepted-failure-rate", type=float, default=0.0)
    p.add_argument("--max-accepted-overseg20-rate", type=float, default=0.20)
    p.add_argument("--max-accepted-overseg50-rate", type=float, default=0.0)
    p.add_argument("--min-accepted-mean-dice", type=float, default=0.80)
    p.add_argument("--min-accepted-high-quality-rate", type=float, default=0.80)
    p.add_argument("--min-accepted", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=11)
    p.add_argument("--active-failure-prob", type=float, default=0.50)
    p.add_argument("--active-predicted-dice", type=float, default=0.70)
    p.add_argument("--active-overseg20-prob", type=float, default=0.50)
    p.add_argument("--active-overseg50-prob", type=float, default=0.50)
    args = p.parse_args()

    for name, value in {
        "max-accepted-failure-rate": args.max_accepted_failure_rate,
        "max-accepted-overseg20-rate": args.max_accepted_overseg20_rate,
        "max-accepted-overseg50-rate": args.max_accepted_overseg50_rate,
        "min-accepted-mean-dice": args.min_accepted_mean_dice,
        "min-accepted-high-quality-rate": args.min_accepted_high_quality_rate,
        "active-failure-prob": args.active_failure_prob,
        "active-predicted-dice": args.active_predicted_dice,
        "active-overseg20-prob": args.active_overseg20_prob,
        "active-overseg50-prob": args.active_overseg50_prob,
    }.items():
        if not 0 <= value <= 1:
            p.error(f"--{name} must be in [0,1]")
    if args.min_accepted < 1:
        p.error("--min-accepted must be >=1")
    if args.grid_size < 3:
        p.error("--grid-size must be >=3")

    rows = read_csv(args.predictions)
    if not rows:
        raise RuntimeError("Round-2 QC OOF predictions are empty")
    required = {
        "fold", "case_id", "actual_dice", "predicted_dice_oof",
        "actual_failure", "predicted_failure_probability_oof", "actual_high_quality",
        "actual_signed_rve", "actual_overseg_gt_20pct", "predicted_overseg20_probability_oof",
        "actual_overseg_gt_50pct", "predicted_overseg50_probability_oof",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    folds = np.asarray([int(r["fold"]) for r in rows], dtype=int)
    case_ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    actual_dice = np.asarray([float(r["actual_dice"]) for r in rows])
    predicted_dice = np.asarray([float(r["predicted_dice_oof"]) for r in rows])
    actual_failure = np.asarray([int(r["actual_failure"]) for r in rows])
    failure_prob = np.asarray([float(r["predicted_failure_probability_oof"]) for r in rows])
    actual_hq = np.asarray([int(r["actual_high_quality"]) for r in rows])
    signed_rve = np.asarray([float(r["actual_signed_rve"]) for r in rows])
    actual_over20 = np.asarray([int(r["actual_overseg_gt_20pct"]) for r in rows])
    over20_prob = np.asarray([float(r["predicted_overseg20_probability_oof"]) for r in rows])
    actual_over50 = np.asarray([int(r["actual_overseg_gt_50pct"]) for r in rows])
    over50_prob = np.asarray([float(r["predicted_overseg50_probability_oof"]) for r in rows])

    numeric = [actual_dice, predicted_dice, failure_prob, signed_rve, over20_prob, over50_prob]
    if not all(np.isfinite(x).all() for x in numeric):
        raise RuntimeError("Round-2 QC predictions contain non-finite values")

    candidates = search_candidates(
        actual_dice, actual_failure, actual_hq, signed_rve, actual_over20, actual_over50,
        predicted_dice, failure_prob, over20_prob, over50_prob, folds,
        args.max_accepted_failure_rate, args.max_accepted_overseg20_rate,
        args.max_accepted_overseg50_rate, args.min_accepted_mean_dice,
        args.min_accepted_high_quality_rate, args.min_accepted, args.grid_size,
    )

    if candidates:
        top = candidates[0]
        accept_p = float(top["accept_failure_probability_max"])
        accept_d = float(top["accept_predicted_dice_min"])
        accept_o20 = float(top["accept_overseg20_probability_max"])
        accept_o50 = float(top["accept_overseg50_probability_max"])
        source = "max-coverage four-signal candidate satisfying development constraints"
    else:
        accept_p, accept_d, accept_o20, accept_o50 = -1.0, 1.1, -1.0, -1.0
        source = "no candidate satisfied constraints; auto-accept disabled"

    decisions = assign_policy(
        predicted_dice, failure_prob, over20_prob, over50_prob,
        accept_p, accept_d, accept_o20, accept_o50,
        args.active_failure_prob, args.active_predicted_dice,
        args.active_overseg20_prob, args.active_overseg50_prob,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cand_path = outdir / "policy_candidates.csv"
    write_csv(cand_path, candidates)

    case_rows = []
    for i in range(len(rows)):
        case_rows.append({
            "fold": int(folds[i]),
            "case_id": str(case_ids[i]),
            "actual_dice": float(actual_dice[i]),
            "predicted_dice_oof": float(predicted_dice[i]),
            "actual_failure": int(actual_failure[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "actual_signed_rve": float(signed_rve[i]),
            "actual_overseg_gt_20pct": int(actual_over20[i]),
            "predicted_overseg20_probability_oof": float(over20_prob[i]),
            "actual_overseg_gt_50pct": int(actual_over50[i]),
            "predicted_overseg50_probability_oof": float(over50_prob[i]),
            "actual_high_quality": int(actual_hq[i]),
            "development_policy_decision": str(decisions[i]),
        })
    case_path = outdir / "policy_case_decisions.csv"
    write_csv(case_path, case_rows)

    buckets = {
        b: subset_summary(
            np.where(decisions == b)[0], actual_dice, actual_failure, actual_hq,
            signed_rve, actual_over20, actual_over50
        )
        for b in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]
    }

    actual_failure_idx = np.where(actual_failure == 1)[0]
    severe_idx = np.where(actual_over50 == 1)[0]
    dangerous_failures = [
        {
            "case_id": str(case_ids[i]),
            "fold": int(folds[i]),
            "actual_dice": float(actual_dice[i]),
            "predicted_dice_oof": float(predicted_dice[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "predicted_overseg20_probability_oof": float(over20_prob[i]),
            "predicted_overseg50_probability_oof": float(over50_prob[i]),
            "development_policy_decision": str(decisions[i]),
        }
        for i in sorted(actual_failure_idx, key=lambda j: failure_prob[j])
    ]
    dangerous_overseg = [
        {
            "case_id": str(case_ids[i]),
            "fold": int(folds[i]),
            "actual_signed_rve": float(signed_rve[i]),
            "actual_dice": float(actual_dice[i]),
            "predicted_overseg20_probability_oof": float(over20_prob[i]),
            "predicted_overseg50_probability_oof": float(over50_prob[i]),
            "predicted_failure_probability_oof": float(failure_prob[i]),
            "development_policy_decision": str(decisions[i]),
        }
        for i in sorted(severe_idx, key=lambda j: over50_prob[j])
    ]

    metadata = {
        "version": "qc_auto_label_policy_round2_overseg_dev_v1",
        "predictions": str(args.predictions),
        "n_cases": len(rows),
        "warning": "Development calibration on original47 OOF predictions only; external31 is excluded and must remain confirmatory.",
        "acceptance_constraints": {
            "max_observed_failure_rate": args.max_accepted_failure_rate,
            "max_observed_overseg20_rate": args.max_accepted_overseg20_rate,
            "max_observed_overseg50_rate": args.max_accepted_overseg50_rate,
            "min_mean_actual_dice": args.min_accepted_mean_dice,
            "min_actual_high_quality_rate": args.min_accepted_high_quality_rate,
            "min_accepted_cases": args.min_accepted,
        },
        "selected_development_policy": {
            "source": source,
            "auto_accept_if": {
                "predicted_failure_probability_lte": accept_p,
                "predicted_dice_gte": accept_d,
                "predicted_overseg20_probability_lte": accept_o20,
                "predicted_overseg50_probability_lte": accept_o50,
            },
            "active_learn_priority_if_not_accepted": {
                "predicted_failure_probability_gte": args.active_failure_prob,
                "OR_predicted_dice_lte": args.active_predicted_dice,
                "OR_predicted_overseg20_probability_gte": args.active_overseg20_prob,
                "OR_predicted_overseg50_probability_gte": args.active_overseg50_prob,
            },
            "otherwise": "REVIEW",
        },
        "bucket_metrics": buckets,
        "n_candidate_acceptance_rules": len(candidates),
        "top_candidate": candidates[0] if candidates else None,
        "actual_failures_ranked_lowest_predicted_failure_first": dangerous_failures,
        "actual_severe_overseg_ranked_lowest_predicted_overseg50_first": dangerous_overseg,
    }
    meta_path = outdir / "qc_policy_development.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 112)
    print("ROUND-2 OVERSEG-AWARE QC DEVELOPMENT POLICY")
    print(
        f"Cases={len(rows)} | failures={int(actual_failure.sum())} | "
        f"overseg>20={int(actual_over20.sum())} | overseg>50={int(actual_over50.sum())}"
    )
    print("=" * 112)
    if candidates:
        top = candidates[0]
        print("\nSELECTED DEVELOPMENT AUTO-ACCEPT GATE")
        print(
            f"P(failure)<={top['accept_failure_probability_max']:.4f} AND "
            f"predDice>={top['accept_predicted_dice_min']:.4f} AND "
            f"P(over20)<={top['accept_overseg20_probability_max']:.4f} AND "
            f"P(over50)<={top['accept_overseg50_probability_max']:.4f}"
        )
        print(
            f"Coverage={top['coverage']:.3f} ({top['accepted_n']}/{len(rows)}) | "
            f"failures={top['observed_failures']} | over20={top['overseg20_count']} | "
            f"over50={top['overseg50_count']} | mean Dice={top['mean_actual_dice']:.4f} | "
            f"HQ rate={top['high_quality_rate']:.3f}"
        )
        print(f"95% Wilson upper bound on accepted failure rate: {top['failure_rate_wilson95_upper']:.3f}")
    else:
        print("\nNo four-signal acceptance gate satisfied constraints. AUTO_ACCEPT disabled.")

    print("\nTHREE-WAY DEVELOPMENT BUCKETS")
    for b in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]:
        s = buckets[b]
        if s["n"] == 0:
            print(f"{b:<22} n=0")
        else:
            print(
                f"{b:<22} n={s['n']:2d} | mean Dice={s['mean_actual_dice']:.4f} | "
                f"failures={s['observed_failures']} | over20={s['overseg20_count']} | "
                f"over50={s['overseg50_count']} | med signedRVE={s['median_signed_rve_pct']:+.1f}% | "
                f"HQ={s['high_quality_rate']:.3f}"
            )

    print("\nLowest-risk ACTUAL severe-overseg cases:")
    for r in dangerous_overseg[:6]:
        print(
            f"  fold {r['fold']} {r['case_id']} | signedRVE={r['actual_signed_rve']:+.1f}% | "
            f"Dice={r['actual_dice']:.4f} | P(over50)={r['predicted_overseg50_probability_oof']:.4f} | "
            f"P(fail)={r['predicted_failure_probability_oof']:.4f} | {r['development_policy_decision']}"
        )

    print("\nOutputs:")
    for path in [cand_path, case_path, meta_path]:
        print(f"  {path}")
    print("\nNOTE: This policy is development-only until evaluated prospectively/confirmatorily.")
    print("=" * 112)


if __name__ == "__main__":
    main()
