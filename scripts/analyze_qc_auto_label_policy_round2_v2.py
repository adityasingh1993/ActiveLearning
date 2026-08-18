#!/usr/bin/env python3
"""Round-2 QC policy calibration using only reliable deployable signals.

Policy inputs:
- predicted Dice
- P(Dice < 0.70)
- P(signed RVE > +20%)

The P(signed RVE > +50%) head is diagnostic only because its Round-2 OOF
performance was not discriminative enough for gating. Actual >+50% outcomes
remain an OOF development constraint when evaluating candidate acceptance rules.
External31 is never used for policy fitting.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DIR = Path("experiments/auto_label_qc_round2_raw_v1")


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def wilson_upper(k, n, z=1.959963984540054):
    if n <= 0:
        return float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return min(1.0, c + h)


def summary(idx, dice, fail, hq, signed, over20, over50):
    idx = np.asarray(idx, dtype=int)
    n = int(len(idx))
    if n == 0:
        return {
            "n": 0, "mean_dice": float("nan"), "median_dice": float("nan"),
            "failures": 0, "failure_rate": float("nan"), "failure_wilson95_upper": float("nan"),
            "hq_count": 0, "hq_rate": float("nan"), "median_signed_rve": float("nan"),
            "over20_count": 0, "over20_rate": float("nan"),
            "over50_count": 0, "over50_rate": float("nan"),
        }
    nf = int(fail[idx].sum())
    nh = int(hq[idx].sum())
    n20 = int(over20[idx].sum())
    n50 = int(over50[idx].sum())
    return {
        "n": n,
        "mean_dice": float(np.mean(dice[idx])),
        "median_dice": float(np.median(dice[idx])),
        "failures": nf,
        "failure_rate": nf / n,
        "failure_wilson95_upper": float(wilson_upper(nf, n)),
        "hq_count": nh,
        "hq_rate": nh / n,
        "median_signed_rve": float(np.median(signed[idx])),
        "over20_count": n20,
        "over20_rate": n20 / n,
        "over50_count": n50,
        "over50_rate": n50 / n,
    }


def grid(x, n):
    return sorted(set([0.0, 1.0] + [float(v) for v in np.quantile(x, np.linspace(0, 1, n))]))


def search_policy(dice, fail, hq, signed, over20, over50,
                  pred_dice, p_fail, p_over20, folds, args):
    candidates = []
    for pf in grid(p_fail, args.grid_size):
        for pd in grid(pred_dice, args.grid_size):
            base = (p_fail <= pf) & (pred_dice >= pd)
            if int(base.sum()) < args.min_accepted:
                continue
            for p20 in grid(p_over20, args.grid_size):
                idx = np.where(base & (p_over20 <= p20))[0]
                if len(idx) < args.min_accepted:
                    continue
                s = summary(idx, dice, fail, hq, signed, over20, over50)
                if s["failure_rate"] > args.max_accepted_failure_rate + 1e-12:
                    continue
                if s["over20_rate"] > args.max_accepted_overseg20_rate + 1e-12:
                    continue
                # OOF outcome constraint only; no P(over50) is used in the gate.
                if s["over50_rate"] > args.max_accepted_overseg50_rate + 1e-12:
                    continue
                if s["mean_dice"] < args.min_accepted_mean_dice:
                    continue
                if s["hq_rate"] < args.min_accepted_high_quality_rate:
                    continue
                fold_counts = {
                    str(f): int(np.sum(folds[idx] == f)) for f in sorted(set(folds.tolist()))
                }
                candidates.append({
                    "accept_failure_probability_max": float(pf),
                    "accept_predicted_dice_min": float(pd),
                    "accept_overseg20_probability_max": float(p20),
                    "accepted_n": int(len(idx)),
                    "coverage": float(len(idx) / len(dice)),
                    **{k: v for k, v in s.items() if k != "n"},
                    "accepted_fold_counts": json.dumps(fold_counts, sort_keys=True),
                })
    candidates.sort(key=lambda r: (
        -r["coverage"], r["failure_rate"], r["over50_rate"], r["over20_rate"],
        -r["mean_dice"], r["accept_failure_probability_max"],
        r["accept_overseg20_probability_max"], -r["accept_predicted_dice_min"],
    ))
    return candidates


def assign(pred_dice, p_fail, p_over20, accept, args):
    out = []
    for pd, pf, p20 in zip(pred_dice, p_fail, p_over20):
        if accept is not None and (
            pf <= accept["accept_failure_probability_max"]
            and pd >= accept["accept_predicted_dice_min"]
            and p20 <= accept["accept_overseg20_probability_max"]
        ):
            out.append("AUTO_ACCEPT")
        elif (
            pf >= args.active_failure_prob
            or pd <= args.active_predicted_dice
            or p20 >= args.active_overseg20_prob
        ):
            out.append("ACTIVE_LEARN_PRIORITY")
        else:
            out.append("REVIEW")
    return np.asarray(out, dtype=object)


def main():
    p = argparse.ArgumentParser(description="Round-2 reliable three-signal QC policy calibration")
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
    args = p.parse_args()

    if args.min_accepted < 1 or args.grid_size < 3:
        p.error("Require --min-accepted >=1 and --grid-size >=3")
    for name in [
        "max_accepted_failure_rate", "max_accepted_overseg20_rate", "max_accepted_overseg50_rate",
        "min_accepted_mean_dice", "min_accepted_high_quality_rate",
        "active_failure_prob", "active_predicted_dice", "active_overseg20_prob",
    ]:
        v = getattr(args, name)
        if not 0 <= v <= 1:
            p.error(f"--{name.replace('_', '-')} must be in [0,1]")

    rows = read_csv(args.predictions)
    if not rows:
        raise RuntimeError("QC OOF predictions are empty")
    req = {
        "fold", "case_id", "actual_dice", "predicted_dice_oof", "actual_failure",
        "predicted_failure_probability_oof", "actual_high_quality", "actual_signed_rve",
        "actual_overseg_gt_20pct", "predicted_overseg20_probability_oof",
        "actual_overseg_gt_50pct", "predicted_overseg50_probability_oof",
    }
    miss = req - set(rows[0])
    if miss:
        raise RuntimeError(f"Missing required columns: {sorted(miss)}")

    folds = np.asarray([int(r["fold"]) for r in rows])
    ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    dice = np.asarray([float(r["actual_dice"]) for r in rows])
    pred_dice = np.asarray([float(r["predicted_dice_oof"]) for r in rows])
    fail = np.asarray([int(r["actual_failure"]) for r in rows])
    p_fail = np.asarray([float(r["predicted_failure_probability_oof"]) for r in rows])
    hq = np.asarray([int(r["actual_high_quality"]) for r in rows])
    signed = np.asarray([float(r["actual_signed_rve"]) for r in rows])
    over20 = np.asarray([int(r["actual_overseg_gt_20pct"]) for r in rows])
    p_over20 = np.asarray([float(r["predicted_overseg20_probability_oof"]) for r in rows])
    over50 = np.asarray([int(r["actual_overseg_gt_50pct"]) for r in rows])
    p_over50 = np.asarray([float(r["predicted_overseg50_probability_oof"]) for r in rows])

    candidates = search_policy(
        dice, fail, hq, signed, over20, over50, pred_dice, p_fail, p_over20, folds, args
    )
    top = candidates[0] if candidates else None
    decisions = assign(pred_dice, p_fail, p_over20, top, args)

    outdir = Path(args.output_dir)
    cand_path = outdir / "policy_candidates.csv"
    case_path = outdir / "policy_case_decisions.csv"
    meta_path = outdir / "qc_policy_development.json"
    write_csv(cand_path, candidates)

    case_rows = []
    for i in range(len(rows)):
        case_rows.append({
            "fold": int(folds[i]), "case_id": str(ids[i]),
            "actual_dice": float(dice[i]), "predicted_dice_oof": float(pred_dice[i]),
            "actual_failure": int(fail[i]), "predicted_failure_probability_oof": float(p_fail[i]),
            "actual_signed_rve": float(signed[i]),
            "actual_overseg_gt_20pct": int(over20[i]),
            "predicted_overseg20_probability_oof": float(p_over20[i]),
            "actual_overseg_gt_50pct": int(over50[i]),
            "predicted_overseg50_probability_oof_DIAGNOSTIC_ONLY": float(p_over50[i]),
            "actual_high_quality": int(hq[i]),
            "development_policy_decision": str(decisions[i]),
        })
    write_csv(case_path, case_rows)

    buckets = {
        b: summary(np.where(decisions == b)[0], dice, fail, hq, signed, over20, over50)
        for b in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]
    }

    policy = {
        "version": "qc_auto_label_policy_round2_overseg_dev_v2",
        "predictions": str(args.predictions),
        "n_cases": len(rows),
        "warning": "Development calibration on original47 OOF only; external31 remains excluded.",
        "policy_signals": {
            "used": ["predicted_dice", "predicted_failure_probability", "predicted_overseg20_probability"],
            "diagnostic_only": ["predicted_overseg50_probability"],
            "severe_head_oof": {"auroc": 0.4106, "auprc": 0.1438, "recall": 0.1667, "precision": 0.0769},
        },
        "acceptance_constraints": {
            "max_observed_failure_rate": args.max_accepted_failure_rate,
            "max_observed_overseg20_rate": args.max_accepted_overseg20_rate,
            "max_observed_overseg50_rate": args.max_accepted_overseg50_rate,
            "min_mean_actual_dice": args.min_accepted_mean_dice,
            "min_actual_high_quality_rate": args.min_accepted_high_quality_rate,
            "min_accepted_cases": args.min_accepted,
            "note": "overseg50 is an OOF outcome constraint only, not a deployable policy signal",
        },
        "selected_development_policy": {
            "source": "max-coverage reliable three-signal candidate" if top else "auto-accept disabled",
            "auto_accept_if": None if top is None else {
                "predicted_failure_probability_lte": float(top["accept_failure_probability_max"]),
                "predicted_dice_gte": float(top["accept_predicted_dice_min"]),
                "predicted_overseg20_probability_lte": float(top["accept_overseg20_probability_max"]),
            },
            "active_learn_priority_if_not_accepted": {
                "predicted_failure_probability_gte": args.active_failure_prob,
                "OR_predicted_dice_lte": args.active_predicted_dice,
                "OR_predicted_overseg20_probability_gte": args.active_overseg20_prob,
            },
            "otherwise": "REVIEW",
        },
        "bucket_metrics": buckets,
        "n_candidate_acceptance_rules": len(candidates),
        "top_candidate": top,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print("=" * 112)
    print("ROUND-2 QC DEVELOPMENT POLICY — RELIABLE 3-SIGNAL GATE")
    print(f"Cases={len(rows)} | failures={int(fail.sum())} | overseg>20={int(over20.sum())} | overseg>50={int(over50.sum())}")
    print("Used: predDice + P(failure) + P(overseg>20%)")
    print("P(overseg>50%): diagnostic only")
    print("=" * 112)
    if top:
        print("\nSELECTED DEVELOPMENT AUTO-ACCEPT GATE")
        print(
            f"P(failure)<={top['accept_failure_probability_max']:.4f} AND "
            f"predDice>={top['accept_predicted_dice_min']:.4f} AND "
            f"P(over20)<={top['accept_overseg20_probability_max']:.4f}"
        )
        print(
            f"Coverage={top['coverage']:.3f} ({top['accepted_n']}/{len(rows)}) | "
            f"failures={top['failures']} | over20={top['over20_count']} | over50={top['over50_count']} | "
            f"mean Dice={top['mean_dice']:.4f} | HQ={top['hq_rate']:.3f}"
        )
        print(f"95% Wilson upper accepted failure bound: {top['failure_wilson95_upper']:.3f}")
    else:
        print("\nNo candidate satisfied constraints; AUTO_ACCEPT disabled.")

    print("\nTHREE-WAY DEVELOPMENT BUCKETS")
    for b in ["AUTO_ACCEPT", "REVIEW", "ACTIVE_LEARN_PRIORITY"]:
        s = buckets[b]
        if s["n"] == 0:
            print(f"{b:<22} n=0")
        else:
            print(
                f"{b:<22} n={s['n']:2d} | meanDice={s['mean_dice']:.4f} | failures={s['failures']} | "
                f"over20={s['over20_count']} | over50={s['over50_count']} | "
                f"medSignedRVE={s['median_signed_rve']:+.1f}% | HQ={s['hq_rate']:.3f}"
            )
    print("\nOutputs:")
    print(f"  {cand_path}\n  {case_path}\n  {meta_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
