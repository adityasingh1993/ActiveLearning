#!/usr/bin/env python3
"""Analyze which Round-2 QC safety constraint blocks AUTO_ACCEPT.

This is development-only policy sensitivity analysis on the original 47 leak-safe OOF QC
rows. It does NOT use external31 and it does NOT define production thresholds.

The deployable gate is fixed to the three QC signals that showed useful OOF discrimination:

  predicted Dice >= D
  P(Dice < 0.70) <= F
  P(signed RVE > +20%) <= O20

P(overseg >+50%) is deliberately NOT used as a gating signal because its OOF classifier was
not discriminative. Actual >+50% outcomes remain available as development safety constraints.

The script compares several outcome-constraint scenarios so we can see whether AUTO_ACCEPT is
blocked by Dice safety, the HQ-rate requirement, mild/moderate volume error, or severe volume
error. Only after this analysis should we decide whether a narrow auto-accept region is
scientifically supportable or whether to proceed directly to Round-3 annotation.
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


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson_upper(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return min(1.0, center + half)


def summarize(idx, actual_dice, actual_failure, actual_hq, signed_rve, over20, over50):
    idx = np.asarray(idx, dtype=int)
    n = int(idx.size)
    if n == 0:
        return {
            "accepted_n": 0,
            "coverage": 0.0,
            "mean_dice": float("nan"),
            "median_dice": float("nan"),
            "failures": 0,
            "failure_rate": float("nan"),
            "failure_wilson95_upper": float("nan"),
            "hq_count": 0,
            "hq_rate": float("nan"),
            "median_signed_rve_pct": float("nan"),
            "median_abs_rve_pct": float("nan"),
            "over20_count": 0,
            "over20_rate": float("nan"),
            "over50_count": 0,
            "over50_rate": float("nan"),
        }

    failures = int(actual_failure[idx].sum())
    hq = int(actual_hq[idx].sum())
    n20 = int(over20[idx].sum())
    n50 = int(over50[idx].sum())
    return {
        "accepted_n": n,
        "mean_dice": float(np.mean(actual_dice[idx])),
        "median_dice": float(np.median(actual_dice[idx])),
        "failures": failures,
        "failure_rate": failures / n,
        "failure_wilson95_upper": float(wilson_upper(failures, n)),
        "hq_count": hq,
        "hq_rate": hq / n,
        "median_signed_rve_pct": float(np.median(signed_rve[idx])),
        "median_abs_rve_pct": float(np.median(np.abs(signed_rve[idx]))),
        "over20_count": n20,
        "over20_rate": n20 / n,
        "over50_count": n50,
        "over50_rate": n50 / n,
    }


def threshold_grid(values, direction):
    values = sorted(set(float(v) for v in values))
    if direction == "lte":
        return sorted(set([0.0, 1.0] + values))
    if direction == "gte":
        return sorted(set([0.0, 1.0] + values))
    raise ValueError(direction)


def passes_constraints(s, scenario):
    if s["accepted_n"] < scenario["min_accepted"]:
        return False
    if s["failure_rate"] > scenario["max_failure_rate"] + 1e-12:
        return False
    if s["mean_dice"] < scenario["min_mean_dice"] - 1e-12:
        return False
    if s["hq_rate"] < scenario["min_hq_rate"] - 1e-12:
        return False
    if s["over20_rate"] > scenario["max_over20_rate"] + 1e-12:
        return False
    if s["over50_rate"] > scenario["max_over50_rate"] + 1e-12:
        return False
    return True


def best_gate_for_scenario(
    predicted_dice,
    failure_prob,
    over20_prob,
    actual_dice,
    actual_failure,
    actual_hq,
    signed_rve,
    actual_over20,
    actual_over50,
    scenario,
):
    pf_grid = threshold_grid(failure_prob, "lte")
    pd_grid = threshold_grid(predicted_dice, "gte")
    o20_grid = threshold_grid(over20_prob, "lte")
    n_total = len(actual_dice)

    best = None
    for pf_thr in pf_grid:
        pf_mask = failure_prob <= pf_thr
        for pd_thr in pd_grid:
            base = pf_mask & (predicted_dice >= pd_thr)
            if int(base.sum()) < scenario["min_accepted"]:
                continue
            for o20_thr in o20_grid:
                idx = np.where(base & (over20_prob <= o20_thr))[0]
                if len(idx) < scenario["min_accepted"]:
                    continue
                s = summarize(
                    idx,
                    actual_dice,
                    actual_failure,
                    actual_hq,
                    signed_rve,
                    actual_over20,
                    actual_over50,
                )
                s["coverage"] = len(idx) / n_total
                if not passes_constraints(s, scenario):
                    continue

                candidate = {
                    "accept_failure_probability_max": float(pf_thr),
                    "accept_predicted_dice_min": float(pd_thr),
                    "accept_overseg20_probability_max": float(o20_thr),
                    **s,
                }
                score = (
                    candidate["coverage"],
                    candidate["mean_dice"],
                    candidate["hq_rate"],
                    -candidate["over50_rate"],
                    -candidate["over20_rate"],
                    -candidate["failure_wilson95_upper"],
                )
                if best is None or score > best[0]:
                    best = (score, candidate)

    return None if best is None else best[1]


def main():
    p = argparse.ArgumentParser(description="Round-2 QC AUTO_ACCEPT constraint sensitivity analysis")
    p.add_argument("--predictions", default=str(DEFAULT_DIR / "qc_oof_predictions.csv"))
    p.add_argument("--output-dir", default=str(DEFAULT_DIR))
    p.add_argument("--min-accepted", type=int, default=5)
    args = p.parse_args()
    if args.min_accepted < 1:
        p.error("--min-accepted must be >= 1")

    prediction_path = Path(args.predictions)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    rows = read_csv(prediction_path)
    if not rows:
        raise RuntimeError("QC OOF predictions are empty")

    required = {
        "case_id",
        "actual_dice",
        "predicted_dice_oof",
        "actual_failure",
        "predicted_failure_probability_oof",
        "actual_high_quality",
        "actual_signed_rve",
        "actual_overseg_gt_20pct",
        "predicted_overseg20_probability_oof",
        "actual_overseg_gt_50pct",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    actual_dice = np.asarray([float(r["actual_dice"]) for r in rows], dtype=float)
    predicted_dice = np.asarray([float(r["predicted_dice_oof"]) for r in rows], dtype=float)
    actual_failure = np.asarray([int(float(r["actual_failure"])) for r in rows], dtype=int)
    failure_prob = np.asarray([float(r["predicted_failure_probability_oof"]) for r in rows], dtype=float)
    actual_hq = np.asarray([int(float(r["actual_high_quality"])) for r in rows], dtype=int)
    signed_rve = np.asarray([float(r["actual_signed_rve"]) for r in rows], dtype=float)
    actual_over20 = np.asarray([int(float(r["actual_overseg_gt_20pct"])) for r in rows], dtype=int)
    over20_prob = np.asarray([float(r["predicted_overseg20_probability_oof"]) for r in rows], dtype=float)
    actual_over50 = np.asarray([int(float(r["actual_overseg_gt_50pct"])) for r in rows], dtype=int)

    numeric = [actual_dice, predicted_dice, failure_prob, signed_rve, over20_prob]
    if not all(np.isfinite(x).all() for x in numeric):
        raise RuntimeError("QC OOF predictions contain non-finite values")

    # Each scenario changes outcome constraints only. The deployable gate remains the same 3 signals.
    scenarios = [
        {
            "name": "A_failure_only",
            "description": "Zero Dice failures + mean Dice >=.80; no HQ/volume restriction",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.0,
            "max_over20_rate": 1.0,
            "max_over50_rate": 1.0,
            "min_accepted": args.min_accepted,
        },
        {
            "name": "B_failure_plus_HQ",
            "description": "Current Dice/HQ quality requirement, volume unrestricted",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.80,
            "max_over20_rate": 1.0,
            "max_over50_rate": 1.0,
            "min_accepted": args.min_accepted,
        },
        {
            "name": "C_failure_plus_over20",
            "description": "Zero failures + <=20% of accepted cases above +20% RVE",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.0,
            "max_over20_rate": 0.20,
            "max_over50_rate": 1.0,
            "min_accepted": args.min_accepted,
        },
        {
            "name": "D_failure_plus_over50",
            "description": "Zero failures + zero accepted cases above +50% RVE",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.0,
            "max_over20_rate": 1.0,
            "max_over50_rate": 0.0,
            "min_accepted": args.min_accepted,
        },
        {
            "name": "E_failure_HQ_over50",
            "description": "Zero failures + HQ>=80% + zero severe overseg; mild overseg unrestricted",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.80,
            "max_over20_rate": 1.0,
            "max_over50_rate": 0.0,
            "min_accepted": args.min_accepted,
        },
        {
            "name": "F_current_all_constraints",
            "description": "Current strict policy: fail=0, HQ>=80%, over20<=20%, over50=0",
            "max_failure_rate": 0.0,
            "min_mean_dice": 0.80,
            "min_hq_rate": 0.80,
            "max_over20_rate": 0.20,
            "max_over50_rate": 0.0,
            "min_accepted": args.min_accepted,
        },
    ]

    output_rows = []
    for scenario in scenarios:
        best = best_gate_for_scenario(
            predicted_dice,
            failure_prob,
            over20_prob,
            actual_dice,
            actual_failure,
            actual_hq,
            signed_rve,
            actual_over20,
            actual_over50,
            scenario,
        )
        row = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "candidate_found": int(best is not None),
            "max_failure_rate": scenario["max_failure_rate"],
            "min_mean_dice": scenario["min_mean_dice"],
            "min_hq_rate": scenario["min_hq_rate"],
            "max_over20_rate": scenario["max_over20_rate"],
            "max_over50_rate": scenario["max_over50_rate"],
            "min_accepted": scenario["min_accepted"],
        }
        if best is not None:
            row.update(best)
        output_rows.append(row)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "policy_constraint_sensitivity.csv"
    json_path = outdir / "policy_constraint_sensitivity.json"
    write_csv(csv_path, output_rows)

    payload = {
        "version": "round2_qc_constraint_sensitivity_v1",
        "predictions": str(prediction_path),
        "n_cases": len(rows),
        "gating_signals": [
            "predicted_dice_oof",
            "predicted_failure_probability_oof",
            "predicted_overseg20_probability_oof",
        ],
        "diagnostic_only_signal": "predicted_overseg50_probability_oof",
        "warning": (
            "Development sensitivity analysis on original47 OOF rows only. The external31 benchmark is excluded. "
            "Do not call any threshold production-safe from this analysis alone."
        ),
        "scenarios": output_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 128)
    print("ROUND-2 QC AUTO_ACCEPT CONSTRAINT SENSITIVITY")
    print("Gate signals fixed: predicted Dice + P(failure) + P(overseg>20%). P(overseg>50%) remains diagnostic only.")
    print("=" * 128)
    print(
        f"{'scenario':<30} {'found':>5} {'n':>4} {'cov':>6} {'meanD':>7} {'HQ':>6} "
        f"{'fail':>5} {'>20':>5} {'>50':>5} {'medRVE':>8}"
    )
    for row in output_rows:
        if not row["candidate_found"]:
            print(f"{row['scenario']:<30} {'NO':>5}")
            continue
        print(
            f"{row['scenario']:<30} {'YES':>5} {int(row['accepted_n']):>4d} "
            f"{float(row['coverage']):>6.3f} {float(row['mean_dice']):>7.4f} "
            f"{float(row['hq_rate']):>6.3f} {int(row['failures']):>5d} "
            f"{int(row['over20_count']):>5d} {int(row['over50_count']):>5d} "
            f"{float(row['median_signed_rve_pct']):>+8.1f}%"
        )
        print(
            f"  gate: P(fail)<={float(row['accept_failure_probability_max']):.4f} AND "
            f"predDice>={float(row['accept_predicted_dice_min']):.4f} AND "
            f"P(over20)<={float(row['accept_overseg20_probability_max']):.4f}"
        )

    print("\nInterpretation:")
    print("  - If A works but B fails: HQ>=.80 is the main blocker.")
    print("  - If A/B work but C fails: moderate +20% volume bias is the main blocker.")
    print("  - If A/B work but D/E fail: severe +50% oversegmentation prevents safe acceptance.")
    print("  - If D/E work while F fails: mild/moderate volume bias, not severe failures, is blocking AUTO_ACCEPT.")
    print("  - If even A has poor coverage: QC separation itself is the limiting factor; proceed to Round-3 labels.")
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 128)


if __name__ == "__main__":
    main()
