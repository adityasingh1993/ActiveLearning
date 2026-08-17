#!/usr/bin/env python3
"""Analyze development OOF QC predictions and propose a conservative auto-label policy.

Consumes qc_oof_predictions.csv from scripts/train_qc_oof.py and evaluates selective
risk/coverage plus a simple deployment-compatible three-way gate:
AUTO_ACCEPT / REVIEW / ACTIVE_LEARN_PRIORITY.

Policy thresholds are selected on the same 47-case development OOF predictions used to
evaluate QC, so all policy metrics are development calibration only. Validate on a future
locked set before production auto-accept.
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
        writer.writeheader(); writer.writerows(rows)


def wilson_upper_bound(k, n, z=1.959963984540054):
    if n <= 0: return float("nan")
    p = k / n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return min(1.0, center + half)


def subset_summary(indices, actual_dice, actual_failure, actual_high_quality):
    idx = np.asarray(indices, dtype=int); n = int(idx.size)
    if n == 0:
        return {"n": 0, "mean_actual_dice": float("nan"), "median_actual_dice": float("nan"),
                "observed_failures": 0, "observed_failure_rate": float("nan"),
                "failure_rate_wilson95_upper": float("nan"), "high_quality_count": 0,
                "high_quality_rate": float("nan")}
    failures = int(actual_failure[idx].sum()); high_quality = int(actual_high_quality[idx].sum())
    return {"n": n, "mean_actual_dice": float(np.mean(actual_dice[idx])),
            "median_actual_dice": float(np.median(actual_dice[idx])),
            "observed_failures": failures, "observed_failure_rate": failures/n,
            "failure_rate_wilson95_upper": float(wilson_upper_bound(failures, n)),
            "high_quality_count": high_quality, "high_quality_rate": high_quality/n}


def risk_coverage_rows(actual_dice, actual_failure, actual_high_quality, failure_prob):
    order = np.argsort(failure_prob, kind="stable"); out=[]; total=len(order)
    for retained in range(1,total+1):
        idx=order[:retained]; s=subset_summary(idx, actual_dice, actual_failure, actual_high_quality)
        out.append({"retained_n": retained, "coverage": retained/total,
                    "max_failure_probability_in_accepted": float(np.max(failure_prob[idx])), **s})
    return out


def thresholds(values): return [0.0] + sorted(set(float(x) for x in values)) + [1.0]


def search_candidates(actual_dice, actual_failure, actual_high_quality, predicted_dice,
                      failure_prob, folds, max_failure_rate, min_mean_dice,
                      min_high_quality_rate, min_accepted):
    candidates=[]; n_total=len(actual_dice)
    for p_thr in thresholds(failure_prob):
        for d_thr in thresholds(predicted_dice):
            idx=np.where((failure_prob<=p_thr)&(predicted_dice>=d_thr))[0]
            if len(idx)<min_accepted: continue
            s=subset_summary(idx, actual_dice, actual_failure, actual_high_quality)
            if s["observed_failure_rate"]>max_failure_rate+1e-12: continue
            if s["mean_actual_dice"]<min_mean_dice or s["high_quality_rate"]<min_high_quality_rate: continue
            fold_counts={}; fold_failures={}
            for fold in sorted(set(int(x) for x in folds)):
                fi=idx[folds[idx]==fold]; fs=subset_summary(fi,actual_dice,actual_failure,actual_high_quality)
                fold_counts[str(fold)]=fs["n"]; fold_failures[str(fold)]=fs["observed_failures"]
            candidates.append({"accept_failure_probability_max": float(p_thr),
                "accept_predicted_dice_min": float(d_thr), "accepted_n": int(len(idx)),
                "coverage": len(idx)/n_total, **{k:v for k,v in s.items() if k!="n"},
                "accepted_fold_counts": json.dumps(fold_counts,sort_keys=True),
                "accepted_fold_failures": json.dumps(fold_failures,sort_keys=True)})
    candidates.sort(key=lambda r:(-r["coverage"],r["accept_failure_probability_max"],-r["accept_predicted_dice_min"]))
    return candidates


def assign_policy(predicted_dice, failure_prob, accept_p, accept_dice, active_p, active_dice):
    out=[]
    for pd,pf in zip(predicted_dice,failure_prob):
        if pf<=accept_p and pd>=accept_dice: out.append("AUTO_ACCEPT")
        elif pf>=active_p or pd<=active_dice: out.append("ACTIVE_LEARN_PRIORITY")
        else: out.append("REVIEW")
    return np.asarray(out,dtype=object)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",default=str(DEFAULT_DIR/"qc_oof_predictions.csv")); p.add_argument("--output-dir",default=str(DEFAULT_DIR))
    p.add_argument("--max-accepted-failure-rate",type=float,default=0.0); p.add_argument("--min-accepted-mean-dice",type=float,default=0.80)
    p.add_argument("--min-accepted-high-quality-rate",type=float,default=0.80); p.add_argument("--min-accepted",type=int,default=5)
    p.add_argument("--active-failure-prob",type=float,default=0.50); p.add_argument("--active-predicted-dice",type=float,default=0.70)
    a=p.parse_args()
    for name,val in [("max failure rate",a.max_accepted_failure_rate),("min mean dice",a.min_accepted_mean_dice),("min HQ rate",a.min_accepted_high_quality_rate),("active failure prob",a.active_failure_prob),("active predicted dice",a.active_predicted_dice)]:
        if not 0<=val<=1: p.error(f"{name} must be in [0,1]")
    if a.min_accepted<1: p.error("--min-accepted must be >=1")

    rows=read_csv(a.predictions)
    if not rows: raise RuntimeError("QC OOF predictions are empty")
    required={"fold","case_id","actual_dice","predicted_dice_oof","actual_failure","predicted_failure_probability_oof","actual_high_quality"}
    missing=required-set(rows[0]);
    if missing: raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    folds=np.asarray([int(r["fold"]) for r in rows]); case_ids=np.asarray([r["case_id"] for r in rows],dtype=object)
    actual_dice=np.asarray([float(r["actual_dice"]) for r in rows]); predicted_dice=np.asarray([float(r["predicted_dice_oof"]) for r in rows])
    actual_failure=np.asarray([int(r["actual_failure"]) for r in rows]); failure_prob=np.asarray([float(r["predicted_failure_probability_oof"]) for r in rows])
    actual_hq=np.asarray([int(r["actual_high_quality"]) for r in rows])
    if not(np.isfinite(actual_dice).all() and np.isfinite(predicted_dice).all() and np.isfinite(failure_prob).all()): raise RuntimeError("Predictions contain non-finite values")

    risk=risk_coverage_rows(actual_dice,actual_failure,actual_hq,failure_prob)
    candidates=search_candidates(actual_dice,actual_failure,actual_hq,predicted_dice,failure_prob,folds,a.max_accepted_failure_rate,a.min_accepted_mean_dice,a.min_accepted_high_quality_rate,a.min_accepted)
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    risk_path=outdir/"policy_risk_coverage.csv"; cand_path=outdir/"policy_candidates.csv"
    write_csv(risk_path,risk); write_csv(cand_path,candidates)

    if candidates:
        accept_p=float(candidates[0]["accept_failure_probability_max"]); accept_d=float(candidates[0]["accept_predicted_dice_min"]); source="max-coverage candidate satisfying development constraints"
    else: accept_p,accept_d,source=-1.0,1.1,"no candidate satisfied constraints; auto-accept disabled"
    decisions=assign_policy(predicted_dice,failure_prob,accept_p,accept_d,a.active_failure_prob,a.active_predicted_dice)
    dec_rows=[{"fold":int(folds[i]),"case_id":str(case_ids[i]),"actual_dice":float(actual_dice[i]),"predicted_dice_oof":float(predicted_dice[i]),"actual_failure":int(actual_failure[i]),"predicted_failure_probability_oof":float(failure_prob[i]),"actual_high_quality":int(actual_hq[i]),"development_policy_decision":str(decisions[i])} for i in range(len(rows))]
    dec_path=outdir/"policy_case_decisions.csv"; write_csv(dec_path,dec_rows)
    buckets={b:subset_summary(np.where(decisions==b)[0],actual_dice,actual_failure,actual_hq) for b in ["AUTO_ACCEPT","REVIEW","ACTIVE_LEARN_PRIORITY"]}
    failure_idx=np.where(actual_failure==1)[0]
    dangerous=[{"case_id":str(case_ids[i]),"fold":int(folds[i]),"actual_dice":float(actual_dice[i]),"predicted_dice_oof":float(predicted_dice[i]),"predicted_failure_probability_oof":float(failure_prob[i]),"development_policy_decision":str(decisions[i])} for i in sorted(failure_idx,key=lambda j:failure_prob[j])]
    meta={"version":"qc_auto_label_policy_dev_v1","predictions":str(a.predictions),"n_cases":len(rows),
          "warning":"Development calibration only; validate on a future locked set before production auto-accept.",
          "acceptance_constraints":{"max_observed_failure_rate":a.max_accepted_failure_rate,"min_mean_actual_dice":a.min_accepted_mean_dice,"min_actual_high_quality_rate":a.min_accepted_high_quality_rate,"min_accepted_cases":a.min_accepted},
          "selected_development_policy":{"source":source,"auto_accept_if":{"predicted_failure_probability_lte":accept_p,"predicted_dice_gte":accept_d},"active_learn_priority_if_not_accepted":{"predicted_failure_probability_gte":a.active_failure_prob,"OR_predicted_dice_lte":a.active_predicted_dice},"otherwise":"REVIEW"},
          "bucket_metrics":buckets,"n_candidate_acceptance_rules":len(candidates),"top_candidate":candidates[0] if candidates else None,"actual_failures_ranked_from_lowest_predicted_failure_probability":dangerous}
    meta_path=outdir/"qc_policy_development.json"; meta_path.write_text(json.dumps(meta,indent=2),encoding="utf-8")

    print("="*100); print("QC AUTO-LABEL DEVELOPMENT POLICY"); print(f"Cases: {len(rows)} | actual failures: {int(actual_failure.sum())}")
    print(f"Acceptance search constraints: failure_rate<={a.max_accepted_failure_rate:.3f}, mean Dice>={a.min_accepted_mean_dice:.3f}, high-quality rate>={a.min_accepted_high_quality_rate:.3f}"); print("="*100)
    if candidates:
        top=candidates[0]; print("\nSELECTED DEVELOPMENT AUTO-ACCEPT GATE")
        print(f"P(failure) <= {top['accept_failure_probability_max']:.4f} AND predicted Dice >= {top['accept_predicted_dice_min']:.4f}")
        print(f"Coverage={top['coverage']:.3f} ({top['accepted_n']}/{len(rows)}) | observed failures={top['observed_failures']} | mean actual Dice={top['mean_actual_dice']:.4f} | actual high-quality rate={top['high_quality_rate']:.3f}")
        print(f"95% Wilson upper bound on accepted failure rate: {top['failure_rate_wilson95_upper']:.3f}")
    else: print("\nNo acceptance gate satisfied constraints. AUTO_ACCEPT disabled.")
    print("\nTHREE-WAY DEVELOPMENT BUCKETS")
    for b in ["AUTO_ACCEPT","REVIEW","ACTIVE_LEARN_PRIORITY"]:
        s=buckets[b]
        print(f"{b:<22} n={s['n']:2d}" if s['n']==0 else f"{b:<22} n={s['n']:2d} | mean Dice={s['mean_actual_dice']:.4f} | failures={s['observed_failures']} | HQ rate={s['high_quality_rate']:.3f}")
    print("\nLowest QC-risk ACTUAL failures:")
    for r in dangerous[:5]: print(f"  fold {r['fold']} {r['case_id']} | actual Dice={r['actual_dice']:.4f} | pred Dice={r['predicted_dice_oof']:.4f} | P(fail)={r['predicted_failure_probability_oof']:.4f} | {r['development_policy_decision']}")
    print("\nOutputs:"); [print(f"  {x}") for x in [risk_path,cand_path,dec_path,meta_path]]
    print("\nNOTE: thresholds are not production-safe until a future locked-set validation."); print("="*100)


if __name__=="__main__": main()
