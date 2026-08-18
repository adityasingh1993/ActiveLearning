#!/usr/bin/env python3
"""Select the Round-3 human annotation batch from all non-accepted pool cases.

Candidate scope
---------------
REVIEW + ACTIVE_LEARN_PRIORITY from run_auto_label_pool_round2_locked.py.
HIGH_CONFIDENCE_PSEUDO_LABEL is excluded from annotation acquisition but remains candidate-only;
it is not automatically promoted to segmentation training.

Risk uses the two QC classifiers that showed useful Round-2 OOF discrimination plus the Dice
regression signal:

    0.40 * P(Dice failure)
  + 0.20 * (1 - predicted Dice)
  + 0.40 * P(overseg > +20%)

P(overseg > +50%) is not used because its OOF discrimination was poor.

Selection combines normalized risk with robust QC-feature diversity. The default 0.75/0.25
risk/diversity blend matches the controlled Round-2 acquisition emphasis while adding the new
volume-risk signal.

Previously selected-but-unlabeled Round-2 IDs are excluded by default so unclear targets are
not repeatedly selected into the next annotation batch.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_active_learning_batch import (
    DIVERSITY_FEATURE_CANDIDATES,
    as_float,
    build_robust_diversity_matrix,
    minmax,
    pairwise_distances,
)

DEFAULT_POOL_DIR = Path("experiments/auto_label_pool_round2_locked_v1")
DEFAULT_ROUND2_AUDIT = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
AUTO_ACCEPT_BUCKET = "HIGH_CONFIDENCE_PSEUDO_LABEL"
ALLOWED_BUCKETS = {"REVIEW", "ACTIVE_LEARN_PRIORITY"}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
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


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def greedy_select_with_risk(case_ids, raw_risk, failure_prob, over20_prob, predicted_dice, X,
                            batch_size, selection_risk_weight):
    n = len(case_ids)
    batch_size = min(int(batch_size), n)
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")

    risk_norm = minmax(raw_risk)
    distances = pairwise_distances(X)

    first = sorted(
        range(n),
        key=lambda i: (
            -raw_risk[i],
            -failure_prob[i],
            -over20_prob[i],
            predicted_dice[i],
            str(case_ids[i]),
        ),
    )[0]

    selected = [first]
    records = [{
        "index": int(first),
        "selection_rank": 1,
        "raw_risk_score": float(raw_risk[first]),
        "normalized_risk_score": float(risk_norm[first]),
        "min_distance_to_selected": float("nan"),
        "normalized_diversity_score": 1.0,
        "combined_selection_score": float(risk_norm[first]),
        "selection_reason": "highest_three_signal_risk_seed",
    }]

    while len(selected) < batch_size:
        remaining = [i for i in range(n) if i not in selected]
        min_dist = np.asarray(
            [float(np.min(distances[i, selected])) for i in remaining], dtype=float
        )
        diversity_norm = minmax(min_dist)
        scored = []
        for pos, i in enumerate(remaining):
            combined = (
                float(selection_risk_weight) * float(risk_norm[i])
                + (1.0 - float(selection_risk_weight)) * float(diversity_norm[pos])
            )
            scored.append((i, combined, float(diversity_norm[pos]), float(min_dist[pos])))

        i, combined, diversity_score, raw_distance = sorted(
            scored,
            key=lambda x: (
                -x[1],
                -raw_risk[x[0]],
                -failure_prob[x[0]],
                -over20_prob[x[0]],
                predicted_dice[x[0]],
                str(case_ids[x[0]]),
            ),
        )[0]
        selected.append(i)
        records.append({
            "index": int(i),
            "selection_rank": len(selected),
            "raw_risk_score": float(raw_risk[i]),
            "normalized_risk_score": float(risk_norm[i]),
            "min_distance_to_selected": raw_distance,
            "normalized_diversity_score": diversity_score,
            "combined_selection_score": combined,
            "selection_reason": "three_signal_risk_plus_diversity",
        })

    return selected, records, risk_norm


def main():
    p = argparse.ArgumentParser(
        description="Select Round-3 annotations from REVIEW + ACTIVE using 3-signal risk + diversity"
    )
    p.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    p.add_argument("--manifest", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--selection-risk-weight", type=float, default=0.75)
    p.add_argument("--failure-risk-weight", type=float, default=0.40)
    p.add_argument("--low-dice-risk-weight", type=float, default=0.20)
    p.add_argument("--overseg20-risk-weight", type=float, default=0.40)
    p.add_argument(
        "--include-previously-unclear",
        action="store_true",
        help="Allow Round-2 selected-but-unlabeled IDs back into acquisition (off by default).",
    )
    p.add_argument("--materialize", action="store_true")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    if not 0.0 <= args.selection_risk_weight <= 1.0:
        p.error("--selection-risk-weight must be in [0,1]")
    risk_weights = np.asarray([
        args.failure_risk_weight,
        args.low_dice_risk_weight,
        args.overseg20_risk_weight,
    ], dtype=float)
    if np.any(risk_weights < 0) or abs(float(risk_weights.sum()) - 1.0) > 1e-8:
        p.error("The three risk weights must be non-negative and sum to 1.0")

    pool_dir = Path(args.pool_dir)
    manifest_path = Path(args.manifest) if args.manifest else pool_dir / "auto_label_manifest.csv"
    feature_path = Path(args.features) if args.features else pool_dir / "unlabeled_qc_features.csv"
    audit_path = Path(args.round2_audit)
    output_dir = Path(args.output_dir) if args.output_dir else pool_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    feature_rows = read_csv(feature_path)
    audit = read_json(audit_path)
    if not manifest_rows or not feature_rows:
        raise RuntimeError("Pool manifest/features are empty")
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")

    feature_by_id = {str(row["case_id"]): row for row in feature_rows}
    if len(feature_by_id) != len(feature_rows):
        raise RuntimeError("Duplicate case IDs in unlabeled_qc_features.csv")

    buckets = set(row.get("qc_bucket", "") for row in manifest_rows)
    allowed_all = ALLOWED_BUCKETS | {AUTO_ACCEPT_BUCKET}
    unknown = sorted(buckets - allowed_all)
    if unknown:
        raise RuntimeError(f"Unexpected QC buckets: {unknown}")

    previous_unclear = set(str(x) for x in audit.get("selected_unlabeled_ids", []))
    candidate_manifest = []
    excluded_unclear = []
    excluded_auto = []
    for row in manifest_rows:
        case_id = str(row["case_id"])
        bucket = row.get("qc_bucket")
        if bucket == AUTO_ACCEPT_BUCKET:
            excluded_auto.append(row)
            continue
        if bucket not in ALLOWED_BUCKETS:
            continue
        if case_id in previous_unclear and not args.include_previously_unclear:
            excluded_unclear.append(row)
            continue
        candidate_manifest.append(row)

    if not candidate_manifest:
        raise RuntimeError("No eligible non-accepted Round-3 candidates remain")

    candidate_rows = []
    for mrow in candidate_manifest:
        case_id = str(mrow["case_id"])
        if case_id not in feature_by_id:
            raise RuntimeError(f"Missing QC feature row for {case_id}")
        row = dict(feature_by_id[case_id])
        row.update({
            "image_path": mrow.get("image_path", row.get("image_path", "")),
            "segmentation_path": mrow.get("segmentation_path", ""),
            "qc_bucket": mrow.get("qc_bucket", ""),
        })
        candidate_rows.append(row)

    case_ids = np.asarray([str(row["case_id"]) for row in candidate_rows], dtype=object)
    predicted_dice = np.asarray([as_float(row.get("predicted_dice")) for row in candidate_rows], dtype=float)
    failure_prob = np.asarray([
        as_float(row.get("predicted_failure_probability")) for row in candidate_rows
    ], dtype=float)
    over20_prob = np.asarray([
        as_float(row.get("predicted_overseg20_probability")) for row in candidate_rows
    ], dtype=float)
    if not (
        np.isfinite(predicted_dice).all()
        and np.isfinite(failure_prob).all()
        and np.isfinite(over20_prob).all()
    ):
        raise RuntimeError("Round-3 candidate QC risk signals contain non-finite values")

    raw_risk = (
        float(args.failure_risk_weight) * failure_prob
        + float(args.low_dice_risk_weight) * (1.0 - predicted_dice)
        + float(args.overseg20_risk_weight) * over20_prob
    )

    X, diversity_features, missing_features, dropped_features = build_robust_diversity_matrix(
        candidate_rows, DIVERSITY_FEATURE_CANDIDATES
    )
    batch_size = min(int(args.batch_size), len(candidate_rows))
    selected, records, risk_norm = greedy_select_with_risk(
        case_ids, raw_risk, failure_prob, over20_prob, predicted_dice, X,
        batch_size, float(args.selection_risk_weight),
    )

    record_by_index = {int(record["index"]): record for record in records}
    selected_set = set(selected)
    risk_only_order = sorted(
        range(len(candidate_rows)),
        key=lambda i: (
            -raw_risk[i], -failure_prob[i], -over20_prob[i], predicted_dice[i], str(case_ids[i])
        ),
    )
    risk_only_rank = {idx: rank + 1 for rank, idx in enumerate(risk_only_order)}

    scored_rows = []
    for i, row in enumerate(candidate_rows):
        rec = record_by_index.get(i)
        scored_rows.append({
            "case_id": str(case_ids[i]),
            "qc_bucket": row["qc_bucket"],
            "image_path": row.get("image_path", ""),
            "segmentation_path": row.get("segmentation_path", ""),
            "predicted_dice": float(predicted_dice[i]),
            "predicted_failure_probability": float(failure_prob[i]),
            "predicted_overseg20_probability": float(over20_prob[i]),
            "raw_three_signal_risk": float(raw_risk[i]),
            "normalized_risk": float(risk_norm[i]),
            "risk_only_rank": int(risk_only_rank[i]),
            "selected": int(i in selected_set),
            "selection_rank": int(rec["selection_rank"]) if rec else "",
            "min_distance_when_selected": rec["min_distance_to_selected"] if rec else "",
            "normalized_diversity_when_selected": rec["normalized_diversity_score"] if rec else "",
            "combined_selection_score": rec["combined_selection_score"] if rec else "",
            "selection_reason": rec["selection_reason"] if rec else "",
        })

    by_id = {row["case_id"]: row for row in scored_rows}
    batch_rows = [by_id[str(case_ids[i])] for i in selected]
    batch_rows.sort(key=lambda row: int(row["selection_rank"]))
    scored_rows.sort(key=lambda row: (
        0 if int(row["selected"]) else 1,
        int(row["selection_rank"]) if row["selection_rank"] != "" else 10**9,
        int(row["risk_only_rank"]),
    ))

    batch_path = output_dir / "active_learning_batch_round3.csv"
    scores_path = output_dir / "active_learning_candidate_scores_round3.csv"
    metadata_path = output_dir / "active_learning_batch_round3_metadata.json"
    write_csv(batch_path, batch_rows)
    write_csv(scores_path, scored_rows)

    materialized_dir = None
    if args.materialize:
        materialized_dir = output_dir / "selected_active_learning_round3"
        if materialized_dir.exists():
            shutil.rmtree(materialized_dir)
        materialized_dir.mkdir(parents=True, exist_ok=True)
        for row in batch_rows:
            src = Path(row["segmentation_path"])
            if not src.exists():
                raise FileNotFoundError(f"Missing selected prediction: {src}")
            shutil.copy2(src, materialized_dir / src.name)

    metadata = {
        "version": "active_learning_round3_three_signal_risk_diversity_v1",
        "manifest": str(manifest_path),
        "features": str(feature_path),
        "round2_audit": str(audit_path),
        "candidate_buckets": sorted(ALLOWED_BUCKETS),
        "excluded_auto_accept_bucket": AUTO_ACCEPT_BUCKET,
        "n_pool_cases": len(manifest_rows),
        "n_auto_accept_excluded": len(excluded_auto),
        "previous_round2_unclear_ids": sorted(previous_unclear),
        "previous_round2_unclear_excluded": sorted(str(x["case_id"]) for x in excluded_unclear),
        "include_previously_unclear": bool(args.include_previously_unclear),
        "n_eligible_nonaccepted_candidates": len(candidate_rows),
        "batch_size": len(batch_rows),
        "risk_definition": (
            f"{args.failure_risk_weight:.2f}*P(failure)+"
            f"{args.low_dice_risk_weight:.2f}*(1-predDice)+"
            f"{args.overseg20_risk_weight:.2f}*P(overseg>20%)"
        ),
        "overseg50_rule": "diagnostic_only_not_used_for_acquisition_risk",
        "selection_risk_weight": float(args.selection_risk_weight),
        "selection_diversity_weight": float(1.0 - args.selection_risk_weight),
        "diversity_features_used": diversity_features,
        "missing_diversity_features": missing_features,
        "dropped_diversity_features": dropped_features,
        "selected_case_ids": [row["case_id"] for row in batch_rows],
        "materialized_dir": str(materialized_dir) if materialized_dir else None,
        "warning": (
            "Round-3 acquisition is a development active-learning decision. HIGH_CONF candidates are excluded "
            "from annotation acquisition but are not automatically added to segmentation training."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 118)
    print("ROUND-3 ACTIVE-LEARNING SELECTION — NON-ACCEPTED RISK + DIVERSITY")
    print(f"Pool cases:                    {len(manifest_rows)}")
    print(f"High-confidence excluded:      {len(excluded_auto)}")
    print(f"Prior unclear excluded:        {len(excluded_unclear)}")
    print(f"Eligible nonaccepted:          {len(candidate_rows)}")
    print(f"Batch size:                    {len(batch_rows)}")
    print(
        f"Risk: {args.failure_risk_weight:.2f} Pfail + {args.low_dice_risk_weight:.2f}(1-Dice) + "
        f"{args.overseg20_risk_weight:.2f} Pover20"
    )
    print(f"Risk/diversity blend:          {args.selection_risk_weight:.2f}/{1-args.selection_risk_weight:.2f}")
    print("Selected:")
    for row in batch_rows:
        print(
            f"  {int(row['selection_rank']):2d}. {row['case_id']} | {row['qc_bucket']:<21} | "
            f"risk={float(row['raw_three_signal_risk']):.4f} | "
            f"Pfail={float(row['predicted_failure_probability']):.4f} | "
            f"Pover20={float(row['predicted_overseg20_probability']):.4f} | "
            f"predDice={float(row['predicted_dice']):.4f}"
        )
    print(f"Batch CSV:                     {batch_path}")
    print(f"Candidate scores:              {scores_path}")
    print(f"Metadata:                      {metadata_path}")
    if materialized_dir:
        print(f"Materialized predictions:      {materialized_dir}")
    print("Next: human-review/annotate the selected Round-3 cases, then provenance-audit before training.")
    print("=" * 118)


if __name__ == "__main__":
    main()
