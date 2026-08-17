#!/usr/bin/env python3
"""Select a Round-2 annotation batch from every non-auto-accepted pool case.

Why this exists
---------------
Round-1 QC has a conservative development AUTO_ACCEPT gate with zero observed accepted
failures, but the auxiliary ACTIVE_LEARN_PRIORITY bucket is based on fixed thresholds
(P(failure)>=0.50 OR predicted Dice<=0.70) and captures only part of the remaining failures.
Therefore acquisition should not discard REVIEW cases.

This selector uses the same deterministic risk + diversity machinery as
scripts/select_active_learning_batch.py, but candidate scope is:

    REVIEW + ACTIVE_LEARN_PRIORITY

HIGH_CONFIDENCE_PSEUDO_LABEL is excluded from annotation acquisition.

Risk:
    0.70 * P(failure) + 0.30 * (1 - predicted Dice)

Selection:
    0.65 * normalized risk + 0.35 * normalized diversity by default.

Inputs are the standard pool runner outputs:
  auto_label_manifest.csv
  unlabeled_qc_features.csv

Outputs:
  active_learning_batch_nonaccepted.csv
  active_learning_candidate_scores_nonaccepted.csv
  active_learning_batch_nonaccepted_metadata.json
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
    greedy_select,
)

AUTO_ACCEPT_BUCKET = "HIGH_CONFIDENCE_PSEUDO_LABEL"
ALLOWED_BUCKETS = {"REVIEW", "ACTIVE_LEARN_PRIORITY"}
DEFAULT_POOL_DIR = Path("experiments/auto_label_pool_round1_raw_v1")


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description="Select risk+diversity Round-2 annotations from REVIEW + ACTIVE_LEARN_PRIORITY"
    )
    p.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    p.add_argument("--manifest", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--selection-risk-weight", type=float, default=0.65)
    p.add_argument("--materialize", action="store_true")
    a = p.parse_args()

    if a.batch_size < 1:
        p.error("--batch-size must be >=1")
    if not 0.0 <= a.selection_risk_weight <= 1.0:
        p.error("--selection-risk-weight must be in [0,1]")

    pool_dir = Path(a.pool_dir)
    manifest_path = Path(a.manifest) if a.manifest else pool_dir / "auto_label_manifest.csv"
    feature_path = Path(a.features) if a.features else pool_dir / "unlabeled_qc_features.csv"
    output_dir = Path(a.output_dir) if a.output_dir else pool_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    feature_rows = read_csv(feature_path)
    if not manifest_rows or not feature_rows:
        raise RuntimeError("Pool manifest/features are empty")

    feature_by_id = {row["case_id"]: row for row in feature_rows}
    if len(feature_by_id) != len(feature_rows):
        raise RuntimeError("Duplicate case IDs in unlabeled_qc_features.csv")

    candidate_manifest = [
        row for row in manifest_rows if row.get("qc_bucket") in ALLOWED_BUCKETS
    ]
    excluded_auto = [
        row for row in manifest_rows if row.get("qc_bucket") == AUTO_ACCEPT_BUCKET
    ]
    unknown_buckets = sorted({
        row.get("qc_bucket", "") for row in manifest_rows
        if row.get("qc_bucket") not in (ALLOWED_BUCKETS | {AUTO_ACCEPT_BUCKET})
    })
    if unknown_buckets:
        raise RuntimeError(f"Unexpected QC buckets in manifest: {unknown_buckets}")
    if not candidate_manifest:
        raise RuntimeError("No REVIEW/ACTIVE_LEARN_PRIORITY cases found")

    candidate_rows = []
    for mrow in candidate_manifest:
        case_id = mrow["case_id"]
        if case_id not in feature_by_id:
            raise RuntimeError(f"Missing QC feature row for candidate {case_id}")
        row = dict(feature_by_id[case_id])
        row.update({
            "image_path": mrow.get("image_path", row.get("image_path", "")),
            "segmentation_path": mrow.get("segmentation_path", ""),
            "qc_bucket": mrow.get("qc_bucket", ""),
            "original_active_learning_rank": mrow.get("active_learning_rank", ""),
        })
        candidate_rows.append(row)

    case_ids = np.asarray([row["case_id"] for row in candidate_rows], dtype=object)
    predicted_dice = np.asarray([as_float(row["predicted_dice"]) for row in candidate_rows], dtype=float)
    failure_prob = np.asarray(
        [as_float(row["predicted_failure_probability"]) for row in candidate_rows], dtype=float
    )
    if not np.isfinite(predicted_dice).all() or not np.isfinite(failure_prob).all():
        raise RuntimeError("Candidate predicted Dice / P(failure) contains non-finite values")

    X, diversity_features, missing_features, dropped_features = build_robust_diversity_matrix(
        candidate_rows, DIVERSITY_FEATURE_CANDIDATES
    )
    batch_size = min(int(a.batch_size), len(candidate_rows))
    selected, records, raw_risk, risk_norm, _ = greedy_select(
        case_ids,
        predicted_dice,
        failure_prob,
        X,
        batch_size,
        float(a.selection_risk_weight),
    )

    selected_set = set(selected)
    record_by_index = {record["index"]: record for record in records}
    risk_only_order = sorted(
        range(len(candidate_rows)),
        key=lambda i: (-raw_risk[i], -failure_prob[i], predicted_dice[i], str(case_ids[i])),
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
            "raw_risk_score": float(raw_risk[i]),
            "normalized_risk_score": float(risk_norm[i]),
            "risk_only_rank": int(risk_only_rank[i]),
            "selected": int(i in selected_set),
            "selection_rank": int(rec["selection_rank"]) if rec else "",
            "min_distance_when_selected": rec["min_distance_to_selected"] if rec else "",
            "normalized_diversity_when_selected": rec["normalized_diversity_score"] if rec else "",
            "combined_selection_score": rec["combined_selection_score"] if rec else "",
            "selection_reason": rec["selection_reason"] if rec else "",
        })

    by_case = {row["case_id"]: row for row in scored_rows}
    batch_rows = [by_case[str(case_ids[i])] for i in selected]
    batch_rows.sort(key=lambda row: int(row["selection_rank"]))
    scored_rows.sort(
        key=lambda row: (
            0 if int(row["selected"]) else 1,
            int(row["selection_rank"]) if row["selection_rank"] != "" else 10**9,
            int(row["risk_only_rank"]),
        )
    )

    batch_path = output_dir / "active_learning_batch_nonaccepted.csv"
    scores_path = output_dir / "active_learning_candidate_scores_nonaccepted.csv"
    metadata_path = output_dir / "active_learning_batch_nonaccepted_metadata.json"
    write_csv(batch_path, batch_rows)
    write_csv(scores_path, scored_rows)

    materialized_dir = None
    if a.materialize:
        materialized_dir = output_dir / "selected_active_learning_nonaccepted"
        if materialized_dir.exists():
            shutil.rmtree(materialized_dir)
        materialized_dir.mkdir(parents=True, exist_ok=True)
        for row in batch_rows:
            src = Path(row["segmentation_path"])
            if not src.exists():
                raise FileNotFoundError(f"Missing selected prediction: {src}")
            shutil.copy2(src, materialized_dir / src.name)

    metadata = {
        "version": "active_learning_nonaccepted_round2_v1",
        "manifest": str(manifest_path),
        "features": str(feature_path),
        "candidate_buckets": sorted(ALLOWED_BUCKETS),
        "excluded_bucket": AUTO_ACCEPT_BUCKET,
        "n_pool_cases": len(manifest_rows),
        "n_auto_accept_excluded": len(excluded_auto),
        "n_nonaccepted_candidates": len(candidate_rows),
        "batch_size": len(batch_rows),
        "risk_definition": "0.70*P(failure)+0.30*(1-predicted_dice)",
        "selection_risk_weight": float(a.selection_risk_weight),
        "diversity_features_used": diversity_features,
        "missing_diversity_features": missing_features,
        "dropped_diversity_features": dropped_features,
        "selected_case_ids": [row["case_id"] for row in batch_rows],
        "materialized_dir": str(materialized_dir) if materialized_dir else None,
        "warning": (
            "Selection uses development-calibrated QC. REVIEW is intentionally included because the fixed "
            "ACTIVE_LEARN_PRIORITY thresholds do not capture every observed OOF failure."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 108)
    print("ROUND-2 ACTIVE-LEARNING SELECTION FROM ALL NON-AUTO-ACCEPTED CASES")
    print(f"Pool cases:               {len(manifest_rows)}")
    print(f"Auto-accept excluded:     {len(excluded_auto)}")
    print(f"Nonaccepted candidates:   {len(candidate_rows)}")
    print(f"Batch size:               {len(batch_rows)}")
    print(f"Risk/diversity blend:     {a.selection_risk_weight:.2f}/{1-a.selection_risk_weight:.2f}")
    print("Selected:")
    for row in batch_rows:
        print(
            f"  {int(row['selection_rank']):2d}. {row['case_id']} | {row['qc_bucket']:<21} | "
            f"risk={float(row['raw_risk_score']):.4f} | P(fail)={float(row['predicted_failure_probability']):.4f} | "
            f"predDice={float(row['predicted_dice']):.4f}"
        )
    print(f"Batch CSV:                {batch_path}")
    print(f"Candidate scores:         {scores_path}")
    print(f"Metadata:                 {metadata_path}")
    if materialized_dir:
        print(f"Materialized predictions: {materialized_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()
