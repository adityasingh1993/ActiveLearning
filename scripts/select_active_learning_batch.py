#!/usr/bin/env python3
"""Select a risk-aware, diversity-aware annotation batch from ACTIVE_LEARN_PRIORITY.

This consumes the outputs of scripts/run_auto_label_pool.py and chooses a first human
annotation batch from the ACTIVE_LEARN_PRIORITY bucket. The selector intentionally does
not simply take the top-N failure probabilities because those cases may all represent the
same failure mode.

Selection strategy
------------------
1. Build a curated deployment-only diversity vector from QC features covering uncertainty,
   student/teacher disagreement, predicted volume/shape/location, boundary confidence,
   and image / predicted-region intensity.
2. Median-impute missing/non-finite values.
3. Robust-scale each feature by median/IQR (falling back to standard deviation); constant
   features are removed.
4. Compute a risk score = 0.70 * P(failure) + 0.30 * (1 - predicted Dice).
5. Seed selection with the single highest-risk case.
6. Greedily add cases using:

       selection_score = selection_risk_weight * normalized_risk
                       + (1-selection_risk_weight) * normalized_min_distance_to_selected

   The default selection_risk_weight=0.65 keeps risk dominant while forcing coverage of
   different QC failure modes.

This is QC-feature diversity v1. It is deliberately simple and deterministic. A later
experiment may compare this against encoder/CoreSet embeddings, but that should be a
controlled comparison rather than silently changing the selection rule.

Default inputs:
  experiments/auto_label_pool_v1/auto_label_manifest.csv
  experiments/auto_label_pool_v1/unlabeled_qc_features.csv

Default outputs:
  experiments/auto_label_pool_v1/active_learning_batch.csv
  experiments/auto_label_pool_v1/active_learning_candidate_scores.csv
  experiments/auto_label_pool_v1/active_learning_batch_metadata.json
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

DEFAULT_POOL_DIR = Path("experiments/auto_label_pool_v1")
DEFAULT_BUCKET = "ACTIVE_LEARN_PRIORITY"

# Curated features intentionally exclude predicted Dice and P(failure): those are handled
# separately as risk. The diversity vector should describe *how* cases differ, not merely
# duplicate the risk ordering.
DIVERSITY_FEATURE_CANDIDATES = [
    # predicted foreground burden / confidence / uncertainty
    "pred_fg_fraction",
    "confidence_mean",
    "confidence_p10",
    "entropy_mean",
    "entropy_p95",
    "uncertain_fraction_margin_0p05",
    "uncertain_fraction_margin_0p10",
    "uncertain_fraction_margin_0p20",
    "pred_region_prob_mean",
    "pred_region_prob_std",
    "pred_region_prob_p10",
    # student / teacher disagreement
    "student_teacher_mae",
    "student_teacher_p95_absdiff",
    "student_teacher_p99_absdiff",
    "student_teacher_hard_disagreement_fraction",
    "student_teacher_fg_fraction_absdiff",
    # connected components / morphology
    "component_count",
    "components_ge_100",
    "components_ge_1000",
    "largest_component_fraction",
    "second_component_fraction",
    # predicted location
    "centroid_d_norm",
    "centroid_h_norm",
    "centroid_w_norm",
    # predicted extent / compactness
    "bbox_d_fraction",
    "bbox_h_fraction",
    "bbox_w_fraction",
    "bbox_volume_fraction",
    "bbox_occupancy",
    "boundary_vox_fraction",
    "boundary_confidence_mean",
    "boundary_entropy_mean",
    # whole-volume intensity / acquisition/domain proxies
    "image_mean",
    "image_std",
    "image_p10",
    "image_p50",
    "image_p90",
    "image_p99",
    # predicted-region appearance
    "pred_region_image_mean",
    "pred_region_image_std",
    "pred_region_image_p10",
    "pred_region_image_p50",
    "pred_region_image_p90",
    "pred_region_image_p99",
    "pred_region_minus_background_mean",
]


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


def as_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def minmax(values):
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def build_robust_diversity_matrix(rows, feature_candidates):
    if not rows:
        raise RuntimeError("No candidate rows supplied")

    available = set(rows[0].keys())
    requested = [name for name in feature_candidates if name in available]
    missing = [name for name in feature_candidates if name not in available]
    if not requested:
        raise RuntimeError("None of the curated diversity features exist in unlabeled_qc_features.csv")

    raw = np.asarray(
        [[as_float(row.get(name)) for name in requested] for row in rows],
        dtype=float,
    )

    # Median imputation, then robust scaling. All-NaN / constant columns are removed.
    kept_names = []
    scaled_cols = []
    dropped = []
    for col_idx, name in enumerate(requested):
        col = raw[:, col_idx].astype(float)
        finite = np.isfinite(col)
        if not finite.any():
            dropped.append({"feature": name, "reason": "all_non_finite"})
            continue

        median = float(np.median(col[finite]))
        col[~finite] = median
        q25, q75 = np.percentile(col, [25, 75])
        scale = float(q75 - q25)
        scale_source = "iqr"
        if scale <= 1e-12:
            scale = float(np.std(col))
            scale_source = "std"
        if scale <= 1e-12:
            dropped.append({"feature": name, "reason": "constant"})
            continue

        z = (col - median) / scale
        # A single extreme morphology feature should not dominate the whole Euclidean space.
        z = np.clip(z, -5.0, 5.0)
        kept_names.append(name)
        scaled_cols.append(z)

    if not scaled_cols:
        raise RuntimeError("All diversity features became unusable after robust scaling")

    matrix = np.stack(scaled_cols, axis=1)
    return matrix, kept_names, missing, dropped


def pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def greedy_select(case_ids, predicted_dice, failure_prob, X, batch_size, selection_risk_weight):
    n = len(case_ids)
    batch_size = min(int(batch_size), n)
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")

    # Risk stays interpretable and bounded in [0,1].
    raw_risk = 0.70 * failure_prob + 0.30 * (1.0 - predicted_dice)
    risk_norm = minmax(raw_risk)
    distances = pairwise_distances(X)

    # Deterministic seed: highest raw risk, then higher P(fail), lower Dice, case ID.
    first = sorted(
        range(n),
        key=lambda i: (-raw_risk[i], -failure_prob[i], predicted_dice[i], str(case_ids[i])),
    )[0]

    selected = [first]
    selection_records = [{
        "index": int(first),
        "selection_rank": 1,
        "raw_risk_score": float(raw_risk[first]),
        "normalized_risk_score": float(risk_norm[first]),
        "min_distance_to_selected": float("nan"),
        "normalized_diversity_score": 1.0,
        "combined_selection_score": float(risk_norm[first]),
        "selection_reason": "highest_risk_seed",
    }]

    while len(selected) < batch_size:
        remaining = [i for i in range(n) if i not in selected]
        min_dist = np.asarray(
            [float(np.min(distances[i, selected])) for i in remaining],
            dtype=float,
        )
        diversity_norm = minmax(min_dist)

        scored = []
        for pos, i in enumerate(remaining):
            combined = (
                selection_risk_weight * float(risk_norm[i])
                + (1.0 - selection_risk_weight) * float(diversity_norm[pos])
            )
            scored.append((
                i,
                combined,
                float(diversity_norm[pos]),
                float(min_dist[pos]),
            ))

        i, combined, diversity_score, raw_distance = sorted(
            scored,
            key=lambda x: (
                -x[1],
                -raw_risk[x[0]],
                -x[2],
                predicted_dice[x[0]],
                str(case_ids[x[0]]),
            ),
        )[0]
        selected.append(i)
        selection_records.append({
            "index": int(i),
            "selection_rank": len(selected),
            "raw_risk_score": float(raw_risk[i]),
            "normalized_risk_score": float(risk_norm[i]),
            "min_distance_to_selected": raw_distance,
            "normalized_diversity_score": diversity_score,
            "combined_selection_score": combined,
            "selection_reason": "risk_plus_diversity",
        })

    return selected, selection_records, raw_risk, risk_norm, distances


def main():
    parser = argparse.ArgumentParser(
        description="Select a risk-aware + diversity-aware active-learning annotation batch"
    )
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--selection-risk-weight",
        type=float,
        default=0.65,
        help="Blend weight for normalized risk; remaining weight is diversity (default 0.65).",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Copy selected .seg.nrrd predictions into selected_active_learning/ for annotation handoff.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >=1")
    if not 0.0 <= args.selection_risk_weight <= 1.0:
        parser.error("--selection-risk-weight must be in [0,1]")

    pool_dir = Path(args.pool_dir)
    manifest_path = Path(args.manifest) if args.manifest else pool_dir / "auto_label_manifest.csv"
    feature_path = Path(args.features) if args.features else pool_dir / "unlabeled_qc_features.csv"
    output_dir = Path(args.output_dir) if args.output_dir else pool_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    feature_rows = read_csv(feature_path)
    if not manifest_rows or not feature_rows:
        raise RuntimeError("Pool manifest/features are empty")

    feature_by_id = {row["case_id"]: row for row in feature_rows}
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}
    if len(feature_by_id) != len(feature_rows):
        raise RuntimeError("Duplicate case IDs in unlabeled_qc_features.csv")
    if len(manifest_by_id) != len(manifest_rows):
        raise RuntimeError("Duplicate case IDs in auto_label_manifest.csv")

    candidate_manifest = [row for row in manifest_rows if row.get("qc_bucket") == args.bucket]
    if not candidate_manifest:
        raise RuntimeError(f"No cases found in QC bucket {args.bucket!r}")

    candidate_rows = []
    for mrow in candidate_manifest:
        case_id = mrow["case_id"]
        if case_id not in feature_by_id:
            raise RuntimeError(f"Missing QC feature row for candidate {case_id}")
        row = dict(feature_by_id[case_id])
        row.update({
            "image_path": mrow.get("image_path", row.get("image_path", "")),
            "segmentation_path": mrow.get("segmentation_path", ""),
            "active_learning_rank": mrow.get("active_learning_rank", ""),
            "qc_bucket": mrow.get("qc_bucket", ""),
        })
        candidate_rows.append(row)

    case_ids = np.asarray([row["case_id"] for row in candidate_rows], dtype=object)
    predicted_dice = np.asarray([as_float(row["predicted_dice"]) for row in candidate_rows], dtype=float)
    failure_prob = np.asarray(
        [as_float(row["predicted_failure_probability"]) for row in candidate_rows], dtype=float
    )
    if not np.isfinite(predicted_dice).all() or not np.isfinite(failure_prob).all():
        raise RuntimeError("Candidate predicted Dice / failure probability contains non-finite values")

    X, diversity_features, missing_features, dropped_features = build_robust_diversity_matrix(
        candidate_rows, DIVERSITY_FEATURE_CANDIDATES
    )
    batch_size = min(args.batch_size, len(candidate_rows))
    selected, records, raw_risk, risk_norm, distances = greedy_select(
        case_ids,
        predicted_dice,
        failure_prob,
        X,
        batch_size,
        args.selection_risk_weight,
    )

    record_by_index = {record["index"]: record for record in records}
    selected_set = set(selected)

    # Risk-only ranking is included so the user can see which cases diversity moved in/out.
    risk_only_order = sorted(
        range(len(candidate_rows)),
        key=lambda i: (-raw_risk[i], -failure_prob[i], predicted_dice[i], str(case_ids[i])),
    )
    risk_only_rank = {idx: rank + 1 for rank, idx in enumerate(risk_only_order)}

    candidate_output = []
    for i, row in enumerate(candidate_rows):
        selection = record_by_index.get(i)
        candidate_output.append({
            "case_id": str(case_ids[i]),
            "image_path": row.get("image_path", ""),
            "segmentation_path": row.get("segmentation_path", ""),
            "predicted_dice": float(predicted_dice[i]),
            "predicted_failure_probability": float(failure_prob[i]),
            "original_active_learning_rank": row.get("active_learning_rank", ""),
            "risk_only_rank": int(risk_only_rank[i]),
            "raw_risk_score": float(raw_risk[i]),
            "normalized_risk_score": float(risk_norm[i]),
            "selected": int(i in selected_set),
            "selection_rank": int(selection["selection_rank"]) if selection else "",
            "min_distance_when_selected": (
                selection["min_distance_to_selected"] if selection else ""
            ),
            "normalized_diversity_when_selected": (
                selection["normalized_diversity_score"] if selection else ""
            ),
            "combined_selection_score": (
                selection["combined_selection_score"] if selection else ""
            ),
            "selection_reason": selection["selection_reason"] if selection else "",
        })

    batch_rows = [next(row for row in candidate_output if row["case_id"] == str(case_ids[i])) for i in selected]
    batch_rows.sort(key=lambda row: int(row["selection_rank"]))
    candidate_output.sort(key=lambda row: (0 if row["selected"] else 1, row["selection_rank"] or 10**9, row["risk_only_rank"]))

    batch_path = output_dir / "active_learning_batch.csv"
    candidate_path = output_dir / "active_learning_candidate_scores.csv"
    metadata_path = output_dir / "active_learning_batch_metadata.json"
    write_csv(batch_path, batch_rows)
    write_csv(candidate_path, candidate_output)

    materialized_dir = output_dir / "selected_active_learning"
    copied = []
    if args.materialize:
        if materialized_dir.exists():
            shutil.rmtree(materialized_dir)
        materialized_dir.mkdir(parents=True, exist_ok=True)
        for row in batch_rows:
            src = Path(row["segmentation_path"])
            if not src.exists():
                raise FileNotFoundError(
                    f"Cannot materialize selection; missing segmentation for {row['case_id']}: {src}"
                )
            dst = materialized_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    selected_risks = np.asarray([float(row["raw_risk_score"]) for row in batch_rows])
    top_risk_only = set(str(case_ids[i]) for i in risk_only_order[:batch_size])
    selected_ids = [row["case_id"] for row in batch_rows]
    overlap = len(set(selected_ids) & top_risk_only)

    metadata = {
        "version": "active_learning_qc_diversity_v1",
        "pool_manifest": str(manifest_path),
        "pool_features": str(feature_path),
        "candidate_bucket": args.bucket,
        "n_candidates": len(candidate_rows),
        "batch_size": batch_size,
        "selection_risk_weight": float(args.selection_risk_weight),
        "selection_diversity_weight": float(1.0 - args.selection_risk_weight),
        "risk_formula": "0.70 * P(failure) + 0.30 * (1 - predicted_dice)",
        "diversity_method": (
            "median imputation + robust median/IQR scaling (std fallback) + clipped Euclidean "
            "distance + greedy minimum-distance-to-selected"
        ),
        "diversity_features_used": diversity_features,
        "diversity_features_missing_from_input": missing_features,
        "diversity_features_dropped": dropped_features,
        "selected_case_ids": selected_ids,
        "risk_only_top_n_case_ids": [str(case_ids[i]) for i in risk_only_order[:batch_size]],
        "selected_vs_risk_only_overlap": overlap,
        "selected_raw_risk_mean": float(np.mean(selected_risks)),
        "selected_raw_risk_min": float(np.min(selected_risks)),
        "selected_raw_risk_max": float(np.max(selected_risks)),
        "materialized": bool(args.materialize),
        "materialized_files": copied,
        "warning": (
            "This is a deterministic development active-learning selector. QC-feature diversity "
            "should be compared with encoder/CoreSet diversity in a later controlled experiment if needed."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 100)
    print("DIVERSITY-AWARE ACTIVE LEARNING SELECTION")
    print(f"Candidate bucket:       {args.bucket}")
    print(f"Candidates:             {len(candidate_rows)}")
    print(f"Selected:               {batch_size}")
    print(f"Risk/diversity weights: {args.selection_risk_weight:.2f} / {1.0-args.selection_risk_weight:.2f}")
    print(f"Diversity features:     {len(diversity_features)} usable")
    print(f"Overlap with risk-only top-{batch_size}: {overlap}/{batch_size}")
    print("-" * 100)
    for row in batch_rows:
        print(
            f"#{int(row['selection_rank']):02d} {row['case_id']} | "
            f"pred Dice={float(row['predicted_dice']):.4f} | "
            f"P(fail)={float(row['predicted_failure_probability']):.4f} | "
            f"risk-only rank={int(row['risk_only_rank']):02d} | "
            f"selection={float(row['combined_selection_score']):.4f}"
        )
    print("-" * 100)
    print(f"Batch manifest:    {batch_path}")
    print(f"Candidate scores:  {candidate_path}")
    print(f"Metadata:          {metadata_path}")
    if args.materialize:
        print(f"Selected masks:    {materialized_dir}")
    print("\nNext: human-review/correct this batch, return the corrected labels, then train Round 1.")
    print("=" * 100)


if __name__ == "__main__":
    main()
