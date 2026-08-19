#!/usr/bin/env python3
"""Select a failure-aware Round-3 HUMAN_GOLD annotation batch from the locked Final62 pool.

This is an acquisition-only workflow. It does NOT train a segmentation model and it does NOT
use external31 ground truth. External31 failure analysis motivated the review categories, but
all selection signals below are available at deployment time from the locked Round-2 pool run.

Input contract
--------------
Consumes outputs from scripts/run_auto_label_pool_round2_locked.py:
  experiments/auto_label_pool_round2_locked_v1/auto_label_manifest.csv
  experiments/auto_label_pool_round2_locked_v1/unlabeled_qc_features.csv

The upstream pool run is intentionally preserved because its Scenario-F gate was calibrated and
validated for the frozen RAW Student+EMA 50/50 prediction at threshold 0.50. HIGH_CONFIDENCE
cases remain candidate-only and are excluded from human acquisition; they are never promoted to
segmentation training automatically by this script.

Failure-aware review pools
--------------------------
1. SMALL_LOCALIZATION_FRAGMENTATION
   Low predicted foreground burden plus fragmented/unstable prediction morphology.
   Suggested human action: REDRAW_OR_LOCALIZE.

2. OVERSEG_FALSE_POSITIVE
   High QC P(over-segmentation >20%) and/or fragmented foreground.
   Suggested human action: REMOVE_FALSE_POSITIVE.

3. LOW_VOLUME_UNCERTAIN
   Low-to-mid predicted foreground with high QC failure/uncertainty and no stronger assignment.
   This is only a deployable UNDER-SEGMENTATION PROXY, not a predicted GT-derived failure class.
   Suggested human action: CHECK_AND_ADD_MISSING_FOREGROUND.

Default quota is 4/3/3. If a proxy pool cannot fill its quota, remaining slots are filled by the
same three-signal risk + robust feature diversity used by the existing Round-3 selector.

No GT-derived quantities (true size, FP/FN, true RVE, GT centroid, true Dice) are used.
"""

import argparse
import csv
import json
import math
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
DEFAULT_ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/round3_failure_aware_v1")

AUTO_ACCEPT_BUCKET = "HIGH_CONFIDENCE_PSEUDO_LABEL"
ALLOWED_BUCKETS = {"REVIEW", "ACTIVE_LEARN_PRIORITY"}

PROFILE_LOCALIZE = "SMALL_LOCALIZATION_FRAGMENTATION"
PROFILE_OVERSEG = "OVERSEG_FALSE_POSITIVE"
PROFILE_LOWVOL = "LOW_VOLUME_UNCERTAIN"
PROFILE_GENERAL = "GENERAL_HIGH_RISK"

ACTION_BY_PROFILE = {
    PROFILE_LOCALIZE: "REDRAW_OR_LOCALIZE",
    PROFILE_OVERSEG: "REMOVE_FALSE_POSITIVE",
    PROFILE_LOWVOL: "CHECK_AND_ADD_MISSING_FOREGROUND",
    PROFILE_GENERAL: "FULL_REVIEW",
}


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_array(rows, key, fallback=None):
    vals = np.asarray([as_float(row.get(key)) for row in rows], dtype=float)
    if np.isfinite(vals).all():
        return vals
    if fallback is not None:
        fb = np.asarray([as_float(row.get(fallback)) for row in rows], dtype=float)
        vals = np.where(np.isfinite(vals), vals, fb)
    if not np.isfinite(vals).all():
        bad = [str(rows[i].get("case_id")) for i in np.where(~np.isfinite(vals))[0]]
        raise RuntimeError(f"Non-finite required feature {key!r} for cases: {bad}")
    return vals


def rank01(values, higher_is_more=True):
    """Stable percentile-style [0,1] rank robust to outliers and constant columns."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return np.ones_like(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    out = ranks / float(len(arr) - 1)
    return out if higher_is_more else 1.0 - out


def select_from_pool(
    eligible,
    already_selected,
    category_score,
    raw_risk,
    X,
    distances,
    quota,
    risk_weight,
    case_ids,
):
    """Greedy category-risk + global diversity selection without duplicate cases."""
    available = [i for i in eligible if i not in already_selected]
    chosen = []
    records = []
    quota = min(int(quota), len(available))
    if quota <= 0:
        return chosen, records

    local_scores = np.asarray([category_score[i] for i in available], dtype=float)
    local_norm = minmax(local_scores)
    local_norm_by_i = {i: float(local_norm[pos]) for pos, i in enumerate(available)}

    while len(chosen) < quota:
        remaining = [i for i in available if i not in chosen]
        scored = []
        for i in remaining:
            if already_selected or chosen:
                anchors = list(already_selected) + list(chosen)
                min_dist = float(np.min(distances[i, anchors]))
            else:
                min_dist = float("nan")
            scored.append((i, min_dist))

        finite_d = np.asarray([x[1] for x in scored], dtype=float)
        if np.isfinite(finite_d).any():
            finite_fill = finite_d.copy()
            fill = float(np.nanmax(finite_fill[np.isfinite(finite_fill)]))
            finite_fill[~np.isfinite(finite_fill)] = fill
            diversity_norm = minmax(finite_fill)
        else:
            diversity_norm = np.ones(len(scored), dtype=float)

        candidates = []
        for pos, (i, min_dist) in enumerate(scored):
            combined = (
                float(risk_weight) * local_norm_by_i[i]
                + (1.0 - float(risk_weight)) * float(diversity_norm[pos])
            )
            candidates.append((
                i,
                combined,
                local_norm_by_i[i],
                float(diversity_norm[pos]),
                min_dist,
            ))

        i, combined, cat_norm, div_norm, min_dist = sorted(
            candidates,
            key=lambda x: (
                -x[1],
                -category_score[x[0]],
                -raw_risk[x[0]],
                str(case_ids[x[0]]),
            ),
        )[0]
        chosen.append(i)
        records.append({
            "index": int(i),
            "category_score": float(category_score[i]),
            "normalized_category_score": float(cat_norm),
            "normalized_diversity_score": float(div_norm),
            "min_distance_to_previous": min_dist,
            "combined_selection_score": float(combined),
        })

    return chosen, records


def materialize_batch(output_dir: Path, batch_rows):
    pack = output_dir / "annotation_pack"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True, exist_ok=True)

    for row in batch_rows:
        rank = int(row["selection_rank"])
        case_id = str(row["case_id"])
        case_dir = pack / f"{rank:02d}_{case_id}"
        image_dir = case_dir / "image"
        preseg_dir = case_dir / "AI_PRESEG"
        image_dir.mkdir(parents=True, exist_ok=True)
        preseg_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(str(row["image_path"]))
        seg_path = Path(str(row["segmentation_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for selected case {case_id}: {image_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing AI preseg for selected case {case_id}: {seg_path}")

        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(seg_path, preseg_dir / seg_path.name)
        provenance = {
            "case_id": case_id,
            "status": "ACTIVE_LEARNING_SELECTED",
            "prediction_status": "AI_PRESEG",
            "human_gold_status": "PENDING",
            "selection_rank": rank,
            "failure_proxy": row["failure_proxy"],
            "suggested_review_action": row["suggested_review_action"],
            "source_prediction": "Final62 locked Student+EMA 50/50 ensemble @ threshold 0.50",
            "warning": "AI_PRESEG is not HUMAN_GOLD. Human verification/correction is required.",
        }
        (case_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return pack


def main():
    p = argparse.ArgumentParser(description="Failure-aware Round-3 annotation acquisition")
    p.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    p.add_argument("--manifest", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--localize-quota", type=int, default=4)
    p.add_argument("--overseg-quota", type=int, default=3)
    p.add_argument("--low-volume-quota", type=int, default=3)
    p.add_argument("--category-risk-weight", type=float, default=0.80)
    p.add_argument("--include-previously-unclear", action="store_true")
    p.add_argument("--materialize", action="store_true")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    if min(args.localize_quota, args.overseg_quota, args.low_volume_quota) < 0:
        p.error("Quotas must be non-negative")
    if not 0.0 <= args.category_risk_weight <= 1.0:
        p.error("--category-risk-weight must be in [0,1]")

    pool_dir = Path(args.pool_dir)
    manifest_path = Path(args.manifest) if args.manifest else pool_dir / "auto_label_manifest.csv"
    features_path = Path(args.features) if args.features else pool_dir / "unlabeled_qc_features.csv"
    audit_path = Path(args.round2_audit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    feature_rows = read_csv(features_path)
    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 HUMAN_GOLD audit is not marked passing")

    feature_by_id = {str(row["case_id"]): row for row in feature_rows}
    if len(feature_by_id) != len(feature_rows):
        raise RuntimeError("Duplicate case IDs in unlabeled_qc_features.csv")

    previous_unclear = set(str(x) for x in audit.get("selected_unlabeled_ids", []))
    candidates = []
    excluded_auto = []
    excluded_previous = []
    for mrow in manifest_rows:
        case_id = str(mrow.get("case_id", ""))
        bucket = str(mrow.get("qc_bucket", ""))
        if bucket == AUTO_ACCEPT_BUCKET:
            excluded_auto.append(case_id)
            continue
        if bucket not in ALLOWED_BUCKETS:
            continue
        if case_id in previous_unclear and not args.include_previously_unclear:
            excluded_previous.append(case_id)
            continue
        if case_id not in feature_by_id:
            raise RuntimeError(f"Missing feature row for eligible case {case_id}")
        row = dict(feature_by_id[case_id])
        row.update({
            "image_path": mrow.get("image_path", row.get("image_path", "")),
            "segmentation_path": mrow.get("segmentation_path", ""),
            "qc_bucket": bucket,
        })
        candidates.append(row)

    if not candidates:
        raise RuntimeError("No eligible REVIEW/ACTIVE_LEARN_PRIORITY candidates remain")

    case_ids = np.asarray([str(r["case_id"]) for r in candidates], dtype=object)
    pred_dice = finite_array(candidates, "predicted_dice")
    p_fail = finite_array(candidates, "predicted_failure_probability")
    p_over20 = finite_array(candidates, "predicted_overseg20_probability")
    pred_fg = finite_array(candidates, "pred_fg_fraction", fallback="pred_fg_fraction_model_space")
    components = finite_array(candidates, "component_count")
    lcf = finite_array(candidates, "largest_component_fraction")
    st_mae = finite_array(candidates, "student_teacher_mae")

    # Optional uncertainty signals. If unavailable, fall back to disagreement so selection
    # remains deployment-shaped and deterministic rather than failing on an older feature file.
    entropy = np.asarray([as_float(r.get("entropy_mean")) for r in candidates], dtype=float)
    if not np.isfinite(entropy).all():
        entropy = st_mae.copy()
    hard_disagree = np.asarray(
        [as_float(r.get("student_teacher_hard_disagreement_fraction")) for r in candidates],
        dtype=float,
    )
    if not np.isfinite(hard_disagree).all():
        hard_disagree = st_mae.copy()

    raw_risk = 0.40 * p_fail + 0.20 * (1.0 - pred_dice) + 0.40 * p_over20

    # Pool-relative predicted size strata are deployment-only. We intentionally avoid any
    # absolute GT-size threshold learned from external31.
    small_cut = float(np.quantile(pred_fg, 1.0 / 3.0))
    mid_cut = float(np.quantile(pred_fg, 2.0 / 3.0))
    is_small = pred_fg <= small_cut
    is_low_mid = pred_fg <= mid_cut

    comp_rank = rank01(np.log1p(np.maximum(components, 0.0)))
    frag_rank = rank01(1.0 - np.clip(lcf, 0.0, 1.0))
    disagree_rank = rank01(st_mae)
    hard_disagree_rank = rank01(hard_disagree)
    uncertainty_rank = rank01(entropy)
    low_volume_rank = rank01(pred_fg, higher_is_more=False)

    fragmentation_score = (
        0.35 * comp_rank
        + 0.30 * frag_rank
        + 0.20 * disagree_rank
        + 0.15 * hard_disagree_rank
    )
    localization_score = (
        0.35 * rank01(raw_risk)
        + 0.30 * fragmentation_score
        + 0.20 * disagree_rank
        + 0.15 * low_volume_rank
    )
    overseg_score = (
        0.55 * p_over20
        + 0.25 * rank01(raw_risk)
        + 0.10 * fragmentation_score
        + 0.10 * disagree_rank
    )
    lowvol_score = (
        0.35 * p_fail
        + 0.25 * (1.0 - pred_dice)
        + 0.20 * uncertainty_rank
        + 0.10 * disagree_rank
        + 0.10 * low_volume_rank
    )

    # Eligibility is deliberately permissive; the score and diversity choose within each proxy.
    localize_eligible = [
        i for i in range(len(candidates))
        if is_small[i]
        and (components[i] >= 2.0 or lcf[i] < 0.90 or fragmentation_score[i] >= 0.50)
    ]
    overseg_threshold = max(0.50, float(np.quantile(p_over20, 0.60)))
    overseg_eligible = [
        i for i in range(len(candidates))
        if p_over20[i] >= overseg_threshold
        or (is_small[i] and components[i] >= 2.0 and p_over20[i] >= float(np.median(p_over20)))
    ]
    fail_threshold = float(np.median(p_fail))
    dice_threshold = float(np.median(pred_dice))
    lowvol_eligible = [
        i for i in range(len(candidates))
        if is_low_mid[i]
        and (p_fail[i] >= fail_threshold or pred_dice[i] <= dice_threshold)
    ]

    X, diversity_features, missing_features, dropped_features = build_robust_diversity_matrix(
        candidates, DIVERSITY_FEATURE_CANDIDATES
    )
    distances = pairwise_distances(X)

    selected = []
    selection_records = []

    def take(profile, eligible, score, quota):
        nonlocal selected, selection_records
        chosen, records = select_from_pool(
            eligible=eligible,
            already_selected=selected,
            category_score=score,
            raw_risk=raw_risk,
            X=X,
            distances=distances,
            quota=quota,
            risk_weight=args.category_risk_weight,
            case_ids=case_ids,
        )
        for rec in records:
            rec["failure_proxy"] = profile
            rec["suggested_review_action"] = ACTION_BY_PROFILE[profile]
            selection_records.append(rec)
        selected.extend(chosen)

    take(PROFILE_LOCALIZE, localize_eligible, localization_score, args.localize_quota)
    take(PROFILE_OVERSEG, overseg_eligible, overseg_score, args.overseg_quota)
    take(PROFILE_LOWVOL, lowvol_eligible, lowvol_score, args.low_volume_quota)

    # Any unfilled requested slots are filled by global three-signal risk + diversity.
    target_n = min(int(args.batch_size), len(candidates))
    if len(selected) < target_n:
        all_indices = list(range(len(candidates)))
        chosen, records = select_from_pool(
            eligible=all_indices,
            already_selected=selected,
            category_score=raw_risk,
            raw_risk=raw_risk,
            X=X,
            distances=distances,
            quota=target_n - len(selected),
            risk_weight=0.75,
            case_ids=case_ids,
        )
        for rec in records:
            rec["failure_proxy"] = PROFILE_GENERAL
            rec["suggested_review_action"] = ACTION_BY_PROFILE[PROFILE_GENERAL]
            selection_records.append(rec)
        selected.extend(chosen)

    # If category quotas sum above batch-size, truncate deterministically in selection order.
    selected = selected[:target_n]
    selected_set = set(selected)
    record_by_idx = {int(r["index"]): r for r in selection_records if int(r["index"]) in selected_set}

    # Preserve actual sequential selection order after truncation.
    rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(selected)}

    scored_rows = []
    for i, row in enumerate(candidates):
        rec = record_by_idx.get(i)
        if rec:
            profile = str(rec["failure_proxy"])
            action = str(rec["suggested_review_action"])
        else:
            # Diagnostic dominant proxy for non-selected candidates; this is not a trained class.
            proxy_scores = {
                PROFILE_LOCALIZE: localization_score[i] if i in localize_eligible else -1.0,
                PROFILE_OVERSEG: overseg_score[i] if i in overseg_eligible else -1.0,
                PROFILE_LOWVOL: lowvol_score[i] if i in lowvol_eligible else -1.0,
                PROFILE_GENERAL: raw_risk[i],
            }
            profile = max(proxy_scores, key=proxy_scores.get)
            action = ACTION_BY_PROFILE[profile]

        size_stratum = "SMALL" if pred_fg[i] <= small_cut else ("MID" if pred_fg[i] <= mid_cut else "LARGE")
        scored_rows.append({
            "case_id": str(case_ids[i]),
            "qc_bucket": row["qc_bucket"],
            "image_path": row.get("image_path", ""),
            "segmentation_path": row.get("segmentation_path", ""),
            "predicted_dice": float(pred_dice[i]),
            "predicted_failure_probability": float(p_fail[i]),
            "predicted_overseg20_probability": float(p_over20[i]),
            "raw_three_signal_risk": float(raw_risk[i]),
            "pred_fg_fraction": float(pred_fg[i]),
            "predicted_size_stratum": size_stratum,
            "component_count": int(round(float(components[i]))),
            "largest_component_fraction": float(lcf[i]),
            "student_teacher_mae": float(st_mae[i]),
            "fragmentation_score": float(fragmentation_score[i]),
            "localization_fragmentation_score": float(localization_score[i]),
            "overseg_proxy_score": float(overseg_score[i]),
            "low_volume_uncertain_score": float(lowvol_score[i]),
            "failure_proxy": profile,
            "suggested_review_action": action,
            "selected": int(i in selected_set),
            "selection_rank": rank_by_idx.get(i, ""),
            "combined_selection_score": rec.get("combined_selection_score", "") if rec else "",
            "normalized_diversity_when_selected": rec.get("normalized_diversity_score", "") if rec else "",
            "provenance_status": "ACTIVE_LEARNING_SELECTED" if i in selected_set else "UNLABELED_POOL",
            "target_label_status": "PENDING_HUMAN_GOLD" if i in selected_set else "UNLABELED",
        })

    by_id = {r["case_id"]: r for r in scored_rows}
    batch_rows = [by_id[str(case_ids[i])] for i in selected]
    batch_rows.sort(key=lambda r: int(r["selection_rank"]))
    scored_rows.sort(key=lambda r: (
        0 if int(r["selected"]) else 1,
        int(r["selection_rank"]) if r["selection_rank"] != "" else 10**9,
        -float(r["raw_three_signal_risk"]),
    ))

    batch_path = output_dir / "active_learning_batch_round3_failure_aware.csv"
    scores_path = output_dir / "active_learning_candidate_scores_round3_failure_aware.csv"
    metadata_path = output_dir / "active_learning_batch_round3_failure_aware_metadata.json"
    write_csv(batch_path, batch_rows)
    write_csv(scores_path, scored_rows)

    pack = materialize_batch(output_dir, batch_rows) if args.materialize else None

    counts = {}
    for row in batch_rows:
        counts[row["failure_proxy"]] = counts.get(row["failure_proxy"], 0) + 1

    metadata = {
        "version": "round3_failure_aware_deployable_proxy_v1",
        "selection_is_training": False,
        "external31_used_for_fitting_or_selection": False,
        "external31_role": "failure-analysis motivation only; no external GT or external metrics are read",
        "pool_manifest": str(manifest_path),
        "pool_features": str(features_path),
        "round2_audit": str(audit_path),
        "source_segmentation_definition": "Final62 locked Student+EMA 50/50 RAW ensemble @ threshold 0.50",
        "source_qc_gate": "locked Scenario-F Round-2 QC",
        "high_confidence_candidates_are_human_gold": False,
        "n_manifest_cases": len(manifest_rows),
        "n_high_confidence_excluded": len(excluded_auto),
        "n_previous_unclear_excluded": len(excluded_previous),
        "previous_unclear_ids_excluded": sorted(excluded_previous),
        "n_eligible_review_active_cases": len(candidates),
        "requested_batch_size": int(args.batch_size),
        "selected_batch_size": len(batch_rows),
        "requested_quotas": {
            PROFILE_LOCALIZE: int(args.localize_quota),
            PROFILE_OVERSEG: int(args.overseg_quota),
            PROFILE_LOWVOL: int(args.low_volume_quota),
        },
        "selected_proxy_counts": counts,
        "predicted_size_thresholds_pool_relative": {
            "small_max_pred_fg_fraction": small_cut,
            "mid_max_pred_fg_fraction": mid_cut,
        },
        "overseg_eligibility_probability_threshold": overseg_threshold,
        "risk_definition": "0.40*P(failure)+0.20*(1-predDice)+0.40*P(overseg>20%)",
        "selection_rule": "failure-proxy quota -> proxy risk + robust QC-feature diversity -> global risk/diversity fill",
        "diversity_features_used": diversity_features,
        "missing_diversity_features": missing_features,
        "dropped_diversity_features": dropped_features,
        "selected_case_ids": [r["case_id"] for r in batch_rows],
        "annotation_pack": str(pack) if pack else None,
        "provenance_rule": "AI_PRESEG and ACTIVE_LEARNING_SELECTED stay distinct from HUMAN_GOLD until human correction",
        "warning": (
            "Failure proxy names are acquisition/review heuristics based only on deployable prediction signals. "
            "They are not GT-derived failure labels and are not a trained multiclass classifier."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("ROUND-3 FAILURE-AWARE ACTIVE-LEARNING BATCH")
    print("=" * 124)
    print(f"Eligible REVIEW/ACTIVE cases: {len(candidates)}")
    print(f"HIGH_CONF excluded:           {len(excluded_auto)}")
    print(f"Previous unclear excluded:    {len(excluded_previous)}")
    print(f"Pool size cuts: SMALL <= {small_cut:.6f} | MID <= {mid_cut:.6f} predicted FG fraction")
    print(f"Selected:                     {len(batch_rows)}")
    print(f"Proxy counts:                 {counts}")
    print("\nrank  case_id              QC bucket                size   proxy                            action")
    print("-" * 124)
    for row in batch_rows:
        print(
            f"{int(row['selection_rank']):>4}  {row['case_id']:<20} "
            f"{row['qc_bucket']:<24} {row['predicted_size_stratum']:<6} "
            f"{row['failure_proxy']:<32} {row['suggested_review_action']}"
        )
    print(f"\nBatch CSV:     {batch_path}")
    print(f"Candidate CSV: {scores_path}")
    print(f"Metadata:      {metadata_path}")
    if pack:
        print(f"Annotation pack: {pack}")
    print("No selected AI preseg is HUMAN_GOLD until human verification/correction.")
    print("=" * 124)


if __name__ == "__main__":
    main()
