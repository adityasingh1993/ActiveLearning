#!/usr/bin/env python3
"""Round-5 active-learning triage using the Final82 A3 five-fold committee.

This resumes HUMAN_GOLD acquisition after the successful Final72-A3 -> Final82-A3 internal
comparison. It does not train a model and it does not use external31 labels/metrics for ranking.

Committee
---------
Five frozen Final82-A3 fold checkpoints from:
    experiments/round4_cv_82_a3/checkpoints/fold_{0..4}/best_checkpoint.pth

Each fold contributes the locked raw Student+EMA 50/50 ensemble at threshold 0.50. Deployment-only
prediction features are aggregated across the five folds and used for acquisition ranking.

Default 10-case acquisition mix
-------------------------------
- 3 SMALL_UNCERTAIN
- 4 LOCALIZATION_UNSTABLE
- 3 BOUNDARY_OVERSEG_PROXY

The shift from Round4's 4/3/3 mix intentionally spends one additional annotation slot on
localization instability while retaining small-target and false-positive/boundary coverage.
Any unfilled quota is filled by general committee risk + robust diversity.

Safeguards
----------
- Requires the passing live Final82 pretraining audit created by the Final82 CV runner.
- Excludes all 82 current HUMAN_GOLD IDs.
- Excludes frozen external31 IDs; only IDs are read, never GT or external metrics for scoring.
- Excludes previously skipped Round2 IDs by default.
- Preserves old locked Final62 Scenario-F HIGH_CONFIDENCE status only as KEEP_CANDIDATE;
  Final82 features are not fed through that old QC calibration.
- No AI prediction is called HUMAN_GOLD.

Example
-------
python scripts/run_round5_active_learning_final82_a3_committee.py \
  --config config_resize128.yaml \
  --gpu 0 \
  --materialize
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path


def _consume_option(argv, name):
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


GPU, CLEAN_ARGV = _consume_option(sys.argv, "--gpu")
if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.argv = CLEAN_ARGV

import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.build_oof_qc_dataset import extract_features, load_models  # noqa: E402
from scripts.run_auto_label_pool import collect_pool_cases  # noqa: E402
from scripts.select_active_learning_batch import (  # noqa: E402
    build_robust_diversity_matrix,
    minmax,
    pairwise_distances,
)
from scripts.run_round4_active_learning_a3_committee import (  # noqa: E402
    aggregate_case_features,
    as_float,
    choose_from_pool,
    finite_or,
    rank01,
)

FINAL82_A3_DIR = Path("experiments/round4_cv_82_a3")
FINAL82_AUDIT = FINAL82_A3_DIR / "final82_live_label_audit.json"
ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
LOCKED_POOL = Path("experiments/auto_label_pool_round2_locked_v1/auto_label_manifest.csv")
EXTERNAL_RESULTS = Path(
    "experiments/external31_round2_qc_gate_v1/external31_locked_gate_case_results.csv"
)
OUTPUT_DIR = Path("experiments/round5_active_final82_a3_committee_v1")

KEEP_BUCKET = "HIGH_CONFIDENCE_PSEUDO_LABEL"
PROFILE_SMALL = "SMALL_UNCERTAIN"
PROFILE_LOCALIZE = "LOCALIZATION_UNSTABLE"
PROFILE_BOUNDARY = "BOUNDARY_OVERSEG_PROXY"
PROFILE_GENERAL = "GENERAL_COMMITTEE_RISK"

ACTION_BY_PROFILE = {
    PROFILE_SMALL: "CHECK_SMALL_OR_MISSED_BLADDER",
    PROFILE_LOCALIZE: "REDRAW_OR_LOCALIZE",
    PROFILE_BOUNDARY: "CHECK_FALSE_POSITIVE_AND_BOUNDARY",
    PROFILE_GENERAL: "FULL_REVIEW",
}

DIVERSITY_FEATURES = [
    "mean_pred_fg_fraction",
    "mean_entropy_mean",
    "mean_entropy_p95",
    "mean_uncertain_fraction_margin_0p10",
    "mean_student_teacher_mae",
    "mean_student_teacher_hard_disagreement_fraction",
    "mean_component_count",
    "mean_largest_component_fraction",
    "mean_centroid_d_norm",
    "mean_centroid_h_norm",
    "mean_centroid_w_norm",
    "mean_bbox_volume_fraction",
    "mean_bbox_occupancy",
    "mean_boundary_entropy_mean",
    "mean_image_mean",
    "mean_image_std",
    "mean_pred_region_image_mean",
    "mean_pred_region_minus_background_mean",
    "std_pred_fg_fraction",
    "std_component_count",
    "centroid_spread",
]


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def materialize_images(output_dir: Path, selected_rows):
    pack = output_dir / "annotation_pack"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        rank = int(row["selection_rank"])
        case_id = str(row["case_id"])
        case_dir = pack / f"{rank:02d}_{case_id}"
        image_dir = case_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        src = Path(str(row["image_path"]))
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, image_dir / src.name)
        provenance = {
            "case_id": case_id,
            "selection_rank": rank,
            "round": 5,
            "selection_model": "Final82 A3 five-fold committee",
            "selection_profile": row["selection_profile"],
            "suggested_review_action": row["suggested_review_action"],
            "human_gold_status": "PENDING",
            "prediction_status": "NO_PRESEG_MATERIALIZED_YET",
            "warning": "Selection score is not a GT-derived quality label. Human annotation/review is required.",
        }
        (case_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return pack


def main():
    p = argparse.ArgumentParser(description="Round5 active-learning selection with Final82 A3 committee")
    p.add_argument("--config", required=True)
    p.add_argument("--input-dir", default=None, help="Default: config.data_dir")
    p.add_argument("--final82-a3-dir", default=str(FINAL82_A3_DIR))
    p.add_argument("--final82-audit", default=str(FINAL82_AUDIT))
    p.add_argument("--round2-audit", default=str(ROUND2_AUDIT))
    p.add_argument("--locked-pool-manifest", default=str(LOCKED_POOL))
    p.add_argument("--external-case-results", default=str(EXTERNAL_RESULTS))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--small-quota", type=int, default=3)
    p.add_argument("--localization-quota", type=int, default=4)
    p.add_argument("--boundary-quota", type=int, default=3)
    p.add_argument("--selection-risk-weight", type=float, default=0.80)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--include-previously-skipped", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Smoke test only; selection on first N eligible cases")
    p.add_argument("--materialize", action="store_true", help="Copy selected images into annotation pack")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    if min(args.small_quota, args.localization_quota, args.boundary_quota) < 0:
        p.error("Quotas must be non-negative")
    if args.small_quota + args.localization_quota + args.boundary_quota > args.batch_size:
        p.error("Profile quotas cannot sum to more than --batch-size")
    if not 0.0 <= args.selection_risk_weight <= 1.0:
        p.error("--selection-risk-weight must be in [0,1]")
    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Round5 Final82-A3 committee screening is locked to threshold 0.50")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}. Use --overwrite intentionally.")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Round5 requires prototype DynUNet Student+EMA mode")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    final82_audit = read_json(Path(args.final82_audit))
    if not final82_audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final82 live label audit is not passing")
    human_ids = set(str(x) for x in final82_audit.get("all_current_human_label_ids", []))
    if len(human_ids) != 82:
        raise RuntimeError(f"Expected exactly 82 audited HUMAN_GOLD IDs, found {len(human_ids)}")

    external_rows = read_csv(Path(args.external_case_results))
    external_ids = {str(r.get("case_id", "")).strip() for r in external_rows}
    external_ids.discard("")
    if len(external_ids) != 31:
        raise RuntimeError(f"Expected 31 frozen external IDs, found {len(external_ids)}")
    if human_ids & external_ids:
        raise RuntimeError("Final82 HUMAN_GOLD overlaps frozen external31")

    skipped_ids = set()
    round2_audit_path = Path(args.round2_audit)
    if round2_audit_path.exists():
        r2audit = read_json(round2_audit_path)
        skipped_ids = set(str(x) for x in r2audit.get("selected_unlabeled_ids", []))
        skipped_ids -= human_ids
    excluded_skipped = set() if args.include_previously_skipped else skipped_ids

    locked_manifest_rows = read_csv(Path(args.locked_pool_manifest))
    locked_by_id = {str(r.get("case_id", "")): r for r in locked_manifest_rows}
    if len(locked_by_id) != len(locked_manifest_rows):
        raise RuntimeError("Duplicate IDs in locked Final62 pool manifest")

    input_dir = Path(args.input_dir) if args.input_dir else Path(config.data_dir)
    excluded = human_ids | external_ids | excluded_skipped
    cases, explicit_human = collect_pool_cases(config, input_dir, excluded, limit=args.limit)
    if not cases:
        raise RuntimeError("No eligible unlabeled pool cases remain")
    case_path = {str(c["id"]): str(c["image"]) for c in cases}

    model_dir = Path(args.final82_a3_dir)
    checkpoints = []
    for fold in range(5):
        ckpt = model_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing Final82 A3 Fold{fold} checkpoint: {ckpt}")
        checkpoints.append(ckpt)

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    fold_feature_rows = []
    per_case = {str(c["id"]): [] for c in cases}

    print("=" * 124)
    print("ROUND 5 ACTIVE LEARNING — FINAL82 A3 FIVE-FOLD COMMITTEE")
    print(f"Input dir:                   {input_dir}")
    print(f"Current HUMAN_GOLD excluded: {len(human_ids)}")
    print(f"External31 excluded:         {len(external_ids)}")
    print(f"Explicit labels excluded:    {len(explicit_human)}")
    print(f"Previously skipped excluded: {len(excluded_skipped)}")
    print(f"Eligible pool:               {len(cases)}")
    print(f"Device:                      {device}")
    print("Default quotas:              SMALL=3 | LOCALIZATION=4 | BOUNDARY=3")
    print("KEEP status remains old locked Scenario-F only; Final82 committee is acquisition-only.")
    print("=" * 124)

    for fold, checkpoint in enumerate(checkpoints):
        print(f"\n[Final82 A3 committee] Fold {fold}/4: {checkpoint}")
        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"Final82 Fold{fold} checkpoint has no EMA teacher")
        loader = DataLoader(Dataset(cases, transform=transform), batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for idx, batch in enumerate(loader, start=1):
                raw_id = batch.get("id")
                case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
                image_t = batch["image"].to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    s_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                    t_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    e_prob_t = 0.5 * (s_prob_t + t_prob_t)
                image_np = image_t[0, 0].detach().float().cpu().numpy()
                s_prob = s_prob_t[0, 0].detach().float().cpu().numpy()
                t_prob = t_prob_t[0, 0].detach().float().cpu().numpy()
                e_prob = e_prob_t[0, 0].detach().float().cpu().numpy()
                features, _ = extract_features(image_np, s_prob, t_prob, e_prob, threshold=args.threshold)
                row = {"case_id": case_id, "fold": fold, "image_path": case_path[case_id]}
                row.update(features)
                fold_feature_rows.append(row)
                per_case[case_id].append(row)
                if idx % 50 == 0 or idx == len(cases):
                    print(f"  scored {idx}/{len(cases)}")
        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "committee_fold_features.csv", fold_feature_rows)

    aggregated = [
        aggregate_case_features(case_id, per_case[case_id], case_path[case_id])
        for case_id in sorted(per_case)
    ]

    for row in aggregated:
        locked = locked_by_id.get(str(row["case_id"]))
        row["locked_final62_qc_bucket"] = str(locked.get("qc_bucket", "")) if locked else ""
        row["locked_keep_candidate"] = int(bool(locked) and str(locked.get("qc_bucket")) == KEEP_BUCKET)

    acquisition = [r for r in aggregated if int(r["locked_keep_candidate"]) == 0]
    if not acquisition:
        raise RuntimeError("No non-KEEP acquisition candidates remain")

    case_ids = np.asarray([str(r["case_id"]) for r in acquisition], dtype=object)
    pred_fg = finite_or([as_float(r.get("mean_pred_fg_fraction")) for r in acquisition], 0.0)
    entropy = finite_or([as_float(r.get("mean_entropy_mean")) for r in acquisition], 1.0)
    st_mae = finite_or([as_float(r.get("mean_student_teacher_mae")) for r in acquisition], 1.0)
    hard_dis = finite_or(
        [as_float(r.get("mean_student_teacher_hard_disagreement_fraction")) for r in acquisition], 1.0
    )
    fg_std = finite_or([as_float(r.get("std_pred_fg_fraction")) for r in acquisition], 1.0)
    centroid_spread = finite_or([as_float(r.get("centroid_spread")) for r in acquisition], 1.0)
    component_mean = finite_or([as_float(r.get("mean_component_count")) for r in acquisition], 0.0)
    component_std = finite_or([as_float(r.get("std_component_count")) for r in acquisition], 0.0)
    lcf = finite_or([as_float(r.get("mean_largest_component_fraction")) for r in acquisition], 0.0)
    uncertain = finite_or(
        [as_float(r.get("mean_uncertain_fraction_margin_0p10")) for r in acquisition], 1.0
    )
    boundary_entropy = np.asarray(
        [as_float(r.get("mean_boundary_entropy_mean")) for r in acquisition], dtype=float
    )
    boundary_entropy = np.where(np.isfinite(boundary_entropy), boundary_entropy, entropy)

    fragmentation = 0.50 * rank01(np.log1p(np.maximum(component_mean, 0.0))) + 0.50 * rank01(
        1.0 - np.clip(lcf, 0.0, 1.0)
    )
    general_risk = (
        0.25 * rank01(entropy)
        + 0.18 * rank01(st_mae)
        + 0.07 * rank01(hard_dis)
        + 0.18 * rank01(fg_std)
        + 0.14 * rank01(centroid_spread)
        + 0.08 * rank01(uncertain)
        + 0.10 * fragmentation
    )
    low_volume = rank01(pred_fg, higher_is_more=False)
    high_volume = rank01(pred_fg, higher_is_more=True)
    small_score = 0.45 * general_risk + 0.35 * low_volume + 0.20 * fragmentation
    localization_score = (
        0.40 * rank01(centroid_spread)
        + 0.20 * rank01(fg_std)
        + 0.15 * rank01(component_std)
        + 0.15 * fragmentation
        + 0.10 * general_risk
    )
    boundary_score = (
        0.35 * general_risk
        + 0.25 * rank01(boundary_entropy)
        + 0.25 * high_volume
        + 0.15 * fragmentation
    )

    small_cut = float(np.quantile(pred_fg, 1.0 / 3.0))
    high_cut = float(np.quantile(pred_fg, 2.0 / 3.0))
    small_eligible = [i for i in range(len(acquisition)) if pred_fg[i] <= small_cut]
    centroid_cut = float(np.median(centroid_spread))
    comp_std_cut = float(np.median(component_std))
    loc_eligible = [
        i for i in range(len(acquisition))
        if centroid_spread[i] >= centroid_cut or component_std[i] > comp_std_cut or lcf[i] < 0.90
    ]
    boundary_cut = float(np.quantile(boundary_entropy, 0.60))
    boundary_eligible = [
        i for i in range(len(acquisition))
        if pred_fg[i] >= high_cut or boundary_entropy[i] >= boundary_cut or component_mean[i] >= 2.0
    ]

    X, diversity_used, diversity_missing, diversity_dropped = build_robust_diversity_matrix(
        acquisition, DIVERSITY_FEATURES
    )
    distances = pairwise_distances(X)

    selected = []
    records = []

    def take(profile, eligible, scores, quota):
        nonlocal selected, records
        chosen, recs = choose_from_pool(
            eligible,
            selected,
            scores,
            general_risk,
            distances,
            quota,
            args.selection_risk_weight,
            case_ids,
        )
        for rec in recs:
            rec["selection_profile"] = profile
            rec["suggested_review_action"] = ACTION_BY_PROFILE[profile]
        selected.extend(chosen)
        records.extend(recs)

    take(PROFILE_SMALL, small_eligible, small_score, args.small_quota)
    take(PROFILE_LOCALIZE, loc_eligible, localization_score, args.localization_quota)
    take(PROFILE_BOUNDARY, boundary_eligible, boundary_score, args.boundary_quota)

    target_n = min(args.batch_size, len(acquisition))
    if len(selected) < target_n:
        remaining = [i for i in range(len(acquisition)) if i not in selected]
        chosen, recs = choose_from_pool(
            remaining,
            selected,
            general_risk,
            general_risk,
            distances,
            target_n - len(selected),
            0.75,
            case_ids,
        )
        for rec in recs:
            rec["selection_profile"] = PROFILE_GENERAL
            rec["suggested_review_action"] = ACTION_BY_PROFILE[PROFILE_GENERAL]
        selected.extend(chosen)
        records.extend(recs)

    selected = selected[:target_n]
    selected_set = set(selected)
    order = {idx: rank + 1 for rank, idx in enumerate(selected)}
    rec_by_idx = {int(r["index"]): r for r in records if int(r["index"]) in selected_set}

    for i, row in enumerate(acquisition):
        row["committee_general_risk"] = float(general_risk[i])
        row["small_uncertain_score"] = float(small_score[i])
        row["localization_instability_score"] = float(localization_score[i])
        row["boundary_overseg_proxy_score"] = float(boundary_score[i])
        row["predicted_size_stratum"] = (
            "SMALL" if pred_fg[i] <= small_cut else ("MID" if pred_fg[i] <= high_cut else "LARGE")
        )
        if i in selected_set:
            rec = rec_by_idx[i]
            row["round5_state"] = "ANNOTATE"
            row["selection_rank"] = order[i]
            row["selection_profile"] = rec["selection_profile"]
            row["suggested_review_action"] = rec["suggested_review_action"]
            row["selection_score"] = rec["combined_selection_score"]
            row["diversity_score_when_selected"] = rec["normalized_diversity_score"]
        else:
            proxy = {
                PROFILE_SMALL: small_score[i],
                PROFILE_LOCALIZE: localization_score[i],
                PROFILE_BOUNDARY: boundary_score[i],
                PROFILE_GENERAL: general_risk[i],
            }
            profile = max(proxy, key=proxy.get)
            row["round5_state"] = "REVIEW"
            row["selection_rank"] = ""
            row["selection_profile"] = profile
            row["suggested_review_action"] = ACTION_BY_PROFILE[profile]
            row["selection_score"] = ""
            row["diversity_score_when_selected"] = ""

    keep_rows = [r for r in aggregated if int(r["locked_keep_candidate"]) == 1]
    for row in keep_rows:
        row["round5_state"] = "KEEP_CANDIDATE"
        row["selection_rank"] = ""
        row["selection_profile"] = "LOCKED_SCENARIO_F_KEEP"
        row["suggested_review_action"] = "KEEP_AS_CANDIDATE_ONLY"
        row["selection_score"] = ""
        row["diversity_score_when_selected"] = ""

    annotation_rows = sorted(
        [r for r in acquisition if r["round5_state"] == "ANNOTATE"],
        key=lambda r: int(r["selection_rank"]),
    )
    review_rows = sorted(
        [r for r in acquisition if r["round5_state"] == "REVIEW"],
        key=lambda r: (-float(r["committee_general_risk"]), str(r["case_id"])),
    )
    keep_rows.sort(key=lambda r: str(r["case_id"]))
    triage_rows = annotation_rows + review_rows + keep_rows

    write_csv(output_dir / "round5_annotation_batch.csv", annotation_rows)
    write_csv(output_dir / "round5_review_queue.csv", review_rows)
    write_csv(output_dir / "round5_keep_candidates.csv", keep_rows)
    write_csv(output_dir / "round5_pool_triage.csv", triage_rows)

    pack = materialize_images(output_dir, annotation_rows) if args.materialize else None

    profile_counts = {}
    for row in annotation_rows:
        key = str(row["selection_profile"])
        profile_counts[key] = profile_counts.get(key, 0) + 1

    metadata = {
        "version": "round5_final82_a3_committee_acquisition_v1",
        "selection_is_training": False,
        "current_human_gold_count": len(human_ids),
        "current_human_gold_source": str(args.final82_audit),
        "final82_a3_committee_dir": str(model_dir),
        "final82_a3_fold_checkpoints": [str(x) for x in checkpoints],
        "prediction_definition_per_fold": "raw Student+EMA 50/50 ensemble @ threshold 0.50",
        "committee_use": "acquisition ranking only; no QC calibration claim",
        "keep_definition": "preserved locked Final62 Scenario-F HIGH_CONFIDENCE_PSEUDO_LABEL status",
        "keep_is_human_gold": False,
        "keep_auto_promoted_to_training": False,
        "external31_used_for_scores": False,
        "external31_ids_excluded": len(external_ids),
        "previously_skipped_ids_excluded": sorted(excluded_skipped),
        "eligible_pool_cases": len(cases),
        "keep_candidates": len(keep_rows),
        "review_cases": len(review_rows),
        "annotation_batch_size": len(annotation_rows),
        "annotation_profile_counts": profile_counts,
        "requested_quotas": {
            PROFILE_SMALL: args.small_quota,
            PROFILE_LOCALIZE: args.localization_quota,
            PROFILE_BOUNDARY: args.boundary_quota,
        },
        "round5_rationale": (
            "Final82 improved precision/RVE and internal Dice, while a catastrophic low-Dice failure remains; "
            "Round5 shifts one slot from SMALL to LOCALIZATION versus Round4."
        ),
        "selection_rule": "profile quota -> profile score + robust diversity -> general risk/diversity fill",
        "diversity_features_used": diversity_used,
        "diversity_features_missing": diversity_missing,
        "diversity_features_dropped": diversity_dropped,
        "selected_case_ids": [str(r["case_id"]) for r in annotation_rows],
        "annotation_pack": str(pack) if pack else None,
        "next_step": "human annotate selected cases; KEEP remains candidate-only; REVIEW stays unlabeled",
    }
    (output_dir / "round5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("ROUND 5 TRIAGE RESULT")
    print("=" * 124)
    print(f"KEEP_CANDIDATE: {len(keep_rows)}")
    print(f"ANNOTATE:       {len(annotation_rows)}")
    print(f"REVIEW:         {len(review_rows)}")
    print("\nANNOTATION BATCH")
    for row in annotation_rows:
        print(
            f"  {int(row['selection_rank']):02d}. {row['case_id']} | "
            f"{row['selection_profile']} | size={row['predicted_size_stratum']} | "
            f"risk={float(row['committee_general_risk']):.4f} | "
            f"fg={float(row['mean_pred_fg_fraction']):.6f} | "
            f"centroidSpread={float(row['centroid_spread']):.4f}"
        )
    print(f"\nTriage CSV:     {output_dir / 'round5_pool_triage.csv'}")
    print(f"Annotation CSV: {output_dir / 'round5_annotation_batch.csv'}")
    print(f"KEEP CSV:       {output_dir / 'round5_keep_candidates.csv'}")
    print(f"Review CSV:     {output_dir / 'round5_review_queue.csv'}")
    if pack:
        print(f"Annotation pack: {pack}")
    print("\nKEEP remains candidate-only. Do not add KEEP predictions to HUMAN_GOLD automatically.")
    print("=" * 124)


if __name__ == "__main__":
    main()
