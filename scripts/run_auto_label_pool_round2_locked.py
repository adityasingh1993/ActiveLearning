#!/usr/bin/env python3
"""Run Final62 + locked Round-2 QC over the remaining unlabeled pool.

This is the deployment-shaped inference stage after the Scenario-F gate passed the frozen
external31 outcome checks. The gate is still treated as a HIGH-CONFIDENCE PSEUDO-LABEL
CANDIDATE rule, not as permission to add pseudo-labels to training automatically.

Safeguards
----------
- Uses final Round-2 62-human-label checkpoint.
- Uses Round-2 QC bundle and exact unrounded Scenario-F thresholds.
- Segmentation definition is frozen RAW student + EMA teacher 50/50 at threshold 0.50.
- Excludes every audited human-training ID.
- Excludes every frozen external31 ID so the benchmark cannot leak into Round-3 acquisition.
- Native inversion is strict; resized-space output fallback is not allowed.
- P(overseg>50%) is diagnostic only.

Non-accepted triage (for acquisition only):
ACTIVE_LEARN_PRIORITY if P(failure)>=0.50 OR predicted Dice<=0.70 OR P(overseg>20%)>=0.50;
otherwise REVIEW. These triage thresholds do not define pseudo-label acceptance.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import extract_features, load_models
from scripts.run_auto_label_pool import (
    BUCKET_DIR,
    collect_pool_cases,
    invert_prediction_exact,
    load_qc_bundle,
    read_json,
    verify_native_mask_before_write,
    verify_saved_geometry,
    write_csv,
)
from scripts.validate_round2_qc_gate_external31 import load_locked_scenario, qc_predict

DEFAULT_CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_QC_BUNDLE = Path("experiments/auto_label_qc_round2_raw_v1/qc_model_bundle.pkl")
DEFAULT_SENSITIVITY = Path(
    "experiments/auto_label_qc_round2_raw_v1/policy_constraint_sensitivity.json"
)
DEFAULT_ROUND2_AUDIT = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
DEFAULT_EXTERNAL_RESULTS = Path(
    "experiments/external31_round2_qc_gate_v1/external31_locked_gate_case_results.csv"
)
DEFAULT_EXTERNAL_VALIDATION = Path(
    "experiments/external31_round2_qc_gate_v1/external31_locked_gate_validation.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_pool_round2_locked_v1")
DEFAULT_SCENARIO = "F_current_all_constraints"


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser(
        description="Run Final62 + locked Scenario-F Round-2 QC over remaining unlabeled cases"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--input-dir", default=None, help="Default: config.data_dir")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--qc-bundle", default=str(DEFAULT_QC_BUNDLE))
    p.add_argument("--sensitivity-json", default=str(DEFAULT_SENSITIVITY))
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--external-case-results", default=str(DEFAULT_EXTERNAL_RESULTS))
    p.add_argument("--external-validation-json", default=str(DEFAULT_EXTERNAL_VALIDATION))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seg-threshold", type=float, default=0.50)
    p.add_argument("--active-failure-prob", type=float, default=0.50)
    p.add_argument("--active-predicted-dice", type=float, default=0.70)
    p.add_argument("--active-overseg20-prob", type=float, default=0.50)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: process first N eligible cases")
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if abs(float(args.seg_threshold) - 0.50) > 1e-8:
        p.error("Locked Round-2 QC inference is frozen at segmentation threshold 0.50")
    for name, value in {
        "active-failure-prob": args.active_failure_prob,
        "active-predicted-dice": args.active_predicted_dice,
        "active-overseg20-prob": args.active_overseg20_prob,
    }.items():
        if not 0.0 <= value <= 1.0:
            p.error(f"--{name} must be in [0,1]")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >=1")

    checkpoint = Path(args.checkpoint)
    bundle_path = Path(args.qc_bundle)
    sensitivity_path = Path(args.sensitivity_json)
    audit_path = Path(args.round2_audit)
    external_results_path = Path(args.external_case_results)
    external_validation_path = Path(args.external_validation_json)
    output_dir = Path(args.output_dir)
    for path in [
        checkpoint, bundle_path, sensitivity_path, audit_path,
        external_results_path, external_validation_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite only when intentionally replacing the complete pool run."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Final62 QC inference requires prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 human-label audit is not marked passing")
    human_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    n_audited = int(audit.get("n_current_valid_human_labels", len(human_ids)))
    if not human_ids or len(human_ids) != n_audited:
        raise RuntimeError(
            f"Round-2 audit human-ID count mismatch: IDs={len(human_ids)} metadata={n_audited}"
        )

    external_validation = read_json(external_validation_path)
    if external_validation.get("evaluation") != "PASS_LOCKED_OUTCOME_CONSTRAINTS":
        raise RuntimeError(
            "Locked external31 gate validation is not PASS_LOCKED_OUTCOME_CONSTRAINTS; "
            "do not operationalize the gate."
        )
    if int(external_validation.get("training_external_overlap_count", -1)) != 0:
        raise RuntimeError("External validation metadata reports training/external overlap")

    external_rows = read_csv(external_results_path)
    external_ids = set(str(row.get("case_id", "")).strip() for row in external_rows)
    external_ids.discard("")
    expected_external = int(external_validation.get("n_external_cases", 31))
    if len(external_ids) != expected_external:
        raise RuntimeError(
            f"External ID count mismatch: case-results={len(external_ids)} metadata={expected_external}"
        )
    overlap = sorted(human_ids & external_ids)
    if overlap:
        raise RuntimeError("Human-training/external overlap detected: " + ", ".join(overlap))

    _, scenario = load_locked_scenario(sensitivity_path, args.scenario)
    accept_p = float(scenario["accept_failure_probability_max"])
    accept_dice = float(scenario["accept_predicted_dice_min"])
    accept_o20 = float(scenario["accept_overseg20_probability_max"])

    validation_gate = external_validation.get("locked_gate", {})
    for name, expected, actual in [
        ("P(failure)", validation_gate.get("predicted_failure_probability_lte"), accept_p),
        ("predicted Dice", validation_gate.get("predicted_dice_gte"), accept_dice),
        ("P(over20)", validation_gate.get("predicted_overseg20_probability_lte"), accept_o20),
    ]:
        if expected is None or abs(float(expected) - float(actual)) > 1e-12:
            raise RuntimeError(
                f"Locked gate drift for {name}: external validation={expected} current scenario={actual}"
            )

    bundle = load_qc_bundle(bundle_path)
    if str(bundle.get("version", "")) != "qc_tabular_round2_overseg_v1":
        raise RuntimeError(f"Unexpected QC bundle version: {bundle.get('version')!r}")
    if "overseg20_model" not in bundle:
        raise RuntimeError("Round-2 QC bundle is missing overseg20_model")

    excluded_ids = human_ids | external_ids
    input_dir = Path(args.input_dir) if args.input_dir else Path(config.data_dir)
    cases, explicit_human = collect_pool_cases(
        config, input_dir, excluded_ids, limit=args.limit
    )

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=True,
        to_tensor=True,
    )
    loader = DataLoader(Dataset(cases, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Final62 checkpoint has no EMA teacher")
    student.eval(); teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    for dirname in BUCKET_DIR.values():
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    feature_rows = []
    case_path_by_id = {str(case["id"]): str(case["image"]) for case in cases}

    print("=" * 120)
    print("ROUND-2 LOCKED QC — REMAINING UNLABELED POOL")
    print(f"Input directory:             {input_dir}")
    print(f"Audited human IDs excluded:  {len(human_ids)}")
    print(f"Frozen external IDs excluded:{len(external_ids)}")
    print(f"Explicit labels excluded:    {len(explicit_human)}")
    print(f"Eligible pool cases:         {len(cases)}")
    print(f"Checkpoint:                  {checkpoint}")
    print(f"QC bundle:                   {bundle_path}")
    print(f"Scenario:                    {args.scenario}")
    print(f"Locked gate: Pfail<={accept_p:.12g} AND predDice>={accept_dice:.12g} AND Pover20<={accept_o20:.12g}")
    print("HIGH_CONFIDENCE_PSEUDO_LABEL remains candidate-only; no automatic training promotion.")
    print("=" * 120)

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            source_path = case_path_by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = teacher_prob_t[0, 0].detach().float().cpu().numpy()
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()
            features, pred_model_np = extract_features(
                image_np, student_prob, teacher_prob, ensemble_prob, threshold=float(args.seg_threshold)
            )
            pred_dice, p_fail, p_over20, p_over50 = qc_predict(bundle, features)

            locked_accept = (
                p_fail <= accept_p
                and pred_dice >= accept_dice
                and p_over20 <= accept_o20
            )
            if locked_accept:
                bucket = "HIGH_CONFIDENCE_PSEUDO_LABEL"
            elif (
                p_fail >= float(args.active_failure_prob)
                or pred_dice <= float(args.active_predicted_dice)
                or p_over20 >= float(args.active_overseg20_prob)
            ):
                bucket = "ACTIVE_LEARN_PRIORITY"
            else:
                bucket = "REVIEW"

            native_pred = invert_prediction_exact(
                ensemble_prob_t, batch, inverse_transform, index=0
            )
            reference_image, native_pred_sitk = verify_native_mask_before_write(native_pred, source_path)
            seg_path = output_dir / BUCKET_DIR[bucket] / f"{case_id}_pred{config.label_suffix}"
            write_mask_with_spatial_geometry(
                str(seg_path), native_pred_sitk, reference_image_path=source_path
            )
            verify_saved_geometry(seg_path, reference_image)

            probability_path = ""
            if args.save_probabilities:
                probability_path = str(probability_dir / f"{case_id}.npz")
                np.savez_compressed(
                    probability_path,
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=teacher_prob.astype(np.float32),
                    ensemble_prob=ensemble_prob.astype(np.float32),
                    prediction_model_space=pred_model_np.astype(np.uint8),
                )

            manifest_rows.append({
                "case_id": case_id,
                "image_path": source_path,
                "segmentation_path": str(seg_path),
                "prediction_source": "student_teacher_50_50_ensemble",
                "segmentation_threshold": float(args.seg_threshold),
                "predicted_dice": pred_dice,
                "predicted_failure_probability": p_fail,
                "predicted_overseg20_probability": p_over20,
                "predicted_overseg50_probability_diagnostic": p_over50,
                "qc_bucket": bucket,
                "locked_gate_accept": int(locked_accept),
                "pred_vox_model_space": int(features["pred_vox"]),
                "pred_fg_fraction_model_space": float(features["pred_fg_fraction"]),
                "student_teacher_mae": float(features["student_teacher_mae"]),
                "component_count": int(features["component_count"]),
                "largest_component_fraction": float(features["largest_component_fraction"]),
                "probability_npz": probability_path,
            })

            frow = {"case_id": case_id, "image_path": source_path}
            frow.update(features)
            frow.update({
                "predicted_dice": pred_dice,
                "predicted_failure_probability": p_fail,
                "predicted_overseg20_probability": p_over20,
                "predicted_overseg50_probability_diagnostic": p_over50,
                "qc_bucket": bucket,
            })
            feature_rows.append(frow)

            print(
                f"[{index:3d}/{len(cases)}] {case_id} | DiceQC={pred_dice:.3f} "
                f"Pfail={p_fail:.3f} Pover20={p_over20:.3f} | {bucket}"
            )

    # Deterministic triage ordering by a three-signal risk score. This does NOT perform final AL selection.
    for row in manifest_rows:
        row["triage_risk_score"] = (
            0.40 * float(row["predicted_failure_probability"])
            + 0.20 * (1.0 - float(row["predicted_dice"]))
            + 0.40 * float(row["predicted_overseg20_probability"])
        )
    active_rows = [r for r in manifest_rows if r["qc_bucket"] == "ACTIVE_LEARN_PRIORITY"]
    active_rows.sort(key=lambda r: (-float(r["triage_risk_score"]), str(r["case_id"])))
    active_rank = {r["case_id"]: i + 1 for i, r in enumerate(active_rows)}
    for row in manifest_rows:
        row["active_learning_rank"] = active_rank.get(row["case_id"], "")

    bucket_order = {"ACTIVE_LEARN_PRIORITY": 0, "REVIEW": 1, "HIGH_CONFIDENCE_PSEUDO_LABEL": 2}
    manifest_rows.sort(key=lambda r: (
        bucket_order[r["qc_bucket"]],
        -float(r["triage_risk_score"]),
        str(r["case_id"]),
    ))
    feature_rows.sort(key=lambda r: str(r["case_id"]))

    manifest_path = output_dir / "auto_label_manifest.csv"
    feature_path = output_dir / "unlabeled_qc_features.csv"
    write_csv(manifest_path, manifest_rows)
    feature_fields = ["case_id", "image_path"] + list(bundle["feature_columns"]) + [
        "predicted_dice",
        "predicted_failure_probability",
        "predicted_overseg20_probability",
        "predicted_overseg50_probability_diagnostic",
        "qc_bucket",
    ]
    write_csv(feature_path, feature_rows, feature_fields)

    counts = {bucket: int(sum(r["qc_bucket"] == bucket for r in manifest_rows)) for bucket in BUCKET_DIR}
    metadata = {
        "version": "auto_label_pool_round2_locked_v1",
        "input_dir": str(input_dir),
        "n_processed": len(manifest_rows),
        "n_human_training_ids_excluded": len(human_ids),
        "n_external31_ids_excluded": len(external_ids),
        "n_explicit_human_labels_excluded": len(explicit_human),
        "checkpoint": str(checkpoint),
        "qc_bundle": str(bundle_path),
        "sensitivity_json": str(sensitivity_path),
        "scenario": args.scenario,
        "round2_audit": str(audit_path),
        "external_validation_json": str(external_validation_path),
        "external_case_results": str(external_results_path),
        "segmentation_threshold": float(args.seg_threshold),
        "prediction_source": "raw_student_teacher_50_50_ensemble",
        "locked_gate": {
            "predicted_failure_probability_lte": accept_p,
            "predicted_dice_gte": accept_dice,
            "predicted_overseg20_probability_lte": accept_o20,
        },
        "nonaccepted_triage": {
            "active_failure_probability_gte": float(args.active_failure_prob),
            "OR_predicted_dice_lte": float(args.active_predicted_dice),
            "OR_predicted_overseg20_probability_gte": float(args.active_overseg20_prob),
        },
        "triage_risk_definition": "0.40*P(failure)+0.20*(1-predDice)+0.40*P(overseg>20%)",
        "bucket_counts": counts,
        "operational_rule": (
            "HIGH_CONFIDENCE_PSEUDO_LABEL is candidate-only. It is excluded from Round-3 annotation acquisition "
            "but is not automatically ingested into segmentation training."
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("ROUND-2 LOCKED POOL COMPLETE")
    print(f"Processed:                      {len(manifest_rows)}")
    print(f"HIGH_CONFIDENCE candidates:     {counts['HIGH_CONFIDENCE_PSEUDO_LABEL']}")
    print(f"REVIEW:                         {counts['REVIEW']}")
    print(f"ACTIVE_LEARN_PRIORITY:          {counts['ACTIVE_LEARN_PRIORITY']}")
    print(f"Manifest:                       {manifest_path}")
    print(f"QC features:                    {feature_path}")
    print(f"Metadata:                       {metadata_path}")
    print("Next: run select_active_learning_batch_round3.py on all non-accepted cases.")
    print("=" * 120)


if __name__ == "__main__":
    main()
