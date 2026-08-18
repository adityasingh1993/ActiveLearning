#!/usr/bin/env python3
"""Validate the locked Round-2 Scenario-F QC gate on the frozen external 31 cases.

This is a one-way evaluation of a gate selected entirely from the original 47 OOF development
rows. The exact (unrounded) gate thresholds are read from policy_constraint_sensitivity.json;
external labels never participate in model fitting, QC fitting, or threshold selection.

Frozen inference:
- final Round-2 62-human-label checkpoint
- RAW student + EMA teacher 50/50 ensemble
- segmentation threshold 0.50
- Round-2 QC full-fit bundle
- gate signals: predicted Dice, P(Dice<0.70), P(signed RVE>+20%)
- P(overseg>+50%) remains diagnostic only
- strict MONAI native inversion; no resized-space metric fallback

The script also hard-fails if any external case ID overlaps the audited Round-2 human-training
set. This protects the external benchmark from accidental training leakage.
"""

import argparse
import csv
import json
import math
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
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import extract_features, load_models
from scripts.run_auto_label_pool import load_qc_bundle
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

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
DEFAULT_POOL_MANIFEST = Path(
    "experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_round2_qc_gate_v1")
DEFAULT_SCENARIO = "F_current_all_constraints"


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def wilson_upper(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return min(1.0, center + half)


def load_locked_scenario(path: Path, scenario_name: str):
    payload = read_json(path)
    scenarios = payload.get("scenarios", [])
    matches = [row for row in scenarios if str(row.get("scenario")) == scenario_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one scenario {scenario_name!r} in {path}, found {len(matches)}"
        )
    row = matches[0]
    if not bool(int(row.get("candidate_found", 0))):
        raise RuntimeError(f"Locked scenario {scenario_name} has no candidate gate")

    required = [
        "accept_failure_probability_max",
        "accept_predicted_dice_min",
        "accept_overseg20_probability_max",
        "max_failure_rate",
        "min_mean_dice",
        "min_hq_rate",
        "max_over20_rate",
        "max_over50_rate",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise RuntimeError(f"Scenario {scenario_name} missing fields: {missing}")
    return payload, row


def qc_predict(bundle, features):
    feature_columns = list(bundle.get("feature_columns", []))
    if not feature_columns:
        raise RuntimeError("QC bundle has no feature_columns")
    missing = [name for name in feature_columns if name not in features]
    if missing:
        raise RuntimeError(f"Extracted QC features are missing bundle columns: {missing}")
    if "overseg20_model" not in bundle:
        raise RuntimeError("Round-2 QC bundle has no overseg20_model")
    X = np.asarray([[float(features[name]) for name in feature_columns]], dtype=np.float64)
    pred_dice = float(np.clip(bundle["regression_model"].predict(X)[0], 0.0, 1.0))
    p_fail = float(bundle["failure_model"].predict_proba(X)[0, 1])
    p_over20 = float(bundle["overseg20_model"].predict_proba(X)[0, 1])
    p_over50 = float("nan")
    if "overseg50_model" in bundle:
        p_over50 = float(bundle["overseg50_model"].predict_proba(X)[0, 1])
    return pred_dice, p_fail, p_over20, p_over50


def summarize(rows, failure_dice, high_quality_dice):
    accepted = [row for row in rows if int(row["locked_gate_accept"]) == 1]
    if not accepted:
        return {
            "n_total": len(rows),
            "accepted_n": 0,
            "coverage": 0.0,
            "mean_dice": float("nan"),
            "median_dice": float("nan"),
            "mean_precision": float("nan"),
            "mean_recall": float("nan"),
            "median_signed_rve_pct": float("nan"),
            "median_abs_rve_pct": float("nan"),
            "failures": 0,
            "failure_rate": float("nan"),
            "failure_wilson95_upper": float("nan"),
            "high_quality_count": 0,
            "high_quality_rate": float("nan"),
            "overseg20_count": 0,
            "overseg20_rate": float("nan"),
            "overseg50_count": 0,
            "overseg50_rate": float("nan"),
            "underseg_minus20_count": 0,
        }

    dice = np.asarray([float(r["dice"]) for r in accepted], dtype=float)
    precision = np.asarray([float(r["precision"]) for r in accepted], dtype=float)
    recall = np.asarray([float(r["recall"]) for r in accepted], dtype=float)
    signed = np.asarray([float(r["signed_rve_pct"]) for r in accepted], dtype=float)
    failures = int(np.sum(dice < failure_dice))
    hq = int(np.sum(dice >= high_quality_dice))
    over20 = int(np.sum(signed > 20.0))
    over50 = int(np.sum(signed > 50.0))
    under20 = int(np.sum(signed < -20.0))
    n = len(accepted)
    return {
        "n_total": len(rows),
        "accepted_n": n,
        "coverage": n / len(rows),
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "failures": failures,
        "failure_rate": failures / n,
        "failure_wilson95_upper": float(wilson_upper(failures, n)),
        "high_quality_count": hq,
        "high_quality_rate": hq / n,
        "overseg20_count": over20,
        "overseg20_rate": over20 / n,
        "overseg50_count": over50,
        "overseg50_rate": over50 / n,
        "underseg_minus20_count": under20,
    }


def main():
    p = argparse.ArgumentParser(
        description="Validate the locked Round-2 Scenario-F QC AUTO_ACCEPT gate on external31"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External GT root; repeat for multiple roots")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--qc-bundle", default=str(DEFAULT_QC_BUNDLE))
    p.add_argument("--sensitivity-json", default=str(DEFAULT_SENSITIVITY))
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seg-threshold", type=float, default=0.50)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    args = p.parse_args()

    if abs(float(args.seg_threshold) - 0.50) > 1e-8:
        p.error("This locked validation is frozen at segmentation threshold 0.50")
    if args.expected_count < 1:
        p.error("--expected-count must be >=1")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        p.error("Require 0 <= failure-dice < high-quality-dice <= 1")

    checkpoint = Path(args.checkpoint)
    bundle_path = Path(args.qc_bundle)
    sensitivity_path = Path(args.sensitivity_json)
    audit_path = Path(args.round2_audit)
    pool_manifest = Path(args.pool_manifest)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, bundle_path, sensitivity_path, audit_path, pool_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)

    _, scenario = load_locked_scenario(sensitivity_path, args.scenario)
    accept_p = float(scenario["accept_failure_probability_max"])
    accept_dice = float(scenario["accept_predicted_dice_min"])
    accept_o20 = float(scenario["accept_overseg20_probability_max"])

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Locked Round-2 gate validation requires prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {case["id"]: case for case in cases}

    # Hard leakage guard: external IDs must not appear in the audited 62 human-training cases.
    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata is not marked passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    if not training_ids:
        raise RuntimeError("Round-2 audit does not contain all_current_human_label_ids")
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError(
            "EXTERNAL/TRAINING LEAKAGE: frozen external31 overlaps Round-2 human training IDs: "
            + ", ".join(overlap)
        )

    bundle = load_qc_bundle(bundle_path)
    if str(bundle.get("version", "")) != "qc_tabular_round2_overseg_v1":
        raise RuntimeError(
            f"Expected Round-2 overseg QC bundle, found version={bundle.get('version')!r}"
        )

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    loader_items = [{"image": case["image"], "id": case["id"]} for case in cases]
    loader = DataLoader(
        Dataset(loader_items, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Final62 checkpoint has no EMA teacher")
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 120)
    print("LOCKED ROUND-2 QC GATE — FROZEN EXTERNAL31 VALIDATION")
    print(f"Cases:                   {len(cases)}")
    print(f"Training/external overlap: {len(overlap)} (must be 0)")
    print(f"Scenario:                {args.scenario}")
    print(f"Exact P(failure) <=       {accept_p:.12g}")
    print(f"Exact predicted Dice >=   {accept_dice:.12g}")
    print(f"Exact P(overseg>20) <=    {accept_o20:.12g}")
    print("P(overseg>50) is diagnostic only; thresholds are not changed from external outcomes.")
    print("=" * 120)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                s_logits = cv.main_prediction(inferer(image_t, student))
                t_logits = cv.main_prediction(inferer(image_t, teacher))
                student_prob_t = torch.sigmoid(s_logits)
                teacher_prob_t = torch.sigmoid(t_logits)
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = teacher_prob_t[0, 0].detach().float().cpu().numpy()
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()
            features, _ = extract_features(
                image_np,
                student_prob,
                teacher_prob,
                ensemble_prob,
                threshold=float(args.seg_threshold),
            )
            pred_dice_qc, p_fail, p_over20, p_over50 = qc_predict(bundle, features)
            accepted = int(
                p_fail <= accept_p
                and pred_dice_qc >= accept_dice
                and p_over20 <= accept_o20
            )

            native_prob = invert_probability_exact(
                ensemble_prob_t,
                batch,
                inverse_transform,
                index=0,
            )
            source_image, native_prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)
            pred = native_prob_zyx > float(args.seg_threshold)
            metrics = binary_metrics(pred, gt)

            row = {
                "case_id": case_id,
                "original_round1_qc_bucket": case.get("qc_bucket", ""),
                "locked_gate_accept": accepted,
                "qc_predicted_dice": pred_dice_qc,
                "qc_failure_probability": p_fail,
                "qc_overseg20_probability": p_over20,
                "qc_overseg50_probability_diagnostic": p_over50,
                **metrics,
            }
            rows.append(row)
            print(
                f"[{index:2d}/{len(cases)}] {case_id} | accept={accepted} | "
                f"QC dice={pred_dice_qc:.3f} Pfail={p_fail:.3f} Pover20={p_over20:.3f} | "
                f"actual Dice={metrics['dice']:.4f} signedRVE={metrics['signed_rve_pct']:+.1f}%"
            )

    summary = summarize(rows, args.failure_dice, args.high_quality_dice)
    locked_constraints = {
        "max_failure_rate": float(scenario["max_failure_rate"]),
        "min_mean_dice": float(scenario["min_mean_dice"]),
        "min_hq_rate": float(scenario["min_hq_rate"]),
        "max_over20_rate": float(scenario["max_over20_rate"]),
        "max_over50_rate": float(scenario["max_over50_rate"]),
    }

    if summary["accepted_n"] == 0:
        evaluation = "INCONCLUSIVE_NO_EXTERNAL_ACCEPTS"
        checks = {key: None for key in locked_constraints}
    else:
        checks = {
            "failure_rate": summary["failure_rate"] <= locked_constraints["max_failure_rate"] + 1e-12,
            "mean_dice": summary["mean_dice"] >= locked_constraints["min_mean_dice"] - 1e-12,
            "hq_rate": summary["high_quality_rate"] >= locked_constraints["min_hq_rate"] - 1e-12,
            "over20_rate": summary["overseg20_rate"] <= locked_constraints["max_over20_rate"] + 1e-12,
            "over50_rate": summary["overseg50_rate"] <= locked_constraints["max_over50_rate"] + 1e-12,
        }
        evaluation = "PASS_LOCKED_OUTCOME_CONSTRAINTS" if all(checks.values()) else "FAIL_LOCKED_OUTCOME_CONSTRAINTS"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external31_locked_gate_case_results.csv", rows)
    metadata = {
        "version": "external31_round2_locked_qc_gate_v1",
        "evaluation": evaluation,
        "scenario": args.scenario,
        "sensitivity_json": str(sensitivity_path),
        "checkpoint": str(checkpoint),
        "qc_bundle": str(bundle_path),
        "pool_manifest": str(pool_manifest),
        "round2_audit": str(audit_path),
        "external_gt_dirs": list(args.gt_dir),
        "external_gt_usage": "evaluation_only",
        "n_external_cases": len(cases),
        "training_external_overlap_count": len(overlap),
        "segmentation_threshold": float(args.seg_threshold),
        "prediction_source": "raw_student_teacher_50_50_ensemble",
        "locked_gate": {
            "predicted_failure_probability_lte": accept_p,
            "predicted_dice_gte": accept_dice,
            "predicted_overseg20_probability_lte": accept_o20,
        },
        "diagnostic_only": "predicted_overseg50_probability",
        "locked_development_outcome_constraints": locked_constraints,
        "external_constraint_checks": checks,
        "external_summary": summary,
        "interpretation_warning": (
            "The external31 labels were not used to fit or select this gate. However, this same external set has "
            "been examined in earlier model/threshold analyses, so treat this as a locked gate evaluation rather "
            "than a pristine prospective validation cohort. Do not retune the gate from these results."
        ),
    }
    (output_dir / "external31_locked_gate_validation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("EXTERNAL31 LOCKED-GATE RESULT")
    print(f"Evaluation:          {evaluation}")
    print(f"Accepted:            {summary['accepted_n']}/{summary['n_total']} ({summary['coverage']:.3f})")
    if summary["accepted_n"]:
        print(f"Mean / median Dice:  {summary['mean_dice']:.4f} / {summary['median_dice']:.4f}")
        print(f"Precision / recall:  {summary['mean_precision']:.4f} / {summary['mean_recall']:.4f}")
        print(f"Failures Dice<.70:   {summary['failures']}")
        print(f"HQ Dice>=.80:        {summary['high_quality_count']}/{summary['accepted_n']} ({summary['high_quality_rate']:.3f})")
        print(f"Overseg >+20%:       {summary['overseg20_count']}/{summary['accepted_n']} ({summary['overseg20_rate']:.3f})")
        print(f"Overseg >+50%:       {summary['overseg50_count']}/{summary['accepted_n']} ({summary['overseg50_rate']:.3f})")
        print(f"Median signed RVE:   {summary['median_signed_rve_pct']:+.2f}%")
        print(f"Failure Wilson95 UB: {summary['failure_wilson95_upper']:.3f}")
    print("Do not retune this gate using the external31 outcomes.")
    print(f"Outputs: {output_dir}")
    print("=" * 120)


if __name__ == "__main__":
    main()
