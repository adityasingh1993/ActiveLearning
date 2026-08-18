#!/usr/bin/env python3
"""Benchmark Final62 student, EMA teacher, and 50/50 ensemble on frozen external31.

Purpose
-------
Choose the simplest production inference mode for the application without adding labels or
changing training. This is an evaluation-only comparison on the same frozen external 31-case
benchmark already used for Final55/Final62 model-update checks.

Frozen conditions
-----------------
- Final62 checkpoint trained from 62 audited human labels.
- Same external31 case IDs and GT resolution as prior validation.
- No external label is used for fitting, threshold tuning, or checkpoint selection.
- Segmentation threshold fixed at 0.50 for all inference modes.
- Same preprocessing and strict native-grid MONAI inversion for every mode.
- Modes: STUDENT, EMA, ENSEMBLE=(student_prob + ema_prob)/2.

This script does not tune a deployment threshold. It reports the best observed single-network
mode separately from the ensemble so an ONNX/app deployment decision can be made explicitly.
"""

import argparse
import csv
import json
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
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
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
DEFAULT_POOL_MANIFEST = Path(
    "experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv"
)
DEFAULT_ROUND2_AUDIT = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/external31_final62_inference_modes")
MODE_ORDER = {"STUDENT": 0, "EMA": 1, "ENSEMBLE": 2}


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


def summarize(rows, failure_dice=0.70, high_quality_dice=0.80):
    out = []
    for mode in ["STUDENT", "EMA", "ENSEMBLE"]:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        dice = np.asarray([float(r["dice"]) for r in subset], dtype=float)
        precision = np.asarray([float(r["precision"]) for r in subset], dtype=float)
        recall = np.asarray([float(r["recall"]) for r in subset], dtype=float)
        signed = np.asarray([float(r["signed_rve_pct"]) for r in subset], dtype=float)
        fp = np.asarray([float(r["fp_vox"]) for r in subset], dtype=float)
        fn = np.asarray([float(r["fn_vox"]) for r in subset], dtype=float)
        out.append({
            "mode": mode,
            "n": len(subset),
            "mean_dice": float(np.mean(dice)),
            "std_dice": float(np.std(dice)),
            "median_dice": float(np.median(dice)),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "mean_signed_rve_pct": float(np.mean(signed)),
            "median_signed_rve_pct": float(np.median(signed)),
            "median_abs_rve_pct": float(np.median(np.abs(signed))),
            "mean_fp_vox": float(np.mean(fp)),
            "mean_fn_vox": float(np.mean(fn)),
            "overseg_gt_20pct": int(np.sum(signed > 20.0)),
            "overseg_gt_50pct": int(np.sum(signed > 50.0)),
            "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
            "failures_dice_lt_070": int(np.sum(dice < failure_dice)),
            "high_quality_dice_gte_080": int(np.sum(dice >= high_quality_dice)),
        })
    out.sort(key=lambda r: MODE_ORDER[r["mode"]])
    return out


def paired_deltas(rows):
    by_key = {(r["case_id"], r["mode"]): r for r in rows}
    case_ids = sorted({r["case_id"] for r in rows})
    out = []
    for case_id in case_ids:
        student = by_key[(case_id, "STUDENT")]
        ema = by_key[(case_id, "EMA")]
        ens = by_key[(case_id, "ENSEMBLE")]
        out.append({
            "case_id": case_id,
            "student_dice": float(student["dice"]),
            "ema_dice": float(ema["dice"]),
            "ensemble_dice": float(ens["dice"]),
            "ema_minus_student_dice": float(ema["dice"]) - float(student["dice"]),
            "ensemble_minus_student_dice": float(ens["dice"]) - float(student["dice"]),
            "ensemble_minus_ema_dice": float(ens["dice"]) - float(ema["dice"]),
            "student_signed_rve_pct": float(student["signed_rve_pct"]),
            "ema_signed_rve_pct": float(ema["signed_rve_pct"]),
            "ensemble_signed_rve_pct": float(ens["signed_rve_pct"]),
        })
    return out


def choose_best_single(summary_rows):
    candidates = [r for r in summary_rows if r["mode"] in {"STUDENT", "EMA"}]
    if len(candidates) != 2:
        raise RuntimeError("Expected STUDENT and EMA summary rows")
    # Prefer fewer clinically relevant failures first, then higher mean Dice, then fewer severe
    # over-segmentation cases, then smaller median absolute volume error.
    return sorted(
        candidates,
        key=lambda r: (
            int(r["failures_dice_lt_070"]),
            -float(r["mean_dice"]),
            int(r["overseg_gt_50pct"]),
            float(r["median_abs_rve_pct"]),
            MODE_ORDER[r["mode"]],
        ),
    )[0]


def main():
    p = argparse.ArgumentParser(
        description="Benchmark Final62 STUDENT vs EMA vs ENSEMBLE on frozen external31"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External GT root; repeat for multiple roots")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Deployment-mode benchmark is frozen at threshold 0.50")
    if args.expected_count < 1:
        p.error("--expected-count must be >=1")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        p.error("Require 0 <= failure-dice < high-quality-dice <= 1")

    checkpoint = Path(args.checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, pool_manifest, audit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Final62 benchmark requires prototype student + EMA checkpoint")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata is not marked passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    overlap = sorted(set(cases_by_id) & training_ids)
    if overlap:
        raise RuntimeError(
            "EXTERNAL/TRAINING LEAKAGE: external31 overlaps Final62 human training IDs: "
            + ", ".join(overlap)
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
    loader_items = [{"image": c["image"], "id": c["id"]} for c in cases]
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
    print("FINAL62 DEPLOYMENT INFERENCE-MODE BENCHMARK — FROZEN EXTERNAL31")
    print(f"Cases:                 {len(cases)}")
    print(f"Training overlap:      {len(overlap)}")
    print(f"Checkpoint:            {checkpoint}")
    print(f"Threshold:             {args.threshold:.2f} (fixed)")
    print(f"Device:                {device}")
    print("Modes: STUDENT | EMA | ENSEMBLE (50/50 probabilities)")
    print("External labels are evaluation-only; no threshold/checkpoint/model fitting occurs here.")
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
                s_prob_t = torch.sigmoid(s_logits)
                t_prob_t = torch.sigmoid(t_logits)
                e_prob_t = 0.5 * (s_prob_t + t_prob_t)

            mode_probs = {
                "STUDENT": s_prob_t,
                "EMA": t_prob_t,
                "ENSEMBLE": e_prob_t,
            }
            case_metrics = {}
            source_image = None
            gt = None
            for mode, prob_t in mode_probs.items():
                native_prob = invert_probability_exact(
                    prob_t, batch, inverse_transform, index=0
                )
                ref, native_prob_zyx = normalize_native_probability(native_prob, case["image"])
                if source_image is None:
                    source_image = ref
                    gt = read_gt_binary(case["gt_path"], source_image)
                pred = native_prob_zyx > float(args.threshold)
                metrics = binary_metrics(pred, gt)
                case_metrics[mode] = metrics
                rows.append({
                    "case_id": case_id,
                    "mode": mode,
                    "checkpoint": str(checkpoint),
                    "threshold": float(args.threshold),
                    **metrics,
                })

            print(
                f"[{index:2d}/{len(cases)}] {case_id} | "
                f"Student={case_metrics['STUDENT']['dice']:.4f} | "
                f"EMA={case_metrics['EMA']['dice']:.4f} | "
                f"Ensemble={case_metrics['ENSEMBLE']['dice']:.4f}"
            )

    rows.sort(key=lambda r: (str(r["case_id"]), MODE_ORDER[r["mode"]]))
    summary_rows = summarize(rows, args.failure_dice, args.high_quality_dice)
    deltas = paired_deltas(rows)
    best_single = choose_best_single(summary_rows)
    ensemble = next(r for r in summary_rows if r["mode"] == "ENSEMBLE")
    ensemble_gain = float(ensemble["mean_dice"]) - float(best_single["mean_dice"])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external31_inference_mode_case_metrics.csv", rows)
    write_csv(output_dir / "external31_inference_mode_summary.csv", summary_rows)
    write_csv(output_dir / "external31_inference_mode_paired_deltas.csv", deltas)

    metadata = {
        "version": "final62_external31_inference_mode_benchmark_v1",
        "checkpoint": str(checkpoint),
        "threshold": float(args.threshold),
        "n_cases": len(cases),
        "training_external_overlap_count": len(overlap),
        "external_gt_dirs": list(args.gt_dir),
        "external_gt_usage": "evaluation_only",
        "modes": ["STUDENT", "EMA", "ENSEMBLE"],
        "best_single_mode": best_single["mode"],
        "best_single_mean_dice": float(best_single["mean_dice"]),
        "ensemble_mean_dice": float(ensemble["mean_dice"]),
        "ensemble_minus_best_single_mean_dice": ensemble_gain,
        "selection_rule_for_reported_best_single": (
            "fewer Dice<0.70 failures, then higher mean Dice, then fewer >+50% overseg cases, "
            "then lower median absolute RVE"
        ),
        "warning": (
            "This external31 set has already been inspected in prior development-stage validations. "
            "Use this benchmark for deployment-mode selection, not as a pristine prospective test."
        ),
    }
    (output_dir / "external31_inference_mode_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("FINAL62 EXTERNAL31 — STUDENT vs EMA vs ENSEMBLE")
    print("=" * 120)
    print(
        f"{'mode':<10} {'n':>3} {'meanDice':>9} {'median':>8} {'precision':>10} {'recall':>8} "
        f"{'med|RVE|':>9} {'>20':>5} {'>50':>5} {'Dice<.70':>9} {'Dice>=.80':>10}"
    )
    for r in summary_rows:
        print(
            f"{r['mode']:<10} {int(r['n']):>3d} {float(r['mean_dice']):>9.4f} "
            f"{float(r['median_dice']):>8.4f} {float(r['mean_precision']):>10.4f} "
            f"{float(r['mean_recall']):>8.4f} {float(r['median_abs_rve_pct']):>8.2f}% "
            f"{int(r['overseg_gt_20pct']):>5d} {int(r['overseg_gt_50pct']):>5d} "
            f"{int(r['failures_dice_lt_070']):>9d} {int(r['high_quality_dice_gte_080']):>10d}"
        )

    print("\nDEPLOYMENT DECISION")
    print(f"Best observed single network: {best_single['mode']}")
    print(f"Best-single mean Dice:        {float(best_single['mean_dice']):.4f}")
    print(f"Ensemble mean Dice:           {float(ensemble['mean_dice']):.4f}")
    print(f"Ensemble - best single:       {ensemble_gain:+.4f}")
    print(
        "Interpret this together with app latency/model-size requirements; this script does not benchmark runtime."
    )
    print(f"Outputs: {output_dir}")
    print("=" * 120)


if __name__ == "__main__":
    main()
