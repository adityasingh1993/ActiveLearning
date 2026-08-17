#!/usr/bin/env python3
"""Sweep conditional largest-connected-component post-processing on frozen OOF predictions.

This is a development-only post-processing experiment. It replays the exact frozen 47-case
five-fold held-out segmentation inference once and evaluates rules of the form:

    if raw_component_count > 1 and largest_component_fraction >= T:
        final_mask = largest_component(raw_mask)
    else:
        final_mask = raw_mask

for a configurable set of dominance thresholds T.

No segmentation checkpoint, QC bundle, QC policy, or unlabeled-pool output is modified.
The purpose is to decide whether a conservative conditional-LCC rule is worth promoting
before rebuilding QC and rerunning the unlabeled pool.

Outputs:
  experiments/conditional_lcc_oof/conditional_lcc_sweep.csv
  experiments/conditional_lcc_oof/conditional_lcc_case_decisions.csv
  experiments/conditional_lcc_oof/conditional_lcc_metadata.json
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

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import load_models
from scripts.evaluate_lcc_oof import keep_largest_component


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/conditional_lcc_oof")
DEFAULT_THRESHOLDS = "0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"


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


def parse_thresholds(value: str):
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        threshold = float(token)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Conditional-LCC threshold must be in [0,1], found {threshold}")
        out.append(threshold)
    if not out:
        raise ValueError("At least one conditional-LCC threshold is required")
    return sorted(set(out))


def finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize_metric_rows(metric_rows):
    dice = np.asarray([row["dice"] for row in metric_rows], dtype=float)
    precision = np.asarray([row["precision"] for row in metric_rows], dtype=float)
    recall = np.asarray([row["recall"] for row in metric_rows], dtype=float)
    rve = np.asarray([row["rve"] for row in metric_rows], dtype=float)
    hd95 = np.asarray([row["hd95"] for row in metric_rows], dtype=float)
    hd95_finite = finite(hd95)

    return {
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "mean_rve_pct": float(np.mean(rve)),
        "median_rve_pct": float(np.median(rve)),
        "mean_abs_rve_pct": float(np.mean(np.abs(rve))),
        "median_abs_rve_pct": float(np.median(np.abs(rve))),
        "mean_hd95_mm": float(np.mean(hd95_finite)) if hd95_finite.size else float("nan"),
        "median_hd95_mm": float(np.median(hd95_finite)) if hd95_finite.size else float("nan"),
        "dice_lt_0p20": int(np.sum(dice < 0.20)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p60": int(np.sum(dice < 0.60)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def metric_dict(metrics):
    return {
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "rve": float(metrics["rve"]),
        "hd95": float(metrics["hd95"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep dominance thresholds for conditional keep-largest-component OOF post-processing"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument(
        "--dominance-thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated largest-component fractions, e.g. 0.60,0.70,0.80,0.90",
    )
    args = parser.parse_args()

    if not 0.0 < args.seg_threshold < 1.0:
        parser.error("--seg-threshold must be in (0,1)")
    try:
        dominance_thresholds = parse_thresholds(args.dominance_thresholds)
    except ValueError as exc:
        parser.error(str(exc))

    experiment_dir = Path(args.experiment_dir)
    split_manifest = (
        Path(args.split_manifest)
        if args.split_manifest
        else experiment_dir / "cv_splits.json"
    )
    output_dir = Path(args.output_dir)
    if not split_manifest.exists():
        raise FileNotFoundError(split_manifest)

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, args.resize_size, epochs=100)
    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, args.folds, split_manifest)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    case_rows = []

    print("=" * 118)
    print("CONDITIONAL LCC OOF THRESHOLD SWEEP")
    print(f"Experiment:          {experiment_dir}")
    print(f"Cases:               {len(cases)}")
    print(f"Segmentation cutoff: {args.seg_threshold:.2f}")
    print("Dominance thresholds: " + ", ".join(f"{x:.2f}" for x in dominance_thresholds))
    print("Rule: apply LCC only when raw components >1 AND largest-component fraction >= threshold")
    print("Development experiment only: QC bundle/policy and unlabeled outputs are not modified.")
    print("=" * 118)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = experiment_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        fold_cases = [by_id[case_id] for case_id in fold_spec["val_ids"]]
        loader = DataLoader(
            Dataset(fold_cases, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint, device)

        for batch in loader:
            image_t = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                if teacher is not None:
                    teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)
                else:
                    ensemble_prob_t = student_prob_t

            raw_np = (
                ensemble_prob_t[0, 0].detach().float().cpu().numpy() > float(args.seg_threshold)
            ).astype(np.uint8)
            lcc_np, component_count, largest_vox, retained_fraction = keep_largest_component(raw_np)

            raw_t = torch.from_numpy(raw_np[None, None].astype(np.float32)).to(device)
            lcc_t = torch.from_numpy(lcc_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            raw_metrics = metric_dict(cv.case_metrics(raw_t, target_t, spacing))
            lcc_metrics = metric_dict(cv.case_metrics(lcc_t, target_t, spacing))

            row = {
                "fold": fold_idx,
                "case_id": case_id,
                "component_count_raw": int(component_count),
                "raw_pred_vox": int(raw_np.sum()),
                "lcc_pred_vox": int(lcc_np.sum()),
                "largest_component_vox": int(largest_vox),
                "largest_component_fraction": float(retained_fraction),
                "removed_fraction_if_lcc": float(1.0 - retained_fraction) if raw_np.sum() else 0.0,
                "raw": raw_metrics,
                "lcc": lcc_metrics,
            }
            case_rows.append(row)

            print(
                f"[fold {fold_idx}] {case_id} | comps={component_count:4d} | "
                f"largest={retained_fraction:6.1%} | Dice raw={raw_metrics['dice']:.4f} "
                f"LCC={lcc_metrics['dice']:.4f}"
            )

        del student
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    case_rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    if len(case_rows) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} OOF cases, generated {len(case_rows)}")
    if sorted(row["case_id"] for row in case_rows) != sorted(case["id"] for case in cases):
        raise RuntimeError("Conditional-LCC OOF rows do not cover every frozen case exactly once")

    raw_metric_rows = [row["raw"] for row in case_rows]
    raw_summary = summarize_metric_rows(raw_metric_rows)

    sweep_rows = []
    decision_rows = []

    # RAW baseline is included as the first sweep row for direct comparison.
    raw_sweep = {
        "rule": "RAW",
        "dominance_threshold": "",
        "n_lcc_applied": 0,
        "n_lcc_applied_multi_component": 0,
        "n_improved": 0,
        "n_worsened": 0,
        "n_unchanged": len(case_rows),
        "n_catastrophic_zeroed": 0,
    }
    raw_sweep.update(raw_summary)
    sweep_rows.append(raw_sweep)

    for threshold in dominance_thresholds:
        chosen_metrics = []
        n_applied = 0
        improved = 0
        worsened = 0
        unchanged = 0
        catastrophic_zeroed = 0

        for row in case_rows:
            apply_lcc = (
                int(row["component_count_raw"]) > 1
                and float(row["largest_component_fraction"]) >= threshold
            )
            selected = row["lcc"] if apply_lcc else row["raw"]
            chosen_metrics.append(selected)
            if apply_lcc:
                n_applied += 1

            delta_dice = float(selected["dice"] - row["raw"]["dice"])
            if delta_dice > 1e-6:
                improved += 1
            elif delta_dice < -1e-6:
                worsened += 1
            else:
                unchanged += 1
            if float(row["raw"]["dice"]) > 1e-6 and float(selected["dice"]) <= 1e-6:
                catastrophic_zeroed += 1

            decision_rows.append({
                "dominance_threshold": threshold,
                "fold": row["fold"],
                "case_id": row["case_id"],
                "component_count_raw": row["component_count_raw"],
                "largest_component_fraction": row["largest_component_fraction"],
                "apply_lcc": int(apply_lcc),
                "raw_dice": row["raw"]["dice"],
                "final_dice": selected["dice"],
                "delta_dice": delta_dice,
                "raw_precision": row["raw"]["precision"],
                "final_precision": selected["precision"],
                "raw_recall": row["raw"]["recall"],
                "final_recall": selected["recall"],
                "raw_rve": row["raw"]["rve"],
                "final_rve": selected["rve"],
                "raw_hd95": row["raw"]["hd95"],
                "final_hd95": selected["hd95"],
            })

        summary = summarize_metric_rows(chosen_metrics)
        sweep = {
            "rule": "CONDITIONAL_LCC",
            "dominance_threshold": threshold,
            "n_lcc_applied": n_applied,
            "n_lcc_applied_multi_component": n_applied,
            "n_improved": improved,
            "n_worsened": worsened,
            "n_unchanged": unchanged,
            "n_catastrophic_zeroed": catastrophic_zeroed,
        }
        sweep.update(summary)
        sweep_rows.append(sweep)

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "conditional_lcc_sweep.csv"
    decisions_path = output_dir / "conditional_lcc_case_decisions.csv"
    write_csv(sweep_path, sweep_rows)
    write_csv(decisions_path, decision_rows)

    metadata = {
        "version": "conditional_lcc_oof_v1",
        "experiment_dir": str(experiment_dir),
        "split_manifest": str(split_manifest),
        "n_cases": len(case_rows),
        "segmentation_threshold": float(args.seg_threshold),
        "prediction_source": "student_teacher_50_50_ensemble when teacher exists",
        "connectivity": "6-connected foreground",
        "rule": "apply LCC iff raw component_count > 1 and largest_component_fraction >= dominance threshold",
        "dominance_thresholds": dominance_thresholds,
        "raw_summary": raw_summary,
        "development_warning": (
            "Threshold selection on these same 47 OOF development cases is tuning, not an unbiased final validation. "
            "If promoted, rebuild QC and evaluate the resulting pipeline before using it to reduce annotation effort."
        ),
        "outputs": {
            "sweep_csv": str(sweep_path),
            "case_decisions_csv": str(decisions_path),
        },
    }
    metadata_path = output_dir / "conditional_lcc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("CONDITIONAL LCC OOF SWEEP SUMMARY")
    print(
        "Rule      Thr   Applied  MeanDice  Precision  Recall   Med|RVE|  MedHD95  Dice<.70  Dice>=.80  Zeroed"
    )
    print("-" * 118)
    for row in sweep_rows:
        threshold_text = "--" if row["dominance_threshold"] == "" else f"{float(row['dominance_threshold']):.2f}"
        print(
            f"{row['rule'][:9]:9s} {threshold_text:>4s}  {int(row['n_lcc_applied']):7d}  "
            f"{float(row['mean_dice']):8.4f}  {float(row['mean_precision']):9.4f}  "
            f"{float(row['mean_recall']):6.4f}  {float(row['median_abs_rve_pct']):8.1f}%  "
            f"{float(row['median_hd95_mm']):7.2f}  {int(row['dice_lt_0p70']):8d}  "
            f"{int(row['dice_ge_0p80']):9d}  {int(row['n_catastrophic_zeroed']):6d}"
        )

    print("\nInterpretation guardrails:")
    print("  - Prefer fewer Dice<0.70 / more Dice>=0.80 without sacrificing mean Dice or recall materially.")
    print("  - Any catastrophic zeroing is a strong reason to reject that threshold.")
    print("  - Do not rerun the unlabeled pool with old QC calibration even if a threshold looks good.")
    print(f"\nSweep CSV:      {sweep_path}")
    print(f"Case decisions: {decisions_path}")
    print(f"Metadata:       {metadata_path}")
    print("=" * 118)


if __name__ == "__main__":
    main()
