#!/usr/bin/env python3
"""Diagnose and reduce Round-1 over-segmentation by sweeping the segmentation threshold.

This is a *post-training* controlled experiment. It does not retrain the network and it does
not use the external QC validation labels. The exact original 47 held-out Round-1 cases and
the already-trained Round-1 fold checkpoints are reused.

Why this exists
---------------
The historical `rve` metric in train_supervised_cv.py is ABSOLUTE relative volume error:

    abs(pred_vox - gt_vox) / gt_vox

so it cannot distinguish over-segmentation from under-segmentation. This script additionally
computes SIGNED RVE:

    100 * (pred_vox - gt_vox) / gt_vox

Positive signed RVE = over-segmentation; negative = under-segmentation.

For each held-out case, inference is run once and the same probability map is evaluated at
multiple thresholds. This isolates threshold calibration from model-training changes.

Outputs
-------
  threshold_sweep_case_metrics.csv
  threshold_sweep_summary.csv
  threshold_sweep_metadata.json

The script also emits a constrained recommendation. A threshold is eligible when its mean
Dice drops by no more than --max-mean-dice-drop and its mean recall drops by no more than
--max-mean-recall-drop versus threshold 0.50. Among eligible thresholds, the one whose median
signed RVE is closest to zero is preferred; ties favor higher Dice and precision.
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
from hassl.training.trainer import build_network
import scripts.train_supervised_cv as cv


DEFAULT_ROUND1_DIR = Path("experiments/round1_cv_55_translation12")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_thresholds(text: str):
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise ValueError(f"Threshold must be in (0,1), found {value}")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("No thresholds supplied")
    if not any(abs(x - 0.50) < 1e-8 for x in values):
        raise ValueError("Threshold list must include 0.50 as the locked baseline")
    return values


def mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def median(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def load_fold_models(config, checkpoint: Path, device, source: str):
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()

    if source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"{checkpoint} has no teacher weights; cannot evaluate source={source}")
    return student, teacher


@torch.no_grad()
def evaluate_fold_thresholds(config, val_ids, checkpoint, source, thresholds, fold_idx):
    by_id = {c["id"]: c for c in cv.collect_cases(config)}
    missing = sorted(set(val_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"Fold {fold_idx}: missing held-out cases: {missing}")

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    loader = DataLoader(
        Dataset([by_id[x] for x in sorted(val_ids)], transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_fold_models(config, checkpoint, device, source)
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].float().to(device)
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            s_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
            if source == "student":
                prob = s_prob
            else:
                t_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                prob = t_prob if source == "teacher" else 0.5 * (s_prob + t_prob)

        spacing = cv.transformed_spacing(image, config)
        for threshold in thresholds:
            pred = (prob > float(threshold)).float()
            metrics = cv.case_metrics(pred, target, spacing)
            gt_vox = float(metrics["gt_vox"])
            pred_vox = float(metrics["pred_vox"])
            signed_rve = 100.0 * (pred_vox - gt_vox) / (gt_vox + 1e-8)
            metrics.update(
                fold=int(fold_idx),
                case_id=str(case_id),
                threshold=float(threshold),
                signed_rve=float(signed_rve),
                volume_ratio=float(pred_vox / (gt_vox + 1e-8)),
                source=source,
            )
            rows.append(metrics)

    del student
    if teacher is not None:
        del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows, thresholds):
    summary = []
    for threshold in thresholds:
        subset = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-8]
        if not subset:
            raise RuntimeError(f"No rows found for threshold={threshold}")
        signed = np.asarray([float(r["signed_rve"]) for r in subset], dtype=float)
        dices = np.asarray([float(r["dice"]) for r in subset], dtype=float)
        precisions = np.asarray([float(r["precision"]) for r in subset], dtype=float)
        recalls = np.asarray([float(r["recall"]) for r in subset], dtype=float)
        hd95 = np.asarray([float(r["hd95"]) for r in subset], dtype=float)

        summary.append({
            "threshold": float(threshold),
            "n": len(subset),
            "mean_dice": mean(dices),
            "median_dice": median(dices),
            "mean_precision": mean(precisions),
            "mean_recall": mean(recalls),
            "mean_signed_rve_pct": mean(signed),
            "median_signed_rve_pct": median(signed),
            "median_abs_rve_pct": median(np.abs(signed)),
            "overseg_gt_20pct": int(np.sum(signed > 20.0)),
            "overseg_gt_50pct": int(np.sum(signed > 50.0)),
            "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
            "dice_lt_070": int(np.sum(dices < 0.70)),
            "dice_lt_050": int(np.sum(dices < 0.50)),
            "dice_gte_080": int(np.sum(dices >= 0.80)),
            "median_hd95_mm": median(hd95),
        })
    return summary


def select_recommended(summary, max_dice_drop, max_recall_drop):
    baseline = next((r for r in summary if abs(float(r["threshold"]) - 0.50) < 1e-8), None)
    if baseline is None:
        raise RuntimeError("Missing threshold 0.50 baseline")

    eligible = []
    for row in summary:
        row["mean_dice_delta_vs_050"] = float(row["mean_dice"]) - float(baseline["mean_dice"])
        row["mean_precision_delta_vs_050"] = float(row["mean_precision"]) - float(baseline["mean_precision"])
        row["mean_recall_delta_vs_050"] = float(row["mean_recall"]) - float(baseline["mean_recall"])
        row["eligible"] = int(
            float(row["mean_dice"]) >= float(baseline["mean_dice"]) - max_dice_drop
            and float(row["mean_recall"]) >= float(baseline["mean_recall"]) - max_recall_drop
        )
        if row["eligible"]:
            eligible.append(row)

    if not eligible:
        return baseline

    return min(
        eligible,
        key=lambda row: (
            abs(float(row["median_signed_rve_pct"])),
            -float(row["mean_dice"]),
            -float(row["mean_precision"]),
            float(row["threshold"]),
        ),
    )


def main():
    p = argparse.ArgumentParser(
        description="Sweep Round-1 held-out segmentation thresholds to diagnose/reduce over-segmentation"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--round1-dir", default=str(DEFAULT_ROUND1_DIR))
    p.add_argument(
        "--thresholds",
        default="0.50,0.525,0.55,0.575,0.60,0.625,0.65",
        help="Comma-separated probability thresholds; must include 0.50",
    )
    p.add_argument("--source", choices=["student", "teacher", "ensemble"], default="ensemble")
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--max-mean-dice-drop", type=float, default=0.005)
    p.add_argument("--max-mean-recall-drop", type=float, default=0.020)
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()

    try:
        thresholds = parse_thresholds(a.thresholds)
    except ValueError as exc:
        p.error(str(exc))
    if a.max_mean_dice_drop < 0 or a.max_mean_recall_drop < 0:
        p.error("Allowed Dice/recall drops must be non-negative")

    round1_dir = Path(a.round1_dir)
    plan_path = round1_dir / "round1_cv_plan.json"
    plan = read_json(plan_path)
    folds = list(plan.get("folds", []))
    if len(folds) != 5:
        raise RuntimeError(f"Expected five Round-1 folds in {plan_path}, found {len(folds)}")

    held_out_ids = [str(x) for fold in folds for x in fold.get("val_ids", [])]
    if len(held_out_ids) != 47 or len(set(held_out_ids)) != 47:
        raise RuntimeError(
            f"Expected exact once-only coverage of 47 original held-out cases, got {len(held_out_ids)} rows / "
            f"{len(set(held_out_ids))} unique"
        )

    config = HASSLConfig.from_yaml(a.config)
    cv.apply_baseline(config, a.resize_size, epochs=1)

    output_dir = Path(a.output_dir) if a.output_dir else round1_dir / "overseg_threshold_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("ROUND-1 OVER-SEGMENTATION THRESHOLD SWEEP")
    print(f"Round-1 directory: {round1_dir}")
    print(f"Held-out cases:     {len(set(held_out_ids))}")
    print(f"Source:             {a.source}")
    print(f"Thresholds:         {thresholds}")
    print("Metric note: signed RVE >0 means over-segmentation; <0 means under-segmentation.")
    print("=" * 110)

    all_rows = []
    for fold in sorted(folds, key=lambda x: int(x["fold"])):
        fold_idx = int(fold["fold"])
        checkpoint = round1_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        val_ids = [str(x) for x in fold["val_ids"]]
        print(f"Fold {fold_idx}: {len(val_ids)} held-out cases | {checkpoint}")
        all_rows.extend(
            evaluate_fold_thresholds(
                config=config,
                val_ids=val_ids,
                checkpoint=checkpoint,
                source=a.source,
                thresholds=thresholds,
                fold_idx=fold_idx,
            )
        )

    expected_rows = 47 * len(thresholds)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} case-threshold rows, found {len(all_rows)}")

    summary = summarize(all_rows, thresholds)
    recommended = select_recommended(
        summary,
        max_dice_drop=float(a.max_mean_dice_drop),
        max_recall_drop=float(a.max_mean_recall_drop),
    )

    case_fields = [
        "fold", "case_id", "threshold", "dice", "precision", "recall", "signed_rve",
        "volume_ratio", "rve", "hd95", "gt_vox", "pred_vox", "gt_fg", "pred_fg", "source",
    ]
    summary_fields = [
        "threshold", "n", "mean_dice", "median_dice", "mean_precision", "mean_recall",
        "mean_signed_rve_pct", "median_signed_rve_pct", "median_abs_rve_pct",
        "overseg_gt_20pct", "overseg_gt_50pct", "underseg_lt_minus20pct",
        "dice_lt_070", "dice_lt_050", "dice_gte_080", "median_hd95_mm",
        "mean_dice_delta_vs_050", "mean_precision_delta_vs_050", "mean_recall_delta_vs_050", "eligible",
    ]
    case_path = output_dir / "threshold_sweep_case_metrics.csv"
    summary_path = output_dir / "threshold_sweep_summary.csv"
    metadata_path = output_dir / "threshold_sweep_metadata.json"
    write_csv(case_path, all_rows, case_fields)
    write_csv(summary_path, summary, summary_fields)

    print("\nTHRESHOLD SUMMARY")
    print(
        "thr   meanDice  precision  recall   medSignedRVE  >+20%  >+50%  <-20%  Dice<.70  Dice>=.80  eligible"
    )
    for row in summary:
        print(
            f"{float(row['threshold']):.3f}  {float(row['mean_dice']):.4f}    "
            f"{float(row['mean_precision']):.4f}    {float(row['mean_recall']):.4f}   "
            f"{float(row['median_signed_rve_pct']):+8.2f}%     "
            f"{int(row['overseg_gt_20pct']):2d}     {int(row['overseg_gt_50pct']):2d}     "
            f"{int(row['underseg_lt_minus20pct']):2d}       {int(row['dice_lt_070']):2d}         "
            f"{int(row['dice_gte_080']):2d}       {int(row['eligible'])}"
        )

    metadata = {
        "version": "round1_overseg_threshold_sweep_v1",
        "round1_dir": str(round1_dir),
        "round1_plan": str(plan_path),
        "n_held_out_cases": 47,
        "source": a.source,
        "thresholds": thresholds,
        "baseline_threshold": 0.50,
        "selection_rule": (
            "Eligible if mean Dice >= threshold0.50 mean Dice - max_mean_dice_drop and mean recall >= "
            "threshold0.50 mean recall - max_mean_recall_drop. Among eligible thresholds choose minimum "
            "absolute median signed RVE; tie-break by higher mean Dice then higher precision."
        ),
        "max_mean_dice_drop": float(a.max_mean_dice_drop),
        "max_mean_recall_drop": float(a.max_mean_recall_drop),
        "recommended_threshold": float(recommended["threshold"]),
        "recommended_summary": recommended,
        "warning": (
            "Do not use the external QC validation labels to choose this threshold. If a non-0.50 threshold is "
            "adopted, Round-1 OOF QC features/policy must be rebuilt because current QC calibration was generated "
            "at segmentation threshold 0.50."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCONSTRAINED RECOMMENDATION")
    print(
        f"Threshold {float(recommended['threshold']):.3f} | mean Dice={float(recommended['mean_dice']):.4f} | "
        f"precision={float(recommended['mean_precision']):.4f} | recall={float(recommended['mean_recall']):.4f} | "
        f"median signed RVE={float(recommended['median_signed_rve_pct']):+.2f}%"
    )
    print(f"Case metrics: {case_path}")
    print(f"Summary:      {summary_path}")
    print(f"Metadata:     {metadata_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
