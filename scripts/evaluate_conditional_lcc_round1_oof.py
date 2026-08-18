#!/usr/bin/env python3
"""Sweep conditional LCC on controlled Round-1 OOF predictions.

Round-1 training contains the frozen 47 source labels plus newly annotated active-learning
labels, but the controlled held-out evaluation deliberately covers only the SAME original
47 cases as Round 0. This script preserves that design:

  - checkpoints: experiments/round1_cv_55_translation12/checkpoints/fold_*/best_checkpoint.pth
  - held-out IDs: exact original folds from cv5_supervised_47_translation12/cv_splits.json
  - provenance reference: experiments/round1_cv_55_translation12/cv_results.csv

It replays each held-out prediction once, verifies raw Dice case-by-case against the saved
Round-1 CV results, then evaluates conditional largest-connected-component rules:

    if raw_component_count > 1 and largest_component_fraction >= T:
        final_mask = largest_component(raw_mask)
    else:
        final_mask = raw_mask

No checkpoint, QC model, or pool output is modified.
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
from scripts.evaluate_conditional_lcc_oof import (
    DEFAULT_THRESHOLDS,
    metric_dict,
    parse_thresholds,
    summarize_metric_rows,
    write_csv,
)


DEFAULT_ROUND1_DIR = Path("experiments/round1_cv_55_translation12")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/round1_conditional_lcc_oof")
EXPECTED_SOURCE_CASES = 47


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_source_manifest(manifest, path: Path):
    all_ids = [str(x) for x in manifest.get("all_case_ids", [])]
    if len(all_ids) != EXPECTED_SOURCE_CASES or len(set(all_ids)) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Expected frozen source manifest with {EXPECTED_SOURCE_CASES} unique IDs, "
            f"found {len(set(all_ids))}: {path}"
        )
    held_out = [str(x) for fold in manifest.get("folds", []) for x in fold.get("val_ids", [])]
    if len(held_out) != EXPECTED_SOURCE_CASES or sorted(held_out) != sorted(all_ids):
        raise RuntimeError("Frozen source manifest does not hold out every source case exactly once")
    if len(manifest.get("folds", [])) != 5:
        raise RuntimeError("Expected exactly five original segmentation folds")
    return set(all_ids)


def verify_round1_results(rows, source_ids, path: Path):
    by_id = {str(row["case_id"]): row for row in rows}
    if len(by_id) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Round-1 CV results must contain {EXPECTED_SOURCE_CASES} unique cases, found {len(by_id)}: {path}"
        )
    if set(by_id) != set(source_ids):
        raise RuntimeError("Round-1 CV results do not contain the exact frozen 47 source IDs")
    return by_id


def verify_raw_replay(case_rows, reference_by_id, tolerance: float):
    failures = []
    deltas = []
    for row in case_rows:
        case_id = str(row["case_id"])
        expected = float(reference_by_id[case_id]["dice"])
        actual = float(row["raw"]["dice"])
        delta = actual - expected
        deltas.append(abs(delta))
        if abs(delta) > tolerance:
            failures.append(
                f"{case_id}: saved={expected:.8f} replay={actual:.8f} delta={delta:+.8f}"
            )
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            "Round-1 raw OOF replay does not reproduce cv_results.csv; refusing LCC tuning.\n" + preview
        )
    return float(max(deltas) if deltas else 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep conditional-LCC thresholds on the controlled Round-1 47-case OOF predictions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--round1-dir", default=str(DEFAULT_ROUND1_DIR))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument("--dominance-thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--replay-dice-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    if not 0.0 < args.seg_threshold < 1.0:
        parser.error("--seg-threshold must be in (0,1)")
    if args.replay_dice_tolerance <= 0:
        parser.error("--replay-dice-tolerance must be >0")
    try:
        thresholds = parse_thresholds(args.dominance_thresholds)
    except ValueError as exc:
        parser.error(str(exc))

    round1_dir = Path(args.round1_dir)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    round1_results_path = round1_dir / "cv_results.csv"

    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    if not round1_results_path.exists():
        raise FileNotFoundError(round1_results_path)

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_ids = verify_source_manifest(manifest, source_manifest_path)
    reference_by_id = verify_round1_results(read_csv(round1_results_path), source_ids, round1_results_path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Round-1 controlled CV used prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=100)

    # Current dataset may contain 55 labels. That is expected. We only evaluate the frozen
    # source IDs according to their original val folds.
    current_cases = cv.collect_cases(config)
    current_by_id = {str(case["id"]): case for case in current_cases}
    missing_source = sorted(source_ids - set(current_by_id))
    if missing_source:
        raise RuntimeError("Frozen source cases are missing from current dataset: " + ", ".join(missing_source))

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    case_rows = []
    print("=" * 118)
    print("ROUND-1 CONDITIONAL LCC OOF THRESHOLD SWEEP")
    print(f"Round-1 CV:          {round1_dir}")
    print(f"Held-out source IDs: {EXPECTED_SOURCE_CASES} (exact Round-0 folds)")
    print(f"Current labels seen: {len(current_by_id)} (new AL labels are TRAIN-ONLY, not OOF targets)")
    print(f"Segmentation cutoff: {args.seg_threshold:.2f}")
    print("Dominance thresholds: " + ", ".join(f"{x:.2f}" for x in thresholds))
    print("Raw replay must match saved Round-1 cv_results.csv before any threshold is trusted.")
    print("=" * 118)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = round1_dir / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        val_ids = [str(x) for x in fold_spec["val_ids"]]
        loader = DataLoader(
            Dataset([current_by_id[x] for x in val_ids], transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"Fold {fold_idx} checkpoint has no EMA teacher")
        student.eval()
        teacher.eval()

        for batch in loader:
            image_t = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

            with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                teacher_prob_t = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            raw_np = (
                ensemble_prob_t[0, 0].detach().float().cpu().numpy() > float(args.seg_threshold)
            ).astype(np.uint8)
            lcc_np, component_count, largest_vox, retained_fraction = keep_largest_component(raw_np)

            raw_t = torch.from_numpy(raw_np[None, None].astype(np.float32)).to(device)
            lcc_t = torch.from_numpy(lcc_np[None, None].astype(np.float32)).to(device)
            spacing = cv.transformed_spacing(image_t, config)
            raw_metrics = metric_dict(cv.case_metrics(raw_t, target_t, spacing))
            lcc_metrics = metric_dict(cv.case_metrics(lcc_t, target_t, spacing))

            case_rows.append({
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
            })

            print(
                f"[fold {fold_idx}] {case_id} | comps={component_count:4d} | "
                f"largest={retained_fraction:6.1%} | raw={raw_metrics['dice']:.4f} | LCC={lcc_metrics['dice']:.4f}"
            )

        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    case_rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    if len(case_rows) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_CASES} OOF rows, generated {len(case_rows)}")
    if {str(row["case_id"]) for row in case_rows} != source_ids:
        raise RuntimeError("Round-1 OOF replay did not cover the exact frozen source IDs")

    max_replay_delta = verify_raw_replay(case_rows, reference_by_id, args.replay_dice_tolerance)
    print(f"\nRound-1 RAW OOF provenance verification: PASS (max |Dice delta|={max_replay_delta:.8f})")

    raw_summary = summarize_metric_rows([row["raw"] for row in case_rows])
    sweep_rows = []
    decision_rows = []

    raw_row = {
        "rule": "RAW",
        "dominance_threshold": "",
        "n_lcc_applied": 0,
        "n_improved": 0,
        "n_worsened": 0,
        "n_unchanged": EXPECTED_SOURCE_CASES,
        "n_catastrophic_zeroed": 0,
    }
    raw_row.update(raw_summary)
    sweep_rows.append(raw_row)

    for threshold in thresholds:
        chosen = []
        n_applied = improved = worsened = unchanged = catastrophic = 0
        for row in case_rows:
            apply_lcc = (
                int(row["component_count_raw"]) > 1
                and float(row["largest_component_fraction"]) >= float(threshold)
            )
            selected = row["lcc"] if apply_lcc else row["raw"]
            chosen.append(selected)
            if apply_lcc:
                n_applied += 1
            delta = float(selected["dice"] - row["raw"]["dice"])
            if delta > 1e-6:
                improved += 1
            elif delta < -1e-6:
                worsened += 1
            else:
                unchanged += 1
            if float(row["raw"]["dice"]) > 1e-6 and float(selected["dice"]) <= 1e-6:
                catastrophic += 1

            decision_rows.append({
                "dominance_threshold": float(threshold),
                "fold": int(row["fold"]),
                "case_id": str(row["case_id"]),
                "component_count_raw": int(row["component_count_raw"]),
                "largest_component_fraction": float(row["largest_component_fraction"]),
                "apply_lcc": int(apply_lcc),
                "raw_dice": float(row["raw"]["dice"]),
                "final_dice": float(selected["dice"]),
                "delta_dice": delta,
                "raw_precision": float(row["raw"]["precision"]),
                "final_precision": float(selected["precision"]),
                "raw_recall": float(row["raw"]["recall"]),
                "final_recall": float(selected["recall"]),
                "raw_rve": float(row["raw"]["rve"]),
                "final_rve": float(selected["rve"]),
                "raw_hd95": float(row["raw"]["hd95"]),
                "final_hd95": float(selected["hd95"]),
            })

        summary = {
            "rule": "CONDITIONAL_LCC",
            "dominance_threshold": float(threshold),
            "n_lcc_applied": n_applied,
            "n_improved": improved,
            "n_worsened": worsened,
            "n_unchanged": unchanged,
            "n_catastrophic_zeroed": catastrophic,
        }
        summary.update(summarize_metric_rows(chosen))
        sweep_rows.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "conditional_lcc_sweep.csv"
    decisions_path = output_dir / "conditional_lcc_case_decisions.csv"
    write_csv(sweep_path, sweep_rows)
    write_csv(decisions_path, decision_rows)

    metadata = {
        "version": "round1_conditional_lcc_oof_v1",
        "round1_dir": str(round1_dir),
        "source_manifest": str(source_manifest_path),
        "round1_cv_results": str(round1_results_path),
        "n_oof_cases": EXPECTED_SOURCE_CASES,
        "n_current_human_labels": len(current_by_id),
        "segmentation_threshold": float(args.seg_threshold),
        "prediction_source": "student_teacher_50_50_ensemble",
        "connectivity": "6-connected foreground",
        "dominance_thresholds": thresholds,
        "raw_replay_dice_tolerance": float(args.replay_dice_tolerance),
        "max_raw_replay_abs_dice_delta": max_replay_delta,
        "raw_summary": raw_summary,
        "warning": (
            "These are development OOF results on the same original 47 held-out cases. The eight new AL cases "
            "were in training for every fold and therefore are not OOF QC examples. Threshold selection on these "
            "47 cases remains development tuning, not independent final validation."
        ),
        "outputs": {
            "sweep_csv": str(sweep_path),
            "case_decisions_csv": str(decisions_path),
        },
    }
    metadata_path = output_dir / "conditional_lcc_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("ROUND-1 CONDITIONAL LCC SWEEP COMPLETE")
    print(f"RAW mean Dice:       {raw_summary['mean_dice']:.4f}")
    print(f"RAW precision:       {raw_summary['mean_precision']:.4f}")
    print(f"RAW recall:          {raw_summary['mean_recall']:.4f}")
    print(f"RAW Dice <0.70:      {raw_summary['dice_lt_0p70']}")
    print(f"RAW Dice >=0.80:     {raw_summary['dice_ge_0p80']}")
    print(f"Sweep:               {sweep_path}")
    print(f"Case decisions:      {decisions_path}")
    print(f"Metadata:            {metadata_path}")
    print("Next: choose a conservative Round-1 conditional-LCC candidate from the sweep; do not reuse 0.65 blindly.")
    print("=" * 118)


if __name__ == "__main__":
    main()
