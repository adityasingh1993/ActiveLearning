#!/usr/bin/env python3
"""Analyze A0/A2/A3 connected-component failures by bladder-size group on original47 OOF.

This script performs leakage-safe OOF inference on the exact frozen validation fold for every
original47 case and evaluates predictions on the same deterministic 128^3 evaluation grid used
by the CV runner. It does NOT use external31 and does NOT use GT to alter predictions.

GT-derived component ranking is diagnostic/oracle information only. In particular,
`best_component_rank_oracle` must never be used as a deployment rule.

Models
------
A0 : Final72 translation +/-12 p=.8, no flip, DiceCE
A2 : Final72 translation +/-4 p=.5, no flip, DiceCE
A3 : Final72 translation +/-4 p=.5 + LR flip p=.5, DiceCE

Outputs
-------
experiments/final72_components_by_size/
    case_component_metrics.csv
    component_size_group_summary.csv
    summary.json
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
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.training.trainer import build_network
import scripts.train_supervised_cv as cv

SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
SIZE_PROFILE = Path("experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv")
OUTPUT = Path("experiments/final72_components_by_size")
EXPECTED_OOF = 47
SIZE_ORDER = ("SMALL", "MEDIUM", "LARGE")
MODEL_SPECS = {
    "A0": {
        "cv_dir": Path("experiments/round3_cv_72_translation12"),
        "results": Path("experiments/round3_cv_72_translation12/cv_results.csv"),
    },
    "A2": {
        "cv_dir": Path("experiments/final72_screen_a2_translation4_p05"),
        "results": Path("experiments/final72_screen_a2_translation4_p05/cv_results.csv"),
    },
    "A3": {
        "cv_dir": Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05"),
        "results": Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05/cv_results.csv"),
    },
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
            writer.writerow({k: row.get(k, "") for k in fields})


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_size_group(row):
    for key in ("size_group", "bladder_size_group", "physical_volume_group"):
        value = str(row.get(key, "")).strip().upper()
        if value in SIZE_ORDER:
            return value
    raise RuntimeError("Size profile is missing a SMALL/MEDIUM/LARGE group column")


def checkpoint_for(cv_dir: Path, fold: int):
    path = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_student_teacher(config, checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    student = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()
    if "teacher" not in state:
        raise RuntimeError(f"Checkpoint has no EMA teacher: {checkpoint}")
    teacher = build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    teacher.load_state_dict(state["teacher"])
    teacher.eval()
    return student, teacher


def binary_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_vox = int(pred.sum())
    gt_vox = int(gt.sum())
    eps = 1e-8
    return {
        "dice": float((2 * tp + eps) / (pred_vox + gt_vox + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "signed_rve_pct": float(100.0 * (pred_vox - gt_vox) / (gt_vox + eps)),
        "tp_vox": tp,
        "fp_vox": fp,
        "fn_vox": fn,
        "pred_vox": pred_vox,
        "gt_vox": gt_vox,
    }


def component_diagnostic(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connected 3D
    labels, n_components = ndimage.label(pred, structure=structure)
    if n_components == 0:
        return {
            "n_components": 0,
            "multi_component": 0,
            "largest_component_fraction": 0.0,
            "largest_component_dice": 0.0,
            "best_component_rank_oracle": 0,
            "best_component_dice_oracle": 0.0,
            "largest_is_best_oracle": 0,
            "wrong_largest_oracle": 0,
            "oracle_best_minus_raw_dice": 0.0,
            "remote_fp_components": 0,
            "remote_fp_vox": 0,
            "remote_fp_fraction_of_pred": 0.0,
        }

    sizes = ndimage.sum(pred, labels=labels, index=np.arange(1, n_components + 1))
    order = np.argsort(-np.asarray(sizes, dtype=float))
    total = int(pred.sum())
    component_rows = []
    remote_fp_components = 0
    remote_fp_vox = 0

    for rank, zero_idx in enumerate(order, start=1):
        label_value = int(zero_idx) + 1
        comp = labels == label_value
        size = int(comp.sum())
        m = binary_metrics(comp, gt)
        if int(m["tp_vox"]) == 0:
            remote_fp_components += 1
            remote_fp_vox += size
        component_rows.append({
            "rank": rank,
            "size": size,
            "dice": float(m["dice"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "overlap": int(m["tp_vox"]),
        })

    largest = component_rows[0]
    best = max(component_rows, key=lambda x: (x["dice"], x["overlap"], -x["rank"]))
    raw_dice = binary_metrics(pred, gt)["dice"]
    largest_is_best = int(best["rank"] == 1)
    return {
        "n_components": int(n_components),
        "multi_component": int(n_components > 1),
        "largest_component_fraction": float(largest["size"] / total) if total else 0.0,
        "largest_component_dice": float(largest["dice"]),
        "best_component_rank_oracle": int(best["rank"]),
        "best_component_dice_oracle": float(best["dice"]),
        "largest_is_best_oracle": largest_is_best,
        "wrong_largest_oracle": int(n_components > 1 and not largest_is_best),
        "oracle_best_minus_raw_dice": float(best["dice"] - raw_dice),
        "remote_fp_components": int(remote_fp_components),
        "remote_fp_vox": int(remote_fp_vox),
        "remote_fp_fraction_of_pred": float(remote_fp_vox / total) if total else 0.0,
    }


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def safe_median(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def summarize(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "mean_dice": safe_mean([r["dice"] for r in rows]),
        "mean_precision": safe_mean([r["precision"] for r in rows]),
        "mean_recall": safe_mean([r["recall"] for r in rows]),
        "mean_components": safe_mean([r["n_components"] for r in rows]),
        "multi_component_cases": int(sum(int(r["multi_component"]) for r in rows)),
        "multi_component_pct": float(100.0 * np.mean([r["multi_component"] for r in rows])),
        "wrong_largest_cases_oracle": int(sum(int(r["wrong_largest_oracle"]) for r in rows)),
        "wrong_largest_pct_oracle": float(100.0 * np.mean([r["wrong_largest_oracle"] for r in rows])),
        "mean_largest_component_fraction": safe_mean([r["largest_component_fraction"] for r in rows]),
        "mean_remote_fp_components": safe_mean([r["remote_fp_components"] for r in rows]),
        "median_remote_fp_fraction_of_pred": safe_median([r["remote_fp_fraction_of_pred"] for r in rows]),
        "mean_largest_component_dice": safe_mean([r["largest_component_dice"] for r in rows]),
        "mean_best_component_dice_oracle": safe_mean([r["best_component_dice_oracle"] for r in rows]),
        "mean_oracle_best_minus_raw_dice": safe_mean([r["oracle_best_minus_raw_dice"] for r in rows]),
        "dice_lt_0p70": int(sum(float(r["dice"]) < 0.70 for r in rows)),
    }


def main():
    p = argparse.ArgumentParser(description="OOF component failure analysis by bladder size")
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    p.add_argument("--size-profile", default=str(SIZE_PROFILE))
    p.add_argument("--output-dir", default=str(OUTPUT))
    p.add_argument("--models", default="A0,A2,A3", help="Comma-separated subset of A0,A2,A3")
    p.add_argument("--threshold", type=float, default=0.50)
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Controlled component analysis locks threshold=0.50")

    selected_models = [x.strip().upper() for x in args.models.split(",") if x.strip()]
    if not selected_models or any(x not in MODEL_SPECS for x in selected_models):
        p.error("--models must be a comma-separated subset of A0,A2,A3")

    manifest = read_json(Path(args.source_manifest))
    all_ids = sorted(str(x) for x in manifest.get("all_case_ids", []))
    if len(set(all_ids)) != EXPECTED_OOF:
        raise RuntimeError(f"Expected frozen original47 manifest, found {len(set(all_ids))} IDs")
    fold_by_id = {}
    for fold_spec in manifest.get("folds", []):
        fold = int(fold_spec["fold"])
        for case_id in fold_spec.get("val_ids", []):
            case_id = str(case_id)
            if case_id in fold_by_id:
                raise RuntimeError(f"Case held out in multiple folds: {case_id}")
            fold_by_id[case_id] = fold
    if set(fold_by_id) != set(all_ids):
        raise RuntimeError("Frozen manifest does not hold out every original47 case exactly once")

    size_rows = read_csv(Path(args.size_profile))
    size_by_id = {}
    for row in size_rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            size_by_id[case_id] = normalize_size_group(row)
    missing_size = sorted(set(all_ids) - set(size_by_id))
    if missing_size:
        raise RuntimeError(f"Missing size group for: {missing_size}")

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=100)
    if int(config.num_classes) != 1:
        raise RuntimeError("Expected binary bladder task (num_classes=1)")

    by_id = {c["id"]: c for c in cv.collect_cases(config)}
    missing_cases = sorted(set(all_ids) - set(by_id))
    if missing_cases:
        raise RuntimeError(f"Original47 cases missing from current Final72 data_dir: {missing_cases}")

    reported = {}
    for model in selected_models:
        rows = read_csv(MODEL_SPECS[model]["results"])
        reported[model] = {str(r["case_id"]): float(r["dice"]) for r in rows}
        if set(reported[model]) != set(all_ids):
            raise RuntimeError(f"{model} results do not contain exact original47")

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    rows = []

    print("=" * 124)
    print("FINAL72 ORIGINAL47 OOF COMPONENT ANALYSIS BY BLADDER SIZE")
    print(f"Models: {selected_models} | grid=128^3 | ensemble Student+EMA @ .50 | connectivity=26")
    print("GT-based best-component fields are DIAGNOSTIC/ORACLE ONLY.")
    print("=" * 124)

    for model in selected_models:
        max_reported_delta = 0.0
        print(f"\n{model}")
        for fold in range(5):
            fold_ids = sorted(x for x in all_ids if fold_by_id[x] == fold)
            checkpoint = checkpoint_for(MODEL_SPECS[model]["cv_dir"], fold)
            student, teacher = load_student_teacher(config, checkpoint, device)
            loader = DataLoader(
                Dataset([by_id[x] for x in fold_ids], transform=transform),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
            print(f"  fold {fold}: n={len(fold_ids)} | {checkpoint}")

            with torch.no_grad():
                for batch in loader:
                    image = batch["image"].to(device)
                    target = batch["label"].float().to(device)
                    case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
                    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                        s_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
                        t_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                        prob = 0.5 * (s_prob + t_prob)
                    pred = (prob > 0.50).detach().cpu().numpy().squeeze().astype(bool)
                    gt = target.detach().cpu().numpy().squeeze().astype(bool)
                    raw = binary_metrics(pred, gt)
                    comp = component_diagnostic(pred, gt)
                    delta_reported = float(raw["dice"] - reported[model][case_id])
                    max_reported_delta = max(max_reported_delta, abs(delta_reported))
                    rows.append({
                        "case_id": case_id,
                        "fold": fold,
                        "model": model,
                        "size_group": size_by_id[case_id],
                        **raw,
                        **comp,
                        "reported_cv_dice": reported[model][case_id],
                        "dice_delta_vs_reported": delta_reported,
                    })

            del student, teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"  max |recomputed Dice - cv_results Dice| = {max_reported_delta:.6f}")
        if max_reported_delta > 0.01:
            print("  WARNING: >0.01 metric discrepancy; inspect transform/checkpoint provenance before using component conclusions.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_component_metrics.csv", rows)

    summaries = []
    for group in SIZE_ORDER:
        for model in selected_models:
            subset = [r for r in rows if r["size_group"] == group and r["model"] == model]
            s = summarize(subset)
            summaries.append({"size_group": group, "model": model, **s})
    write_csv(output_dir / "component_size_group_summary.csv", summaries)

    payload = {
        "version": "final72_oof_components_by_size_v1",
        "grid": [128, 128, 128],
        "threshold": 0.50,
        "prediction": "raw Student+EMA 50/50 ensemble",
        "connectivity": 26,
        "models": selected_models,
        "summary": summaries,
        "oracle_warning": "best-component and wrong-largest fields use GT for diagnosis only and are not deployment selectors",
        "external31_access": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("COMPONENT SUMMARY BY SIZE")
    for group in SIZE_ORDER:
        print(f"\n{group}")
        for model in selected_models:
            s = next(x for x in summaries if x["size_group"] == group and x["model"] == model)
            print(
                f"  {model}: Dice={s['mean_dice']:.4f} | Prec={s['mean_precision']:.4f} | Rec={s['mean_recall']:.4f} | "
                f"components={s['mean_components']:.2f} | multi={s['multi_component_cases']}/{s['n']} "
                f"({s['multi_component_pct']:.1f}%) | wrong-largest={s['wrong_largest_cases_oracle']}/{s['n']} | "
                f"remoteFPfrac(med)={100*s['median_remote_fp_fraction_of_pred']:.1f}% | "
                f"oracle-best gain={s['mean_oracle_best_minus_raw_dice']:+.4f}"
            )
    print(f"\nOutputs: {output_dir}")
    print("=" * 124)


if __name__ == "__main__":
    main()
