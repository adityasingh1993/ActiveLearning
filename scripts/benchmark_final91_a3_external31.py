#!/usr/bin/env python3
"""Evaluate the fully trained Final91 A3 model once on the frozen external31 benchmark.

Primary predeclared operating point: raw Student+EMA 50/50 probability ensemble @ threshold 0.50,
no LCC and no threshold/post-processing tuning. Student and EMA are reported diagnostically only.
External labels are evaluation-only and are never used to select training epochs or weights.
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
from scipy.ndimage import binary_erosion, distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import _strip_suffix, get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    geometry_equal,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)

CHECKPOINT = Path("experiments/final91_a3_all91/checkpoints/final_checkpoint.pth")
TRAIN_META = Path("experiments/final91_a3_all91/final_training_metadata.json")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
BASELINE_FINAL62 = Path("experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv")
OUTPUT = Path("experiments/external31_final91_a3_locked")
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
EXPECTED_CASES = 31
THRESHOLD = 0.50
MODE_ORDER = {"STUDENT": 0, "EMA": 1, "ENSEMBLE": 2}


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def collect_exact(root: Path, suffix: str):
    if not root.exists():
        raise FileNotFoundError(root)
    by_id = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        case_id = _strip_suffix(path.name, suffix)
        if case_id in by_id:
            raise RuntimeError(f"Duplicate case ID under {root}: {case_id}")
        by_id[case_id] = path
    return by_id


def hd95_mm(pred, gt, spacing_xyz):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    p_surface = np.logical_xor(pred, binary_erosion(pred, border_value=0))
    g_surface = np.logical_xor(gt, binary_erosion(gt, border_value=0))
    spacing_zyx = tuple(float(x) for x in reversed(spacing_xyz))
    d_to_g = distance_transform_edt(~g_surface, sampling=spacing_zyx)[p_surface]
    d_to_p = distance_transform_edt(~p_surface, sampling=spacing_zyx)[g_surface]
    distances = np.concatenate([d_to_g, d_to_p])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def summarize(rows):
    out = []
    for mode in ["STUDENT", "EMA", "ENSEMBLE"]:
        subset = [r for r in rows if r["mode"] == mode]
        dice = np.asarray([float(r["dice"]) for r in subset])
        precision = np.asarray([float(r["precision"]) for r in subset])
        recall = np.asarray([float(r["recall"]) for r in subset])
        signed = np.asarray([float(r["signed_rve_pct"]) for r in subset])
        hd = np.asarray([float(r["hd95_mm"]) for r in subset])
        finite_hd = hd[np.isfinite(hd)]
        out.append({
            "mode": mode,
            "n": len(subset),
            "mean_dice": float(np.mean(dice)),
            "std_dice": float(np.std(dice)),
            "median_dice": float(np.median(dice)),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "median_signed_rve_pct": float(np.median(signed)),
            "median_abs_rve_pct": float(np.median(np.abs(signed))),
            "mean_hd95_mm": float(np.mean(finite_hd)) if finite_hd.size else float("inf"),
            "median_hd95_mm": float(np.median(finite_hd)) if finite_hd.size else float("inf"),
            "dice_lt_0p70": int(np.sum(dice < 0.70)),
            "dice_lt_0p50": int(np.sum(dice < 0.50)),
            "dice_ge_0p80": int(np.sum(dice >= 0.80)),
        })
    return out


def maybe_compare_final62(current_rows, baseline_path: Path, output_dir: Path):
    if not baseline_path.exists():
        print(f"Final62 baseline case metrics not found; skipping paired historical comparison: {baseline_path}")
        return None
    old_rows = [r for r in read_csv(baseline_path) if str(r.get("mode", "")).upper() == "ENSEMBLE"]
    new_rows = [r for r in current_rows if r["mode"] == "ENSEMBLE"]
    old = {str(r["case_id"]): r for r in old_rows}
    new = {str(r["case_id"]): r for r in new_rows}
    if set(old) != set(new) or len(new) != EXPECTED_CASES:
        raise RuntimeError("Final62 historical baseline and Final91 external benchmark do not contain identical 31 IDs")
    paired = []
    for case_id in sorted(new):
        a, b = old[case_id], new[case_id]
        paired.append({
            "case_id": case_id,
            "final62_ensemble_dice": float(a["dice"]),
            "final91_ensemble_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "final62_precision": float(a["precision"]),
            "final91_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "final62_recall": float(a["recall"]),
            "final91_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "final62_signed_rve_pct": float(a["signed_rve_pct"]),
            "final91_signed_rve_pct": float(b["signed_rve_pct"]),
        })
    delta = np.asarray([x["delta_dice"] for x in paired])
    summary = {
        "n": len(paired),
        "final62_mean_dice": float(np.mean([x["final62_ensemble_dice"] for x in paired])),
        "final91_mean_dice": float(np.mean([x["final91_ensemble_dice"] for x in paired])),
        "delta_mean_dice": float(np.mean(delta)),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "improved_ge_0p05": int(np.sum(delta >= 0.05)),
        "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
    }
    write_csv(output_dir / "final91_vs_final62_external31_case_comparison.csv", paired)
    (output_dir / "final91_vs_final62_external31_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFINAL62 ENSEMBLE -> FINAL91 ENSEMBLE — EXTERNAL31")
    print(f"Mean Dice: {summary['final62_mean_dice']:.4f} -> {summary['final91_mean_dice']:.4f} ({summary['delta_mean_dice']:+.4f})")
    print(f"Cases: improved={summary['improved']} | worsened={summary['worsened']} | +>=.05={summary['improved_ge_0p05']} | <=-.05={summary['worsened_le_minus_0p05']}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Locked Final91 A3 evaluation on frozen external31")
    p.add_argument("--config", required=True)
    p.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--training-metadata", default=str(TRAIN_META))
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--baseline-final62-case-metrics", default=str(BASELINE_FINAL62))
    p.add_argument("--output-dir", default=str(OUTPUT))
    p.add_argument("--expected-count", type=int, default=EXPECTED_CASES)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()

    if args.expected_count != EXPECTED_CASES:
        p.error("Frozen benchmark requires expected-count=31")
    if abs(float(args.threshold) - THRESHOLD) > 1e-8:
        p.error("External benchmark is locked to threshold 0.50")

    checkpoint = Path(args.checkpoint)
    train_meta_path = Path(args.training_metadata)
    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, train_meta_path, audit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_meta = read_json(train_meta_path)
    if int(train_meta.get("n_total_human_labels", -1)) != 91:
        raise RuntimeError("Training metadata is not the all91 Final91 model")
    if Path(str(train_meta.get("deployment_checkpoint", ""))) != checkpoint:
        raise RuntimeError("Requested checkpoint differs from Final91 training metadata deployment checkpoint")
    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final91 label audit is not passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(training_ids) != 91:
        raise RuntimeError("Final91 audit does not contain exactly 91 training IDs")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Final91 benchmark requires prototype DynUNet Student+EMA")
    cv.apply_baseline(config, resize_size=128, epochs=1)

    images = collect_exact(Path(args.image_dir), config.image_suffix)
    labels = collect_exact(Path(args.gt_dir), config.label_suffix)
    common = sorted(set(images) & set(labels))
    if len(common) != EXPECTED_CASES:
        raise RuntimeError(
            f"Frozen external31 mismatch: expected 31 image+label IDs, found {len(common)}. "
            f"Images={len(images)}, labels={len(labels)}"
        )
    extra_images = sorted(set(images) - set(common))
    extra_labels = sorted(set(labels) - set(common))
    if extra_images or extra_labels:
        raise RuntimeError(f"External directories contain unmatched IDs. image-only={extra_images}, label-only={extra_labels}")
    overlap = sorted(set(common) & training_ids)
    if overlap:
        raise RuntimeError("EXTERNAL/TRAINING LEAKAGE: " + ", ".join(overlap))

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    items = [{"id": case_id, "image": str(images[case_id])} for case_id in common]
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Final91 checkpoint has no EMA teacher")
    student.eval(); teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 120)
    print("FINAL91 A3 — LOCKED EXTERNAL31 EVALUATION")
    print(f"Cases:              {len(common)}")
    print(f"Training overlap:   {len(overlap)}")
    print(f"Checkpoint:         {checkpoint}")
    print("Primary result:     ENSEMBLE = Student+EMA 50/50 @ 0.50")
    print("Postprocessing:     raw, no LCC")
    print("Student/EMA:        diagnostic only; not used to choose deployment mode here")
    print("External31 labels:  evaluation only")
    print("=" * 120)

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            raw_id = batch["id"]
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            image_t = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                s_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                t_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                e_prob = 0.5 * (s_prob + t_prob)
            case_metrics = {}
            for mode, prob_t in {"STUDENT": s_prob, "EMA": t_prob, "ENSEMBLE": e_prob}.items():
                native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
                ref, prob_zyx = normalize_native_probability(native_prob, images[case_id])
                gt = read_gt_binary(labels[case_id], ref)
                pred = prob_zyx > THRESHOLD
                metrics = binary_metrics(pred, gt)
                metrics["hd95_mm"] = hd95_mm(pred, gt, ref.GetSpacing())
                row = {"case_id": case_id, "mode": mode, "threshold": THRESHOLD, **metrics}
                rows.append(row)
                case_metrics[mode] = row
            print(
                f"[{idx:02d}/31] {case_id} | Student={case_metrics['STUDENT']['dice']:.4f} | "
                f"EMA={case_metrics['EMA']['dice']:.4f} | Ensemble={case_metrics['ENSEMBLE']['dice']:.4f}"
            )

    rows.sort(key=lambda r: (str(r["case_id"]), MODE_ORDER[r["mode"]]))
    summary_rows = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external31_case_metrics.csv", rows)
    write_csv(output_dir / "external31_summary.csv", summary_rows)

    primary = next(x for x in summary_rows if x["mode"] == "ENSEMBLE")
    metadata = {
        "version": "final91_a3_external31_locked_v1",
        "checkpoint": str(checkpoint),
        "training_metadata": str(train_meta_path),
        "audit_metadata": str(audit_path),
        "n_training_human_gold": 91,
        "n_external": len(common),
        "training_external_overlap": len(overlap),
        "primary_mode": "ENSEMBLE",
        "prediction_definition": "Student+EMA 50/50 raw probability ensemble @ 0.50",
        "threshold": THRESHOLD,
        "postprocessing": "raw_no_lcc",
        "external_gt_usage": "evaluation_only",
        "primary_summary": primary,
        "warning": "External31 has been used in prior historical evaluations, so it is a frozen comparison benchmark, not a pristine prospective test set.",
    }
    historical = maybe_compare_final62(rows, Path(args.baseline_final62_case_metrics), output_dir)
    metadata["historical_final62_comparison_available"] = historical is not None
    if historical is not None:
        metadata["historical_final62_comparison"] = historical
    (output_dir / "external31_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("FINAL91 A3 — EXTERNAL31 PRIMARY ENSEMBLE RESULT")
    print(f"Mean Dice:          {primary['mean_dice']:.4f}")
    print(f"Median Dice:        {primary['median_dice']:.4f}")
    print(f"Precision:          {primary['mean_precision']:.4f}")
    print(f"Recall:             {primary['mean_recall']:.4f}")
    print(f"Median signed RVE:  {primary['median_signed_rve_pct']:+.2f}%")
    print(f"Median |RVE|:       {primary['median_abs_rve_pct']:.2f}%")
    print(f"Mean HD95:          {primary['mean_hd95_mm']:.3f} mm")
    print(f"Dice <0.70:         {primary['dice_lt_0p70']}")
    print(f"Dice <0.50:         {primary['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {primary['dice_ge_0p80']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
