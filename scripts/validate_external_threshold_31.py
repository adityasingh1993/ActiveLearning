#!/usr/bin/env python3
"""Validate the locked segmentation threshold on external real labels.

This script is intentionally a VALIDATION step, not a tuning step.

It preserves the QC bucket assigned by the original pool run (normally threshold 0.50),
runs the same final student + EMA-teacher 50/50 ensemble once per case, inverts the native
probability map with the exact MONAI transform instance, and evaluates two fixed operating
points against external real labels:

    baseline threshold  = 0.50
    candidate threshold = 0.85

The external labels are never used to select or optimize the threshold. They are used only
after the threshold was chosen on the original 47-case OOF development set.

Metrics are computed in native image geometry so threshold 0.50 can be compared directly
with the deployment-shaped masks produced by run_auto_label_pool.py.

Outputs
-------
  external_threshold_case_metrics.csv
  external_threshold_bucket_summary.csv
  external_threshold_paired_deltas.csv
  external_threshold_validation_metadata.json

The default expected case count is 31 to protect the current frozen external benchmark.
Override --expected-count only when intentionally validating a different frozen set.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("validate_external_threshold_31.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import _strip_suffix, get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv


DEFAULT_CHECKPOINT = Path(
    "experiments/final_supervised_round1_55_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_POOL_MANIFEST = Path(
    "experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/qc_external_validation_31_threshold085")

BUCKET_ORDER = {
    "HIGH_CONFIDENCE_PSEUDO_LABEL": 0,
    "REVIEW": 1,
    "ACTIVE_LEARN_PRIORITY": 2,
    "ALL": 3,
}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def median(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def collect_gt(gt_dirs, label_suffix):
    """Recursively collect external labels, hard-failing on duplicate exact case IDs."""
    by_id = {}
    for root_text in gt_dirs:
        root = Path(root_text)
        if not root.exists():
            raise FileNotFoundError(f"GT directory does not exist: {root}")
        for path in sorted(root.rglob(f"*{label_suffix}")):
            case_id = _strip_suffix(path.name, label_suffix)
            if case_id in by_id:
                raise RuntimeError(
                    "Duplicate GT case ID found across validation directories:\n"
                    f"  {case_id}\n  {by_id[case_id]}\n  {path}"
                )
            by_id[case_id] = path
    if not by_id:
        raise RuntimeError(
            f"No labels ending with {label_suffix!r} found under: {', '.join(gt_dirs)}"
        )
    return by_id


def resolve_validation_cases(pool_rows, gt_by_id, expected_count):
    manifest_by_id = {}
    for row in pool_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise RuntimeError("Pool manifest contains a row with empty case_id")
        if case_id in manifest_by_id:
            raise RuntimeError(f"Duplicate case_id in pool manifest: {case_id}")
        bucket = str(row.get("qc_bucket", "")).strip()
        if bucket not in BUCKET_ORDER or bucket == "ALL":
            raise RuntimeError(f"Unexpected QC bucket for {case_id}: {bucket!r}")
        manifest_by_id[case_id] = row

    matched_ids = sorted(set(manifest_by_id) & set(gt_by_id))
    if expected_count > 0 and len(matched_ids) != expected_count:
        gt_not_in_pool = sorted(set(gt_by_id) - set(manifest_by_id))
        raise RuntimeError(
            "Frozen external-validation count mismatch. Refusing to silently change the benchmark.\n"
            f"Expected matched cases: {expected_count}\n"
            f"Matched cases:          {len(matched_ids)}\n"
            f"GT labels not in pool manifest ({len(gt_not_in_pool)}): {gt_not_in_pool[:20]}\n"
            "Pass --expected-count explicitly only if you intentionally changed the frozen set."
        )

    cases = []
    for case_id in matched_ids:
        row = manifest_by_id[case_id]
        image_path = Path(row.get("image_path", ""))
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image path recorded in pool manifest no longer exists for {case_id}: {image_path}"
            )
        cases.append({
            "id": case_id,
            "image": str(image_path),
            "gt_path": str(gt_by_id[case_id]),
            "qc_bucket": row["qc_bucket"],
            "original_predicted_dice": row.get("predicted_dice", ""),
            "original_failure_probability": row.get("predicted_failure_probability", ""),
        })
    return cases


def invert_probability_exact(prob_tensor, batch_data, inverse_transform, index=0):
    """Invert a probability map without thresholding, using the original transform trace."""
    try:
        samples = decollate_batch(batch_data)
        if index >= len(samples):
            raise IndexError(f"decollated batch has {len(samples)} samples, requested {index}")
        sample = samples[index]
        sample["pred"] = prob_tensor[index].detach().cpu()
        inv_out = inverse_transform(sample)
        inv_prob = inv_out["pred"]
        if inv_prob.ndim == 4:
            inv_prob = inv_prob[0]
        if torch.is_tensor(inv_prob):
            inv_prob = inv_prob.detach().float().cpu().numpy()
        return np.asarray(inv_prob, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Exact MONAI native probability inversion failed. No resized-space fallback is allowed."
        ) from exc


def normalize_native_probability(native_prob, reference_image_path):
    """Normalize MONAI native XYZ order to SimpleITK numpy ZYX without resampling."""
    ref = sitk.ReadImage(str(reference_image_path))
    source_size_xyz = tuple(int(x) for x in ref.GetSize())
    expected_zyx = tuple(reversed(source_size_xyz))
    arr = np.squeeze(np.asarray(native_prob, dtype=np.float32))
    actual = tuple(arr.shape)

    if actual == source_size_xyz:
        arr = np.transpose(arr, (2, 1, 0))
    elif actual == expected_zyx:
        pass
    else:
        raise RuntimeError(
            "Native inversion did not return the exact source grid.\n"
            f"Image: {reference_image_path}\n"
            f"Source size XYZ: {source_size_xyz}\n"
            f"Expected numpy ZYX: {expected_zyx}\n"
            f"Inverted probability shape: {actual}"
        )
    return ref, np.ascontiguousarray(arr, dtype=np.float32)


def geometry_equal(a, b):
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        return False
    return (
        np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=1e-6, atol=1e-6)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=1e-6, atol=1e-6)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=1e-6, atol=1e-6)
    )


def read_gt_binary(path, source_image):
    gt_img = sitk.ReadImage(str(path))
    if not geometry_equal(gt_img, source_image):
        raise RuntimeError(
            "GT geometry does not exactly match the source image. Refusing implicit resampling.\n"
            f"GT: {path}\n"
            f"GT size/spacing/origin: {gt_img.GetSize()} / {gt_img.GetSpacing()} / {gt_img.GetOrigin()}\n"
            f"Image size/spacing/origin: {source_image.GetSize()} / {source_image.GetSpacing()} / {source_image.GetOrigin()}"
        )
    arr = np.asarray(sitk.GetArrayFromImage(gt_img))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected a single 3D binary segmentation for {path}, got numpy shape {arr.shape}"
        )
    return arr > 0


def binary_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    pred_vox = int(pred.sum())
    gt_vox = int(gt.sum())
    eps = 1e-8

    dice = (2.0 * tp + eps) / (pred_vox + gt_vox + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    signed_rve = 100.0 * (pred_vox - gt_vox) / (gt_vox + eps)
    abs_rve = abs(signed_rve)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "signed_rve_pct": float(signed_rve),
        "abs_rve_pct": float(abs_rve),
        "volume_ratio": float(pred_vox / (gt_vox + eps)),
        "tp_vox": tp,
        "fp_vox": fp,
        "fn_vox": fn,
        "pred_vox": pred_vox,
        "gt_vox": gt_vox,
    }


def summarize(case_rows, thresholds, failure_dice=0.70, high_quality_dice=0.80):
    buckets = ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY", "ALL"]
    rows = []
    for bucket in buckets:
        for threshold in thresholds:
            subset = [
                row for row in case_rows
                if abs(float(row["threshold"]) - threshold) < 1e-8
                and (bucket == "ALL" or row["qc_bucket"] == bucket)
            ]
            if not subset:
                continue
            dice = np.asarray([float(x["dice"]) for x in subset], dtype=float)
            precision = np.asarray([float(x["precision"]) for x in subset], dtype=float)
            recall = np.asarray([float(x["recall"]) for x in subset], dtype=float)
            signed = np.asarray([float(x["signed_rve_pct"]) for x in subset], dtype=float)
            fp = np.asarray([float(x["fp_vox"]) for x in subset], dtype=float)
            fn = np.asarray([float(x["fn_vox"]) for x in subset], dtype=float)
            rows.append({
                "qc_bucket": bucket,
                "threshold": threshold,
                "n": len(subset),
                "mean_dice": mean(dice),
                "median_dice": median(dice),
                "mean_precision": mean(precision),
                "mean_recall": mean(recall),
                "mean_signed_rve_pct": mean(signed),
                "median_signed_rve_pct": median(signed),
                "median_abs_rve_pct": median(np.abs(signed)),
                "mean_fp_vox": mean(fp),
                "mean_fn_vox": mean(fn),
                "overseg_gt_20pct": int(np.sum(signed > 20.0)),
                "overseg_gt_50pct": int(np.sum(signed > 50.0)),
                "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
                "failures_dice_lt_070": int(np.sum(dice < failure_dice)),
                "high_quality_dice_gte_080": int(np.sum(dice >= high_quality_dice)),
            })
    rows.sort(key=lambda x: (BUCKET_ORDER[x["qc_bucket"]], float(x["threshold"])))
    return rows


def paired_deltas(summary_rows, baseline, candidate):
    by_key = {(row["qc_bucket"], float(row["threshold"])): row for row in summary_rows}
    rows = []
    for bucket in ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY", "ALL"]:
        b = by_key.get((bucket, baseline))
        c = by_key.get((bucket, candidate))
        if b is None or c is None:
            continue
        rows.append({
            "qc_bucket": bucket,
            "n": int(b["n"]),
            "baseline_threshold": baseline,
            "candidate_threshold": candidate,
            "delta_mean_dice": float(c["mean_dice"]) - float(b["mean_dice"]),
            "delta_mean_precision": float(c["mean_precision"]) - float(b["mean_precision"]),
            "delta_mean_recall": float(c["mean_recall"]) - float(b["mean_recall"]),
            "delta_median_signed_rve_pct": float(c["median_signed_rve_pct"]) - float(b["median_signed_rve_pct"]),
            "delta_median_abs_rve_pct": float(c["median_abs_rve_pct"]) - float(b["median_abs_rve_pct"]),
            "baseline_failures": int(b["failures_dice_lt_070"]),
            "candidate_failures": int(c["failures_dice_lt_070"]),
            "baseline_high_quality": int(b["high_quality_dice_gte_080"]),
            "candidate_high_quality": int(c["high_quality_dice_gte_080"]),
            "baseline_overseg_gt_20pct": int(b["overseg_gt_20pct"]),
            "candidate_overseg_gt_20pct": int(c["overseg_gt_20pct"]),
        })
    return rows


def print_summary(summary_rows, deltas, baseline, candidate):
    print("\n" + "=" * 118)
    print("EXTERNAL THRESHOLD VALIDATION — ORIGINAL QC BUCKETS FROZEN")
    print("=" * 118)
    print(
        "bucket                         thr    n   meanDice  precision  recall   medSignedRVE  >+20%  Dice<.70  Dice>=.80"
    )
    for row in summary_rows:
        print(
            f"{row['qc_bucket']:<30} {float(row['threshold']):.3f}  {int(row['n']):>3d}   "
            f"{float(row['mean_dice']):.4f}    {float(row['mean_precision']):.4f}    "
            f"{float(row['mean_recall']):.4f}    {float(row['median_signed_rve_pct']):+8.2f}%    "
            f"{int(row['overseg_gt_20pct']):>3d}       {int(row['failures_dice_lt_070']):>3d}        "
            f"{int(row['high_quality_dice_gte_080']):>3d}"
        )

    print("\nPAIRED CHANGE (candidate - baseline)")
    print(
        "bucket                           Dice      Precision   Recall    medSignedRVE   failures   high-quality"
    )
    for row in deltas:
        print(
            f"{row['qc_bucket']:<30} {float(row['delta_mean_dice']):+8.4f}   "
            f"{float(row['delta_mean_precision']):+8.4f}  {float(row['delta_mean_recall']):+8.4f}   "
            f"{float(row['delta_median_signed_rve_pct']):+9.2f} pts     "
            f"{int(row['baseline_failures'])}->{int(row['candidate_failures'])}        "
            f"{int(row['baseline_high_quality'])}->{int(row['candidate_high_quality'])}"
        )

    hc = next((r for r in deltas if r["qc_bucket"] == "HIGH_CONFIDENCE_PSEUDO_LABEL"), None)
    review = next((r for r in deltas if r["qc_bucket"] == "REVIEW"), None)
    print("\nVALIDATION QUESTIONS")
    if hc:
        print(
            "  Auto-label safety: HIGH_CONF failures "
            f"{hc['baseline_failures']} -> {hc['candidate_failures']} at {baseline:.2f}->{candidate:.2f}."
        )
    if review:
        print(
            "  Review over-segmentation: median signed-RVE change "
            f"{float(review['delta_median_signed_rve_pct']):+.2f} percentage points."
        )
    print("  This script does not retune the threshold or QC policy from these labels.")


def main():
    p = argparse.ArgumentParser(
        description="Validate locked threshold 0.85 against the frozen external real-label benchmark"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External GT root; repeat for multiple roots")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--baseline-threshold", type=float, default=0.50)
    p.add_argument("--candidate-threshold", type=float, default=0.85)
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--failure-dice", type=float, default=0.70)
    p.add_argument("--high-quality-dice", type=float, default=0.80)
    args = p.parse_args()

    if not 0.0 < args.baseline_threshold < 1.0 or not 0.0 < args.candidate_threshold < 1.0:
        p.error("Thresholds must be in (0,1)")
    if args.candidate_threshold <= args.baseline_threshold:
        p.error("--candidate-threshold must be greater than --baseline-threshold for this experiment")
    if args.expected_count < 0:
        p.error("--expected-count must be >= 0")
    if not 0 <= args.failure_dice < args.high_quality_dice <= 1:
        p.error("Require 0 <= failure-dice < high-quality-dice <= 1")

    checkpoint = Path(args.checkpoint)
    pool_manifest = Path(args.pool_manifest)
    output_dir = Path(args.output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("This validation expects prototype student + EMA-teacher inference")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(pool_manifest)
    cases = resolve_validation_cases(pool_rows, gt_by_id, args.expected_count)

    bucket_counts = {
        bucket: sum(case["qc_bucket"] == bucket for case in cases)
        for bucket in ["HIGH_CONFIDENCE_PSEUDO_LABEL", "REVIEW", "ACTIVE_LEARN_PRIORITY"]
    }

    print("=" * 110)
    print("FROZEN EXTERNAL THRESHOLD VALIDATION")
    print(f"Cases:               {len(cases)}")
    print(f"Original QC buckets: {bucket_counts}")
    print(f"Baseline threshold:  {args.baseline_threshold:.3f}")
    print(f"Candidate threshold: {args.candidate_threshold:.3f}  (LOCKED before external evaluation)")
    print(f"Checkpoint:          {checkpoint}")
    print(f"Pool manifest:       {pool_manifest}")
    print("External labels are evaluation targets only; they do not alter bucket assignment or threshold.")
    print("=" * 110)

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
    loader_items = [{"image": case["image"], "id": case["id"]} for case in cases]
    loader = DataLoader(Dataset(loader_items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    by_id = {case["id"]: case for case in cases}
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Final checkpoint has no EMA teacher; cannot reproduce the validated ensemble")
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    thresholds = [float(args.baseline_threshold), float(args.candidate_threshold)]
    case_rows = []

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            case_raw = batch["id"]
            case_id = case_raw[0] if isinstance(case_raw, (list, tuple)) else str(case_raw)
            meta = by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                s_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                t_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                ensemble_prob = 0.5 * (s_prob + t_prob)

            native_prob = invert_probability_exact(ensemble_prob, batch, inverse_transform, index=0)
            source_img, prob_zyx = normalize_native_probability(native_prob, meta["image"])
            gt = read_gt_binary(meta["gt_path"], source_img)

            baseline_dice = None
            candidate_dice = None
            for threshold in thresholds:
                pred = prob_zyx > threshold
                metrics = binary_metrics(pred, gt)
                row = {
                    "case_id": case_id,
                    "qc_bucket": meta["qc_bucket"],
                    "threshold": threshold,
                    "image_path": meta["image"],
                    "gt_path": meta["gt_path"],
                    "original_predicted_dice": meta["original_predicted_dice"],
                    "original_failure_probability": meta["original_failure_probability"],
                }
                row.update(metrics)
                case_rows.append(row)
                if abs(threshold - args.baseline_threshold) < 1e-8:
                    baseline_dice = metrics["dice"]
                if abs(threshold - args.candidate_threshold) < 1e-8:
                    candidate_dice = metrics["dice"]

            print(
                f"[{index:2d}/{len(cases)}] {case_id} | {meta['qc_bucket']:<28} | "
                f"Dice {args.baseline_threshold:.2f}={baseline_dice:.4f} -> "
                f"{args.candidate_threshold:.2f}={candidate_dice:.4f} ({candidate_dice - baseline_dice:+.4f})"
            )

    summary_rows = summarize(
        case_rows,
        thresholds,
        failure_dice=args.failure_dice,
        high_quality_dice=args.high_quality_dice,
    )
    delta_rows = paired_deltas(
        summary_rows,
        float(args.baseline_threshold),
        float(args.candidate_threshold),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "external_threshold_case_metrics.csv"
    summary_path = output_dir / "external_threshold_bucket_summary.csv"
    delta_path = output_dir / "external_threshold_paired_deltas.csv"
    metadata_path = output_dir / "external_threshold_validation_metadata.json"

    case_rows.sort(key=lambda x: (BUCKET_ORDER[x["qc_bucket"]], x["case_id"], float(x["threshold"])))
    write_csv(case_path, case_rows)
    write_csv(summary_path, summary_rows)
    write_csv(delta_path, delta_rows)

    metadata = {
        "version": "external_threshold_validation_v1",
        "purpose": "locked external validation; not threshold tuning",
        "n_cases": len(cases),
        "expected_count": args.expected_count,
        "case_ids": sorted(case["id"] for case in cases),
        "original_qc_bucket_counts": bucket_counts,
        "baseline_threshold": float(args.baseline_threshold),
        "candidate_threshold": float(args.candidate_threshold),
        "candidate_threshold_selection_source": "original 47-case Round-1 OOF development set",
        "checkpoint": str(checkpoint),
        "pool_manifest": str(pool_manifest),
        "gt_dirs": [str(Path(x)) for x in args.gt_dir],
        "failure_dice": float(args.failure_dice),
        "high_quality_dice": float(args.high_quality_dice),
        "prediction_source": "student_teacher_50_50_ensemble",
        "native_inversion": "exact MONAI transform trace; no fallback resampling",
        "outputs": {
            "case_metrics": str(case_path),
            "bucket_summary": str(summary_path),
            "paired_deltas": str(delta_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_summary(
        summary_rows,
        delta_rows,
        float(args.baseline_threshold),
        float(args.candidate_threshold),
    )
    print("\nSaved:")
    print(f"  {case_path}")
    print(f"  {summary_path}")
    print(f"  {delta_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()
