#!/usr/bin/env python3
"""Generate a diagnostic failure-analysis report for the deployable Final62 EMA model.

This is an evaluation-only analysis on the frozen external 31-case benchmark. It does not
change training, thresholds, QC policy, or labels.

Outputs
-------
experiments/external31_final62_ema_failure_analysis/
  ema_failure_metrics.csv
  ema_failure_summary.json
  failure_review_template.csv
  report.html
  overlays/<case_id>.png

The report emphasizes Dice<0.70 cases and quantifies whether each failure is dominated by
false-positive volume, false-negative volume, fragmentation, centroid displacement, or a
relatively small target. Image/GT ambiguity is intentionally left for manual review.
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from scipy import ndimage

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("This analysis requires SimpleITK") from exc

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
DEFAULT_OUTPUT_DIR = Path("experiments/external31_final62_ema_failure_analysis")


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
        writer.writerows(rows)


def component_stats(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, 0, 0.0
    labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    frac = float(largest / max(1, int(mask.sum())))
    return int(n), largest, frac


def centroid_zyx(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)
    return coords.mean(axis=0)


def centroid_distance_mm(pred, gt, spacing_xyz):
    cp = centroid_zyx(pred)
    cg = centroid_zyx(gt)
    if not np.isfinite(cp).all() or not np.isfinite(cg).all():
        return float("nan")
    spacing_zyx = np.asarray(tuple(reversed(spacing_xyz)), dtype=float)
    return float(np.linalg.norm((cp - cg) * spacing_zyx))


def informative_slice(mask_union, axis):
    reduce_axes = tuple(i for i in range(3) if i != axis)
    counts = mask_union.sum(axis=reduce_axes)
    if counts.size == 0:
        return 0
    if np.max(counts) <= 0:
        return int(mask_union.shape[axis] // 2)
    return int(np.argmax(counts))


def get_slice(arr, axis, index):
    if axis == 0:
        return arr[index, :, :]
    if axis == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def normalize_image_slice(image2d):
    arr = np.asarray(image2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def save_overlay(case_id, image_zyx, gt, pred, metrics, output_path):
    union = np.logical_or(gt, pred)
    orientations = [(0, "Axial"), (1, "Coronal"), (2, "Sagittal")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (axis, name) in zip(axes, orientations):
        idx = informative_slice(union, axis)
        image2d = normalize_image_slice(get_slice(image_zyx, axis, idx))
        gt2d = get_slice(gt, axis, idx)
        pred2d = get_slice(pred, axis, idx)
        ax.imshow(image2d, cmap="gray", origin="lower")
        if gt2d.any():
            ax.contour(gt2d.astype(float), levels=[0.5], colors=["lime"], linewidths=1.4)
        if pred2d.any():
            ax.contour(pred2d.astype(float), levels=[0.5], colors=["red"], linewidths=1.2)
        ax.set_title(f"{name} | slice {idx}")
        ax.axis("off")
    fig.suptitle(
        f"{case_id} | Dice={metrics['dice']:.3f} | Prec={metrics['precision']:.3f} | "
        f"Rec={metrics['recall']:.3f} | RVE={metrics['signed_rve_pct']:+.1f}%\n"
        "GT=green, EMA prediction=red"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_flags(row, small_gt_vox_threshold):
    flags = []
    if int(row["gt_vox"]) <= int(small_gt_vox_threshold):
        flags.append("SMALL_TARGET_RELATIVE")
    rve = float(row["signed_rve_pct"])
    if rve > 50.0:
        flags.append("SEVERE_OVERSEG_VOLUME")
    elif rve > 20.0:
        flags.append("OVERSEG_VOLUME")
    if rve < -20.0:
        flags.append("UNDERSEG_VOLUME")
    if int(row["pred_component_count"]) >= 3 and float(row["largest_component_fraction"]) < 0.90:
        flags.append("FRAGMENTED_PREDICTION")
    if float(row["centroid_distance_mm"]) >= 5.0:
        flags.append("CENTROID_SHIFT")
    fp = int(row["fp_vox"])
    fn = int(row["fn_vox"])
    tp = int(row["tp_vox"])
    if fp > fn * 1.5 and fp > max(100, int(0.25 * max(1, tp))):
        flags.append("FP_DOMINANT")
    elif fn > fp * 1.5 and fn > max(100, int(0.25 * max(1, tp))):
        flags.append("FN_DOMINANT")
    elif fp + fn > max(200, int(0.40 * max(1, tp))):
        flags.append("MIXED_FP_FN")
    return flags


def build_html(rows, output_path, failure_dice):
    failures = [r for r in rows if float(r["dice"]) < failure_dice]
    ordered = sorted(rows, key=lambda r: (float(r["dice"]), str(r["case_id"])))
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Final62 EMA External31 Failure Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:6px;font-size:13px;}th{background:#f2f2f2;position:sticky;top:0;}",
        ".bad{background:#ffe5e5}.ok{background:#eef8ee}img{max-width:900px;width:100%;}</style></head><body>",
        f"<h1>Final62 EMA External31 Failure Analysis</h1><p>Failures Dice&lt;{failure_dice:.2f}: "
        f"<b>{len(failures)}/{len(rows)}</b>. GT ambiguity and image quality require manual review.</p>",
        "<table><tr><th>Case</th><th>Dice</th><th>Precision</th><th>Recall</th><th>RVE</th>",
        "<th>GT vox</th><th>Pred vox</th><th>Components</th><th>Centroid mm</th><th>Flags</th><th>Overlay</th></tr>",
    ]
    for r in ordered:
        cls = "bad" if float(r["dice"]) < failure_dice else "ok"
        overlay_rel = html.escape(str(r["overlay_relative_path"]))
        lines.append(
            f"<tr class='{cls}'><td>{html.escape(str(r['case_id']))}</td>"
            f"<td>{float(r['dice']):.4f}</td><td>{float(r['precision']):.4f}</td>"
            f"<td>{float(r['recall']):.4f}</td><td>{float(r['signed_rve_pct']):+.1f}%</td>"
            f"<td>{int(r['gt_vox'])}</td><td>{int(r['pred_vox'])}</td>"
            f"<td>{int(r['pred_component_count'])}</td><td>{float(r['centroid_distance_mm']):.2f}</td>"
            f"<td>{html.escape(str(r['heuristic_failure_flags']))}</td>"
            f"<td><a href='{overlay_rel}'>view</a></td></tr>"
        )
    lines.append("</table><h2>Low-Dice overlays</h2>")
    for r in sorted(failures, key=lambda x: float(x["dice"])):
        overlay_rel = html.escape(str(r["overlay_relative_path"]))
        lines.append(
            f"<h3>{html.escape(str(r['case_id']))} — Dice {float(r['dice']):.4f}</h3>"
            f"<p>{html.escape(str(r['heuristic_failure_flags']))}</p><img src='{overlay_rel}'>"
        )
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Analyze Final62 EMA failure modes on frozen external31")
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--failure-dice", type=float, default=0.70)
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Failure analysis is frozen at segmentation threshold 0.50")

    checkpoint = Path(args.checkpoint)
    pool_manifest = Path(args.pool_manifest)
    audit_path = Path(args.round2_audit)
    output_dir = Path(args.output_dir)
    for path in [checkpoint, pool_manifest, audit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Final62 EMA benchmark expects prototype checkpoint format")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    cases = resolve_validation_cases(read_csv(pool_manifest), gt_by_id, args.expected_count)
    cases_by_id = {c["id"]: c for c in cases}

    audit = read_json(audit_path)
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))
    overlap = sorted(training_ids & set(cases_by_id))
    if overlap:
        raise RuntimeError("External/training leakage detected: " + ", ".join(overlap))

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    loader = DataLoader(
        Dataset([{"image": c["image"], "id": c["id"]} for c in cases], transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, ema = load_models(config, checkpoint, device)
    if ema is None:
        raise RuntimeError("Final62 checkpoint has no EMA teacher")
    del student
    ema.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch.get("id")
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, ema))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
            source_image, prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_image)
            pred = prob_zyx > float(args.threshold)
            metrics = binary_metrics(pred, gt)

            image_native = sitk.ReadImage(str(case["image"]))
            image_zyx = np.asarray(sitk.GetArrayFromImage(image_native), dtype=np.float32)
            image_zyx = np.squeeze(image_zyx)
            if image_zyx.shape != gt.shape:
                raise RuntimeError(
                    f"Native image/GT shape mismatch for {case_id}: image={image_zyx.shape}, gt={gt.shape}"
                )

            comp_count, largest_comp, largest_frac = component_stats(pred)
            spacing_xyz = tuple(float(x) for x in source_image.GetSpacing())
            voxel_volume_mm3 = float(np.prod(np.asarray(spacing_xyz, dtype=float)))
            centroid_mm = centroid_distance_mm(pred, gt, spacing_xyz)
            overlay_path = overlay_dir / f"{case_id}.png"
            save_overlay(case_id, image_zyx, gt, pred, metrics, overlay_path)

            row = {
                "case_id": case_id,
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "signed_rve_pct": metrics["signed_rve_pct"],
                "abs_rve_pct": metrics["abs_rve_pct"],
                "tp_vox": metrics["tp_vox"],
                "fp_vox": metrics["fp_vox"],
                "fn_vox": metrics["fn_vox"],
                "gt_vox": metrics["gt_vox"],
                "pred_vox": metrics["pred_vox"],
                "voxel_volume_mm3": voxel_volume_mm3,
                "gt_volume_mm3": float(metrics["gt_vox"] * voxel_volume_mm3),
                "pred_volume_mm3": float(metrics["pred_vox"] * voxel_volume_mm3),
                "pred_component_count": comp_count,
                "largest_component_vox": largest_comp,
                "largest_component_fraction": largest_frac,
                "centroid_distance_mm": centroid_mm,
                "overlay_relative_path": str(Path("overlays") / f"{case_id}.png"),
            }
            rows.append(row)
            print(
                f"[{index:2d}/{len(cases)}] {case_id} | Dice={metrics['dice']:.4f} | "
                f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} | "
                f"RVE={metrics['signed_rve_pct']:+.1f}% | comp={comp_count} | centroid={centroid_mm:.2f}mm"
            )

    gt_vox_values = np.asarray([int(r["gt_vox"]) for r in rows], dtype=float)
    small_thr = int(np.percentile(gt_vox_values, 25))
    for row in rows:
        flags = build_flags(row, small_thr)
        row["heuristic_failure_flags"] = ";".join(flags) if flags else "NONE"
        row["manual_review_required"] = int(float(row["dice"]) < float(args.failure_dice))

    rows.sort(key=lambda r: (float(r["dice"]), str(r["case_id"])))
    write_csv(output_dir / "ema_failure_metrics.csv", rows)

    review_rows = []
    for r in rows:
        if int(r["manual_review_required"]) != 1:
            continue
        review_rows.append({
            "case_id": r["case_id"],
            "dice": r["dice"],
            "precision": r["precision"],
            "recall": r["recall"],
            "signed_rve_pct": r["signed_rve_pct"],
            "heuristic_failure_flags": r["heuristic_failure_flags"],
            "overlay_relative_path": r["overlay_relative_path"],
            "manual_failure_type": "",
            "gt_quality_or_ambiguity": "",
            "image_quality_issue": "",
            "notes": "",
        })
    write_csv(output_dir / "failure_review_template.csv", review_rows)

    failures = [r for r in rows if float(r["dice"]) < float(args.failure_dice)]
    summary = {
        "version": "final62_ema_external31_failure_analysis_v1",
        "checkpoint": str(checkpoint),
        "inference_mode": "EMA_ONLY",
        "threshold": float(args.threshold),
        "n_cases": len(rows),
        "n_failures_dice_lt_threshold": len(failures),
        "failure_dice_threshold": float(args.failure_dice),
        "mean_dice": float(np.mean([float(r["dice"]) for r in rows])),
        "median_dice": float(np.median([float(r["dice"]) for r in rows])),
        "small_target_relative_q25_gt_vox_threshold": small_thr,
        "failure_case_ids": [str(r["case_id"]) for r in failures],
        "training_external_overlap_count": len(overlap),
        "manual_review_note": (
            "Heuristic flags describe measurable prediction behavior only. They do not determine whether the "
            "anatomy is truly unclear, the GT is questionable, or the acquisition is clinically unsegmentable."
        ),
    }
    (output_dir / "ema_failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_html(rows, output_dir / "report.html", float(args.failure_dice))

    print("\n" + "=" * 112)
    print("FINAL62 EMA FAILURE ANALYSIS COMPLETE")
    print(f"Cases:                    {len(rows)}")
    print(f"Mean / median Dice:       {summary['mean_dice']:.4f} / {summary['median_dice']:.4f}")
    print(f"Dice<{args.failure_dice:.2f}:                {len(failures)}")
    print(f"Relative small-target Q25: <= {small_thr} GT voxels")
    print("Low-Dice cases:")
    for r in failures:
        print(
            f"  {r['case_id']} | Dice={float(r['dice']):.4f} | Prec={float(r['precision']):.4f} | "
            f"Rec={float(r['recall']):.4f} | RVE={float(r['signed_rve_pct']):+.1f}% | "
            f"flags={r['heuristic_failure_flags']}"
        )
    print(f"CSV:             {output_dir / 'ema_failure_metrics.csv'}")
    print(f"Review template: {output_dir / 'failure_review_template.csv'}")
    print(f"HTML report:     {output_dir / 'report.html'}")
    print("=" * 112)


if __name__ == "__main__":
    main()
