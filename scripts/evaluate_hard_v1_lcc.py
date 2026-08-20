#!/usr/bin/env python3
"""Evaluate RAW vs largest-connected-component post-processing on hard_dataset/v1.

This script operates on the Slicer-readable OOF predictions produced by
`scripts/save_hard_v1_oof_predictions.py` and compares three modes:

  RAW              : original binary prediction
  LCC1             : keep only the single largest 3D connected component
  CONDITIONAL_LCC  : apply LCC1 only when the largest component contains at least
                     --conditional-dominance (default 0.60) of predicted foreground

Both Final62 and Final72 predictions are scored against the exact same hard-v1 GT.
No threshold tuning is performed here; the input predictions are already the locked
Student+EMA ensemble at threshold 0.50.

By default, 26-connected 3D connectivity is used (SimpleITK fully connected).

Example:
  python scripts/evaluate_hard_v1_lcc.py \
    --gt-dir /data/hard_dataset/v1/label \
    --image-dir /data/hard_dataset/v1/image

Outputs:
  experiments/hard_v1_oof_final62_vs_final72/lcc_analysis/
    hard_v1_lcc_case_metrics.csv
    hard_v1_lcc_summary.csv
    hard_v1_lcc_summary.json
    filtered_predictions/<case_id>/final62_lcc1.seg.nrrd
    filtered_predictions/<case_id>/final72_lcc1.seg.nrrd
    filtered_predictions/<case_id>/final62_conditional_lcc.seg.nrrd
    filtered_predictions/<case_id>/final72_conditional_lcc.seg.nrrd
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("evaluate_hard_v1_lcc.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.data.nrrd_utils import write_mask_with_spatial_geometry

DEFAULT_PRED_ROOT = Path(
    "experiments/hard_v1_oof_final62_vs_final72/slicer_predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/hard_v1_oof_final62_vs_final72/lcc_analysis"
)
LABEL_SUFFIX = ".seg.nrrd"
IMAGE_SUFFIX = ".mha"
MODEL_FILES = {
    "FINAL62": "final62_oof_pred.seg.nrrd",
    "FINAL72": "final72_oof_pred.seg.nrrd",
}
MODES = ("RAW", "LCC1", "CONDITIONAL_LCC")


def strip_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(f"{name!r} does not end with {suffix!r}")
    return name[: -len(suffix)]


def collect_by_id(root: Path, suffix: str):
    if not root.exists():
        raise FileNotFoundError(root)
    out = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        case_id = strip_suffix(path.name, suffix)
        if case_id in out:
            raise RuntimeError(
                f"Duplicate case ID {case_id!r} under {root}: {out[case_id]} and {path}"
            )
        out[case_id] = path
    return out


def geometry_equal(a, b, atol=1e-6):
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=1e-6, atol=atol)
    )


def read_binary(path: Path):
    img = sitk.ReadImage(str(path))
    arr = np.asarray(sitk.GetArrayFromImage(img))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D mask at {path}, got shape={arr.shape}")
    return img, np.ascontiguousarray(arr > 0)


def connected_component_stats(mask: np.ndarray):
    """Return LCC mask, n_components, largest_fraction and component voxel sizes."""
    mask = np.asarray(mask, dtype=np.uint8)
    total = int(mask.sum())
    if total == 0:
        return mask.astype(bool), 0, 0.0, []

    img = sitk.GetImageFromArray(mask)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)  # 26-connected in 3D
    labeled = cc_filter.Execute(img)

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SortByObjectSizeOn()
    relabeled = relabel.Execute(labeled)
    sizes = [int(x) for x in relabel.GetSizeOfObjectsInPixels()]

    lab = np.asarray(sitk.GetArrayFromImage(relabeled))
    n_components = len(sizes)
    lcc = lab == 1 if n_components > 0 else np.zeros_like(mask, dtype=bool)
    largest = int(sizes[0]) if sizes else 0
    largest_fraction = float(largest / total) if total > 0 else 0.0
    return np.ascontiguousarray(lcc), n_components, largest_fraction, sizes


def binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_vox = int(pred.sum())
    gt_vox = int(gt.sum())
    eps = 1e-8

    dice = (2.0 * tp + eps) / (pred_vox + gt_vox + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    signed_rve = 100.0 * (pred_vox - gt_vox) / (gt_vox + eps)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "signed_rve_pct": float(signed_rve),
        "abs_rve_pct": float(abs(signed_rve)),
        "tp_vox": tp,
        "fp_vox": fp,
        "fn_vox": fn,
        "pred_vox": pred_vox,
        "gt_vox": gt_vox,
    }


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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def summarize(rows):
    out = []
    for model in MODEL_FILES:
        for mode in MODES:
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode]
            if not subset:
                continue
            dice = np.asarray([float(r["dice"]) for r in subset], dtype=float)
            precision = np.asarray([float(r["precision"]) for r in subset], dtype=float)
            recall = np.asarray([float(r["recall"]) for r in subset], dtype=float)
            signed = np.asarray([float(r["signed_rve_pct"]) for r in subset], dtype=float)
            comps = np.asarray([int(r["n_components_raw"]) for r in subset], dtype=int)
            dominance = np.asarray([float(r["largest_component_fraction_raw"]) for r in subset])
            out.append({
                "model": model,
                "mode": mode,
                "n": len(subset),
                "mean_dice": float(np.mean(dice)),
                "median_dice": float(np.median(dice)),
                "mean_precision": float(np.mean(precision)),
                "mean_recall": float(np.mean(recall)),
                "median_signed_rve_pct": float(np.median(signed)),
                "median_abs_rve_pct": float(np.median(np.abs(signed))),
                "dice_lt_0p70": int(np.sum(dice < 0.70)),
                "dice_lt_0p50": int(np.sum(dice < 0.50)),
                "dice_ge_0p80": int(np.sum(dice >= 0.80)),
                "mean_raw_components": float(np.mean(comps)),
                "mean_largest_component_fraction": float(np.mean(dominance)),
            })
    return out


def main():
    p = argparse.ArgumentParser(description="RAW vs LCC1 benchmark on hard-v1 OOF predictions")
    p.add_argument("--pred-root", default=str(DEFAULT_PRED_ROOT))
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--conditional-dominance", type=float, default=0.60)
    p.add_argument("--expected-count", type=int, default=None)
    p.add_argument(
        "--no-save-filtered",
        action="store_true",
        help="Do not write LCC/conditional Slicer .seg.nrrd masks.",
    )
    args = p.parse_args()

    if not 0.0 <= args.conditional_dominance <= 1.0:
        p.error("--conditional-dominance must be between 0 and 1")

    pred_root = Path(args.pred_root)
    gt_dir = Path(args.gt_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    gt_by_id = collect_by_id(gt_dir, LABEL_SUFFIX)
    image_by_id = collect_by_id(image_dir, IMAGE_SUFFIX)

    case_dirs = sorted([x for x in pred_root.iterdir() if x.is_dir()]) if pred_root.exists() else []
    pred_case_ids = [x.name for x in case_dirs if any((x / fn).exists() for fn in MODEL_FILES.values())]
    if not pred_case_ids:
        raise RuntimeError(f"No saved hard-v1 predictions found under {pred_root}")

    if args.expected_count is not None and len(pred_case_ids) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} prediction case directories, found {len(pred_case_ids)}"
        )

    missing_gt = sorted(set(pred_case_ids) - set(gt_by_id))
    missing_img = sorted(set(pred_case_ids) - set(image_by_id))
    if missing_gt or missing_img:
        raise RuntimeError(
            f"Hard-v1 inputs incomplete. Missing GT={missing_gt}; missing images={missing_img}"
        )

    rows = []
    print("=" * 128)
    print("HARD-V1 CONNECTED-COMPONENT ANALYSIS")
    print(f"Cases:                    {len(pred_case_ids)}")
    print("Connectivity:             26-connected / fully connected")
    print("LCC mode:                 keep n_components=1")
    print(f"Conditional dominance:    >= {args.conditional_dominance:.2f}")
    print("Input predictions:        locked OOF ensemble @ 0.50")
    print("=" * 128)

    for case_id in sorted(pred_case_ids):
        gt_img, gt = read_binary(gt_by_id[case_id])
        source_img = sitk.ReadImage(str(image_by_id[case_id]))
        if not geometry_equal(gt_img, source_img):
            raise RuntimeError(f"GT/image geometry mismatch for {case_id}")

        print(f"\n{case_id}")
        for model, filename in MODEL_FILES.items():
            pred_path = pred_root / case_id / filename
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred_img, raw = read_binary(pred_path)
            if not geometry_equal(pred_img, source_img):
                raise RuntimeError(f"Prediction/image geometry mismatch for {case_id} {model}")

            lcc, n_components, largest_fraction, component_sizes = connected_component_stats(raw)
            apply_conditional = bool(
                n_components > 1 and largest_fraction >= args.conditional_dominance
            )
            conditional = lcc if apply_conditional else raw

            mode_masks = {
                "RAW": raw,
                "LCC1": lcc,
                "CONDITIONAL_LCC": conditional,
            }

            mode_metrics = {}
            for mode, mask in mode_masks.items():
                metrics = binary_metrics(mask, gt)
                row = {
                    "case_id": case_id,
                    "model": model,
                    "mode": mode,
                    "n_components_raw": int(n_components),
                    "largest_component_fraction_raw": float(largest_fraction),
                    "largest_component_vox": int(component_sizes[0]) if component_sizes else 0,
                    "second_component_vox": int(component_sizes[1]) if len(component_sizes) > 1 else 0,
                    "conditional_lcc_applied": int(apply_conditional),
                    "conditional_dominance_threshold": float(args.conditional_dominance),
                    **metrics,
                }
                rows.append(row)
                mode_metrics[mode] = metrics

            raw_dice = mode_metrics["RAW"]["dice"]
            lcc_dice = mode_metrics["LCC1"]["dice"]
            cond_dice = mode_metrics["CONDITIONAL_LCC"]["dice"]
            print(
                f"  {model}: components={n_components:>2d} | largest={largest_fraction:6.1%} | "
                f"RAW={raw_dice:.4f} | LCC1={lcc_dice:.4f} ({lcc_dice-raw_dice:+.4f}) | "
                f"COND={cond_dice:.4f} ({cond_dice-raw_dice:+.4f}) | "
                f"conditional_applied={apply_conditional}"
            )

            if not args.no_save_filtered:
                case_out = output_dir / "filtered_predictions" / case_id
                case_out.mkdir(parents=True, exist_ok=True)
                lower = model.lower()
                write_mask_with_spatial_geometry(
                    str(case_out / f"{lower}_lcc1.seg.nrrd"),
                    lcc.astype(np.uint8),
                    reference_image_path=str(image_by_id[case_id]),
                    segment_name=f"{model}_LCC1",
                    segment_id=f"{model}_LCC1",
                )
                write_mask_with_spatial_geometry(
                    str(case_out / f"{lower}_conditional_lcc.seg.nrrd"),
                    conditional.astype(np.uint8),
                    reference_image_path=str(image_by_id[case_id]),
                    segment_name=f"{model}_ConditionalLCC",
                    segment_id=f"{model}_ConditionalLCC",
                )

    summary_rows = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hard_v1_lcc_case_metrics.csv", rows)
    write_csv(output_dir / "hard_v1_lcc_summary.csv", summary_rows)

    summary_json = {
        "version": "hard_v1_lcc_analysis_v1",
        "n_cases": len(pred_case_ids),
        "connectivity": "26-connected / fully connected",
        "lcc_rule": "keep exactly the largest connected component (n_component=1)",
        "conditional_rule": (
            "apply LCC1 only when n_components>1 and largest_component_fraction >= "
            f"{args.conditional_dominance:.3f}"
        ),
        "rows": summary_rows,
    }
    (output_dir / "hard_v1_lcc_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 128)
    print("SUMMARY")
    print(f"{'model':<10} {'mode':<17} {'n':>3} {'meanDice':>10} {'median':>10} {'precision':>10} {'recall':>10} {'med|RVE|':>10}")
    print("-" * 86)
    for r in summary_rows:
        print(
            f"{r['model']:<10} {r['mode']:<17} {int(r['n']):>3d} "
            f"{float(r['mean_dice']):>10.4f} {float(r['median_dice']):>10.4f} "
            f"{float(r['mean_precision']):>10.4f} {float(r['mean_recall']):>10.4f} "
            f"{float(r['median_abs_rve_pct']):>9.2f}%"
        )

    print(f"\nCase metrics: {output_dir / 'hard_v1_lcc_case_metrics.csv'}")
    print(f"Summary CSV:  {output_dir / 'hard_v1_lcc_summary.csv'}")
    print(f"Summary JSON: {output_dir / 'hard_v1_lcc_summary.json'}")
    if not args.no_save_filtered:
        print(f"Slicer masks: {output_dir / 'filtered_predictions'}")
    print("=" * 128)


if __name__ == "__main__":
    main()
