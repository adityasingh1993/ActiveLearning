#!/usr/bin/env python3
"""Leakage-safe OOF benchmark for a curated hard-case dataset.

This script is intended for hard cases copied from the frozen original-47 CV cohort. Each hard
case is evaluated ONLY with the checkpoint from the fold where that case was held out. It
compares the frozen Final62 DynUNet CV checkpoints against the Final72 DynUNet CV checkpoints
using the same preprocessing, raw Student+EMA 50/50 probability ensemble, and threshold 0.50.

Important:
- This is NOT an all-data/final-model benchmark. Using a model trained on the evaluated case
  would leak training information.
- Every hard-set case must appear exactly once in the frozen original-47 validation folds.
- The hard-set labels may be corrected/reviewed labels; both Final62 and Final72 are rescored
  against the exact same supplied hard-v1 GT.
- No threshold tuning or post-processing is performed.

Expected layout (recursive discovery is supported):
  /data/hard_dataset/v1/image/<case_id>.mha
  /data/hard_dataset/v1/label/<case_id>.seg.nrrd

Example:
  python scripts/benchmark_hard_v1_oof_final62_final72.py \
    --config config_resize128.yaml \
    --image-dir /data/hard_dataset/v1/image \
    --gt-dir /data/hard_dataset/v1/label \
    --gpu 0
"""

import csv
import json
import os
import sys
from pathlib import Path


def _consume_option(argv, name):
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


GPU, CLEAN_ARGV = _consume_option(sys.argv, "--gpu")
if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.argv = CLEAN_ARGV

import argparse  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
import scripts.train_active_learning_round2_cv_from_audit as r2  # noqa: E402

DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_FINAL62_CV = Path("experiments/round2_cv_62_translation12")
DEFAULT_FINAL72_CV = Path("experiments/round3_cv_72_translation12")
DEFAULT_OUTPUT = Path("experiments/hard_v1_oof_final62_vs_final72")
IMAGE_SUFFIX = ".mha"
LABEL_SUFFIX = ".seg.nrrd"
THRESHOLD = 0.50


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def strip_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(f"{name!r} does not end with {suffix!r}")
    return name[: -len(suffix)]


def collect_by_id(root: Path, suffix: str):
    if not root.exists():
        raise FileNotFoundError(root)
    by_id = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        case_id = strip_suffix(path.name, suffix)
        if case_id in by_id:
            raise RuntimeError(
                f"Duplicate case ID {case_id!r} under {root}: {by_id[case_id]} and {path}"
            )
        by_id[case_id] = path
    return by_id


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


def build_fold_map(source_manifest):
    fold_by_id = {}
    for spec in source_manifest.get("folds", []):
        fold = int(spec["fold"])
        for case_id in spec.get("val_ids", []):
            case_id = str(case_id)
            if case_id in fold_by_id:
                raise RuntimeError(f"Frozen source case {case_id} appears in multiple validation folds")
            fold_by_id[case_id] = fold
    all_ids = set(str(x) for x in source_manifest.get("all_case_ids", []))
    if len(all_ids) != 47 or len(fold_by_id) != 47 or set(fold_by_id) != all_ids:
        raise RuntimeError(
            "Source manifest must contain exactly the frozen 47 cases, each held out exactly once"
        )
    return fold_by_id


def checkpoint_for(cv_dir: Path, fold: int):
    path = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing fold-{fold} checkpoint: {path}")
    return path


def summarize(rows):
    dice = np.asarray([float(x["dice"]) for x in rows], dtype=float)
    precision = np.asarray([float(x["precision"]) for x in rows], dtype=float)
    recall = np.asarray([float(x["recall"]) for x in rows], dtype=float)
    signed = np.asarray([float(x["signed_rve"]) for x in rows], dtype=float)
    hd95 = np.asarray([float(x["hd95"]) for x in rows], dtype=float)
    finite_hd = hd95[np.isfinite(hd95)]
    return {
        "n": int(len(rows)),
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_signed_rve_pct": float(np.median(signed)),
        "median_abs_rve_pct": float(np.median(np.abs(signed))),
        "mean_hd95_mm": float(np.mean(finite_hd)) if finite_hd.size else float("nan"),
        "median_hd95_mm": float(np.median(finite_hd)) if finite_hd.size else float("nan"),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def main():
    p = argparse.ArgumentParser(
        description="Leakage-safe OOF Final62-vs-Final72 benchmark on hard_dataset/v1"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--image-suffix", default=IMAGE_SUFFIX)
    p.add_argument("--label-suffix", default=LABEL_SUFFIX)
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--final62-cv-dir", default=str(DEFAULT_FINAL62_CV))
    p.add_argument("--final72-cv-dir", default=str(DEFAULT_FINAL72_CV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--expected-count", type=int, default=None)
    args = p.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    final62_cv = Path(args.final62_cv_dir)
    final72_cv = Path(args.final72_cv_dir)
    output_dir = Path(args.output_dir)

    images = collect_by_id(image_dir, args.image_suffix)
    labels = collect_by_id(gt_dir, args.label_suffix)
    image_ids = set(images)
    label_ids = set(labels)
    missing_gt = sorted(image_ids - label_ids)
    missing_image = sorted(label_ids - image_ids)
    if missing_gt or missing_image:
        raise RuntimeError(
            "Hard-v1 image/GT IDs do not match exactly.\n"
            f"Images without GT ({len(missing_gt)}): {missing_gt}\n"
            f"GT without image ({len(missing_image)}): {missing_image}"
        )
    hard_ids = sorted(image_ids)
    if not hard_ids:
        raise RuntimeError("No hard-set image/GT pairs found")
    if args.expected_count is not None and len(hard_ids) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} hard cases, found {len(hard_ids)}: {hard_ids}"
        )

    source_manifest = read_json(Path(args.source_manifest))
    fold_by_id = build_fold_map(source_manifest)
    not_oof = sorted(set(hard_ids) - set(fold_by_id))
    if not_oof:
        raise RuntimeError(
            "This benchmark only permits cases from the frozen original47 OOF cohort. "
            "These hard-v1 IDs are not in that cohort: " + ", ".join(not_oof)
        )

    # Direct hard-set mapping. cv.evaluate_fold normally discovers cases from config.data_dir;
    # overriding collection here ensures it reads only the supplied hard-v1 image and GT files.
    hard_cases = [
        {"id": case_id, "image": str(images[case_id]), "label": str(labels[case_id])}
        for case_id in hard_ids
    ]
    original_collect_cases = cv.collect_cases
    cv.collect_cases = lambda _config: list(hard_cases)

    # The evaluation transform/model settings must match the frozen DynUNet CV recipe.
    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=100)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary segmentation num_classes=1, got {config.num_classes}")

    print("=" * 124)
    print("HARD_DATASET/V1 — LEAKAGE-SAFE OOF FINAL62 vs FINAL72")
    print(f"Cases:                  {len(hard_ids)}")
    print(f"Images:                 {image_dir}")
    print(f"GT:                     {gt_dir}")
    print(f"Final62 CV:             {final62_cv}")
    print(f"Final72 CV:             {final72_cv}")
    print("Prediction:              Student+EMA 50/50 probability ensemble")
    print(f"Threshold:               {THRESHOLD:.2f} fixed")
    print("Leakage guard:           each case uses ONLY its frozen held-out fold checkpoint")
    print("Post-processing:         RAW / no LCC")
    print("=" * 124)

    final62_rows = []
    final72_rows = []
    try:
        for fold in range(5):
            fold_ids = sorted(x for x in hard_ids if fold_by_id[x] == fold)
            if not fold_ids:
                continue
            ck62 = checkpoint_for(final62_cv, fold)
            ck72 = checkpoint_for(final72_cv, fold)
            print(f"\nFold {fold}: {len(fold_ids)} hard case(s)")
            print(f"  Final62 checkpoint: {ck62}")
            print(f"  Final72 checkpoint: {ck72}")

            rows62 = cv.evaluate_fold(config, fold_ids, ck62, "ensemble", THRESHOLD)
            rows72 = cv.evaluate_fold(config, fold_ids, ck72, "ensemble", THRESHOLD)
            for row in rows62:
                row = r2.enrich_volume_metrics(row)
                row.update(fold=fold, model="FINAL62", threshold=THRESHOLD)
                final62_rows.append(row)
            for row in rows72:
                row = r2.enrich_volume_metrics(row)
                row.update(fold=fold, model="FINAL72", threshold=THRESHOLD)
                final72_rows.append(row)
    finally:
        cv.collect_cases = original_collect_cases

    by62 = {str(x["case_id"]): x for x in final62_rows}
    by72 = {str(x["case_id"]): x for x in final72_rows}
    if set(by62) != set(hard_ids) or set(by72) != set(hard_ids):
        raise RuntimeError("Evaluation did not return exactly one Final62 and Final72 row per hard case")

    paired = []
    for case_id in hard_ids:
        a = by62[case_id]
        b = by72[case_id]
        paired.append({
            "case_id": case_id,
            "fold": int(fold_by_id[case_id]),
            "final62_dice": float(a["dice"]),
            "final72_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "final62_precision": float(a["precision"]),
            "final72_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "final62_recall": float(a["recall"]),
            "final72_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "final62_signed_rve_pct": float(a["signed_rve"]),
            "final72_signed_rve_pct": float(b["signed_rve"]),
            "final62_abs_rve_pct": abs(float(a["signed_rve"])),
            "final72_abs_rve_pct": abs(float(b["signed_rve"])),
            "final62_hd95_mm": float(a["hd95"]),
            "final72_hd95_mm": float(b["hd95"]),
        })

    s62 = summarize(final62_rows)
    s72 = summarize(final72_rows)
    deltas = np.asarray([x["delta_dice"] for x in paired], dtype=float)
    summary = {
        "version": "hard_v1_oof_final62_vs_final72_v1",
        "n_cases": len(hard_ids),
        "threshold": THRESHOLD,
        "prediction": "raw Student+EMA 50/50 probability ensemble",
        "fold_rule": "Each hard case evaluated only by its frozen original47 held-out fold checkpoint",
        "final62": s62,
        "final72": s72,
        "delta": {
            "mean_dice": s72["mean_dice"] - s62["mean_dice"],
            "median_dice": s72["median_dice"] - s62["median_dice"],
            "mean_precision": s72["mean_precision"] - s62["mean_precision"],
            "mean_recall": s72["mean_recall"] - s62["mean_recall"],
            "median_abs_rve_pct": s72["median_abs_rve_pct"] - s62["median_abs_rve_pct"],
            "dice_lt_0p70": s72["dice_lt_0p70"] - s62["dice_lt_0p70"],
            "dice_lt_0p50": s72["dice_lt_0p50"] - s62["dice_lt_0p50"],
            "dice_ge_0p80": s72["dice_ge_0p80"] - s62["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(deltas > 1e-6)),
            "worsened": int(np.sum(deltas < -1e-6)),
            "unchanged": int(np.sum(np.abs(deltas) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(deltas >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(deltas <= -0.05)),
        },
        "case_folds": {case_id: int(fold_by_id[case_id]) for case_id in hard_ids},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hard_v1_final62_rows.csv", final62_rows)
    write_csv(output_dir / "hard_v1_final72_rows.csv", final72_rows)
    write_csv(output_dir / "hard_v1_paired_comparison.csv", paired)
    (output_dir / "hard_v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("HARD-V1 OOF SUMMARY")
    print(f"{'metric':<25} {'Final62':>12} {'Final72':>12} {'delta':>12}")
    print("-" * 65)
    print(f"{'Mean Dice':<25} {s62['mean_dice']:>12.4f} {s72['mean_dice']:>12.4f} {s72['mean_dice']-s62['mean_dice']:>+12.4f}")
    print(f"{'Median Dice':<25} {s62['median_dice']:>12.4f} {s72['median_dice']:>12.4f} {s72['median_dice']-s62['median_dice']:>+12.4f}")
    print(f"{'Precision':<25} {s62['mean_precision']:>12.4f} {s72['mean_precision']:>12.4f} {s72['mean_precision']-s62['mean_precision']:>+12.4f}")
    print(f"{'Recall':<25} {s62['mean_recall']:>12.4f} {s72['mean_recall']:>12.4f} {s72['mean_recall']-s62['mean_recall']:>+12.4f}")
    print(f"{'Median |RVE| %':<25} {s62['median_abs_rve_pct']:>12.2f} {s72['median_abs_rve_pct']:>12.2f} {s72['median_abs_rve_pct']-s62['median_abs_rve_pct']:>+12.2f}")
    print(f"{'Dice <0.70':<25} {s62['dice_lt_0p70']:>12d} {s72['dice_lt_0p70']:>12d} {s72['dice_lt_0p70']-s62['dice_lt_0p70']:>+12d}")
    print(f"{'Dice <0.50':<25} {s62['dice_lt_0p50']:>12d} {s72['dice_lt_0p50']:>12d} {s72['dice_lt_0p50']-s62['dice_lt_0p50']:>+12d}")
    print(f"{'Dice >=0.80':<25} {s62['dice_ge_0p80']:>12d} {s72['dice_ge_0p80']:>12d} {s72['dice_ge_0p80']-s62['dice_ge_0p80']:>+12d}")
    print(
        "Case effects: "
        f"improved={summary['case_effects']['improved']} | "
        f"worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | "
        f"<=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print("\nCASE-LEVEL")
    for row in sorted(paired, key=lambda x: x["delta_dice"]):
        print(
            f"  fold={row['fold']} {row['case_id']}: "
            f"{row['final62_dice']:.4f} -> {row['final72_dice']:.4f} "
            f"({row['delta_dice']:+.4f})"
        )
    print(f"\nOutputs: {output_dir}")
    print("=" * 124)


if __name__ == "__main__":
    main()
