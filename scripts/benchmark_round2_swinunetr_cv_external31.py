#!/usr/bin/env python3
"""Evaluate already-trained Final62 SwinUNETR B1 CV checkpoints on frozen external31.

NO TRAINING is performed.

This is a fast external generalization screen for the five B1 CV checkpoints:
  experiments/round2_cv_62_translation12_swinunetr_fs24_v1/
    checkpoints/fold_{0..4}/best_checkpoint.pth

The script reuses the established strict native-grid Final62 external31 benchmark and changes
only the network factory so those checkpoints are reconstructed as the controlled B1 model:
  SwinUNETR, in_channels=1, out_channels=1, feature_size=24, use_checkpoint=True.

For each fold it reports STUDENT, EMA and the 50/50 probability ENSEMBLE at threshold 0.50.
It then aggregates the five fold-level external31 summaries.

Important interpretation
------------------------
Each B1 CV checkpoint was trained with its original source held-out fold excluded, while the
15 post-source HUMAN_GOLD cases were train-only. Therefore this is a SCREENING result and is
not an apples-to-apples replacement for the final all-62 DynUNet model trained on all 62 labels.
A final all-62 SwinUNETR model should be trained only after this screen if the result remains
promising.

Example
-------
python scripts/benchmark_round2_swinunetr_cv_external31.py \
  --config config_resize128.yaml \
  --gt-dir /path/to/external31/labels \
  --gpu 1
"""

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path


def _consume_scalar(argv, name):
    cleaned = [argv[0]]
    value = None
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


# Fix CUDA visibility before importing torch/MONAI.
GPU_RAW, CLEAN_ARGV = _consume_scalar(sys.argv, "--gpu")
if GPU_RAW is not None:
    if not GPU_RAW.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU_RAW!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_RAW
sys.argv = CLEAN_ARGV

import numpy as np  # noqa: E402
from monai.networks.nets import SwinUNETR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_supervised_cv as cv  # noqa: E402
import scripts.benchmark_final62_student_ema_ensemble_external31 as external_bench  # noqa: E402

FEATURE_SIZE = 24
CV_DIR = Path("experiments/round2_cv_62_translation12_swinunetr_fs24_v1")
DEFAULT_POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
DEFAULT_ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/external31_round2_swinunetr_cv_screen")
MODE_ORDER = {"STUDENT": 0, "EMA": 1, "ENSEMBLE": 2}

ORIGINAL_BUILD_NETWORK = cv.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_b1_network(backbone: str, num_classes: int, dropout: float):
    if backbone != "swinunetr_b1_fs24":
        return ORIGINAL_BUILD_NETWORK(backbone, num_classes, dropout)
    if int(num_classes) != 1:
        raise RuntimeError(f"B1 requires binary num_classes=1, got {num_classes}")

    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "feature_size": FEATURE_SIZE,
        "use_checkpoint": True,
    }
    signature = inspect.signature(SwinUNETR.__init__)
    if "spatial_dims" in signature.parameters:
        kwargs["spatial_dims"] = 3
    if "img_size" in signature.parameters:
        param = signature.parameters["img_size"]
        if param.default is inspect.Parameter.empty:
            kwargs["img_size"] = (128, 128, 128)
    return SwinUNETR(**kwargs)


def apply_b1(config, resize_size, epochs):
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {config.num_classes}")
    if int(resize_size) != 128:
        raise RuntimeError(f"Controlled B1 external screen is frozen at resize128, got {resize_size}")
    config.unet_backbone = "swinunetr_b1_fs24"
    config.swinunetr_feature_size = FEATURE_SIZE
    config.use_gradient_checkpointing = True


# The strict benchmark and load_models() both resolve these attributes from the shared cv module.
cv.build_network = build_b1_network
cv.apply_baseline = apply_b1


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


def parse_folds(value):
    if str(value).lower() == "all":
        return list(range(5))
    try:
        idx = int(value)
    except ValueError as exc:
        raise ValueError("--fold must be 0..4 or all") from exc
    if idx not in range(5):
        raise ValueError("--fold must be 0..4 or all")
    return [idx]


def checkpoint_for_fold(fold):
    return CV_DIR / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"


def benchmark_fold(args, fold):
    checkpoint = checkpoint_for_fold(fold)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing B1 fold checkpoint: {checkpoint}\n"
            "This script does not train missing folds. Complete SwinUNETR CV first."
        )

    fold_output = Path(args.output_dir) / f"fold_{fold}"
    bench_argv = [
        "benchmark_final62_student_ema_ensemble_external31.py",
        "--config", args.config,
        "--checkpoint", str(checkpoint),
        "--pool-manifest", args.pool_manifest,
        "--round2-audit", args.round2_audit,
        "--output-dir", str(fold_output),
        "--expected-count", "31",
        "--resize-size", "128",
        "--threshold", "0.50",
    ]
    for gt_dir in args.gt_dir:
        bench_argv.extend(["--gt-dir", gt_dir])

    print("\n" + "=" * 124)
    print(f"SWINUNETR B1 EXTERNAL31 SCREEN | FOLD {fold}")
    print(f"Checkpoint: {checkpoint}")
    print("Architecture: SwinUNETR fs24 | binary output | checkpointing-compatible reconstruction")
    print("Training: NONE | threshold: 0.50 | strict native-grid inversion")
    print("=" * 124)

    previous = list(sys.argv)
    try:
        sys.argv = bench_argv
        external_bench.main()
    finally:
        sys.argv = previous

    summary = read_csv(fold_output / "external31_inference_mode_summary.csv")
    for row in summary:
        row["fold"] = int(fold)
        row["architecture"] = "B1_SWINUNETR_FS24"
        row["checkpoint"] = str(checkpoint)
    return summary


def aggregate(fold_rows):
    result = []
    for mode in ["STUDENT", "EMA", "ENSEMBLE"]:
        subset = [r for r in fold_rows if r["mode"] == mode]
        if not subset:
            continue

        def values(key):
            return np.asarray([float(r[key]) for r in subset], dtype=float)

        mean_dice = values("mean_dice")
        result.append({
            "architecture": "B1_SWINUNETR_FS24",
            "mode": mode,
            "n_folds": len(subset),
            "external_cases_per_fold": int(float(subset[0]["n"])),
            "mean_of_fold_mean_dice": float(np.mean(mean_dice)),
            "std_of_fold_mean_dice": float(np.std(mean_dice)),
            "min_fold_mean_dice": float(np.min(mean_dice)),
            "max_fold_mean_dice": float(np.max(mean_dice)),
            "mean_of_fold_median_dice": float(np.mean(values("median_dice"))),
            "mean_precision": float(np.mean(values("mean_precision"))),
            "mean_recall": float(np.mean(values("mean_recall"))),
            "mean_median_signed_rve_pct": float(np.mean(values("median_signed_rve_pct"))),
            "mean_median_abs_rve_pct": float(np.mean(values("median_abs_rve_pct"))),
            "mean_overseg_gt_20pct": float(np.mean(values("overseg_gt_20pct"))),
            "mean_overseg_gt_50pct": float(np.mean(values("overseg_gt_50pct"))),
            "mean_underseg_lt_minus20pct": float(np.mean(values("underseg_lt_minus20pct"))),
            "mean_failures_dice_lt_070": float(np.mean(values("failures_dice_lt_070"))),
            "min_failures_dice_lt_070": int(np.min(values("failures_dice_lt_070"))),
            "max_failures_dice_lt_070": int(np.max(values("failures_dice_lt_070"))),
            "mean_high_quality_dice_gte_080": float(np.mean(values("high_quality_dice_gte_080"))),
        })
    result.sort(key=lambda r: MODE_ORDER[r["mode"]])
    return result


def print_summary(rows):
    print("\n" + "=" * 140)
    print("EXTERNAL31 SWINUNETR B1 CV SCREEN — EXISTING CHECKPOINTS ONLY")
    print("=" * 140)
    print(
        f"{'mode':<10} {'folds':>5} {'meanDice':>9} {'sd':>7} {'median':>9} "
        f"{'precision':>10} {'recall':>8} {'med|RVE|':>10} {'mean<.70':>10} {'mean>=.80':>10}"
    )
    for r in rows:
        print(
            f"{r['mode']:<10} {int(r['n_folds']):>5} "
            f"{float(r['mean_of_fold_mean_dice']):>9.4f} "
            f"{float(r['std_of_fold_mean_dice']):>7.4f} "
            f"{float(r['mean_of_fold_median_dice']):>9.4f} "
            f"{float(r['mean_precision']):>10.4f} "
            f"{float(r['mean_recall']):>8.4f} "
            f"{float(r['mean_median_abs_rve_pct']):>9.2f}% "
            f"{float(r['mean_failures_dice_lt_070']):>10.2f} "
            f"{float(r['mean_high_quality_dice_gte_080']):>10.2f}"
        )
    print("=" * 140)
    print("SCREENING ONLY: fold models were not trained on all 62 labels.")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate existing B1 SwinUNETR CV folds on frozen external31; no training"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True, help="External31 GT root; repeat if needed")
    p.add_argument("--fold", default="all", help="0..4 or all")
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = p.parse_args()

    try:
        folds = parse_folds(args.fold)
    except ValueError as exc:
        p.error(str(exc))

    for required in [Path(args.pool_manifest), Path(args.round2_audit)]:
        if not required.exists():
            raise FileNotFoundError(required)
    for fold in folds:
        if not checkpoint_for_fold(fold).exists():
            raise FileNotFoundError(checkpoint_for_fold(fold))

    print("=" * 124)
    print("FINAL62 B1 SWINUNETR — FROZEN EXTERNAL31 CV SCREEN")
    print(f"Folds:        {folds}")
    print("External n:   31")
    print("Training:     NONE")
    print("Architecture: SwinUNETR feature_size=24")
    print("Threshold:    0.50 fixed")
    print("Evaluation:   same strict native-grid benchmark as Final62")
    print("=" * 124)

    fold_rows = []
    for fold in folds:
        fold_rows.extend(benchmark_fold(args, fold))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary_path = output_dir / "swinunetr_external31_fold_summary.csv"
    write_csv(fold_summary_path, fold_rows)

    aggregate_rows = aggregate(fold_rows)
    aggregate_path = output_dir / "swinunetr_external31_cv_summary.csv"
    write_csv(aggregate_path, aggregate_rows)

    metadata = {
        "version": "final62_b1_swinunetr_cv_external31_screen_v1",
        "training_performed": False,
        "architecture": "SwinUNETR",
        "feature_size": FEATURE_SIZE,
        "binary_out_channels": 1,
        "folds": folds,
        "external_count": 31,
        "threshold": 0.50,
        "resize_size": [128, 128, 128],
        "evaluation": "strict native-grid benchmark via benchmark_final62_student_ema_ensemble_external31.py",
        "interpretation_warning": (
            "These are five CV checkpoints, not a final all-62 model. Each fold excludes its original "
            "held-out source cases from training. Use this as an external screening result only."
        ),
    }
    (output_dir / "swinunetr_external31_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print_summary(aggregate_rows)
    print(f"Fold summary: {fold_summary_path}")
    print(f"CV summary:   {aggregate_path}")


if __name__ == "__main__":
    main()
