#!/usr/bin/env python3
"""Evaluate the already-trained Round-2 augmentation CV checkpoints on external31.

NO TRAINING is performed by this script.

The two augmentation profiles currently exist as five CV checkpoints each:
  A1 appearance:
    experiments/round2_cv_62_translation12_aug_appearance_v1/checkpoints/fold_{0..4}/best_checkpoint.pth
  A2 mild affine:
    experiments/round2_cv_62_translation12_aug_mildaffine_v1/checkpoints/fold_{0..4}/best_checkpoint.pth

For each existing checkpoint this wrapper calls the established strict Final62 external31
benchmark. That benchmark uses deterministic inference preprocessing, exact native-grid MONAI
inversion, threshold 0.50, and the frozen external31 leakage guard. Student, EMA and 50/50
student/EMA ensemble are reported for each fold checkpoint.

Because these are CV models rather than a final all-62 model, the aggregate profile result is a
SCREENING result. Each fold model was trained with the original fold's held-out source cases
excluded, while the 15 post-source HUMAN_GOLD cases were train-only. Do not present the aggregate
as an apples-to-apples replacement for the final all-62 A0 model.

Outputs
-------
  <output-dir>/<profile>/fold_<k>/...          existing benchmark outputs
  <output-dir>/augmentation_external31_fold_summary.csv
  <output-dir>/augmentation_external31_profile_summary.csv
  <output-dir>/augmentation_external31_metadata.json

Example
-------
python scripts/benchmark_round2_augmentation_cv_external31.py \
  --config config_resize128.yaml \
  --gt-dir /path/to/external31/labels \
  --profile both \
  --gpu 1
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_final62_student_ema_ensemble_external31.py"

PROFILE_DIRS = {
    "appearance": Path("experiments/round2_cv_62_translation12_aug_appearance_v1"),
    "mild_affine": Path("experiments/round2_cv_62_translation12_aug_mildaffine_v1"),
}
PROFILE_LABELS = {
    "appearance": "A1_APPEARANCE",
    "mild_affine": "A2_MILD_AFFINE",
}

DEFAULT_POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
DEFAULT_ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/external31_round2_augmentation_cv_screen")
MODE_ORDER = {"STUDENT": 0, "EMA": 1, "ENSEMBLE": 2}


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
            writer.writerow({key: row.get(key, "") for key in fields})


def selected_profiles(value):
    if value == "both":
        return ["appearance", "mild_affine"]
    return [value]


def checkpoint_path(profile, fold):
    return PROFILE_DIRS[profile] / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"


def benchmark_one(args, profile, fold):
    checkpoint = checkpoint_path(profile, fold)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing trained checkpoint for {profile} fold {fold}: {checkpoint}\n"
            "This script never trains missing checkpoints. Complete the requested CV fold first."
        )

    fold_output = Path(args.output_dir) / profile / f"fold_{fold}"
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--config", args.config,
        "--checkpoint", str(checkpoint),
        "--pool-manifest", args.pool_manifest,
        "--round2-audit", args.round2_audit,
        "--output-dir", str(fold_output),
        "--expected-count", str(args.expected_count),
        "--resize-size", "128",
        "--threshold", "0.50",
    ]
    for gt_dir in args.gt_dir:
        cmd.extend(["--gt-dir", gt_dir])

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("\n" + "=" * 118)
    print(f"EXTERNAL31 SCREEN | {PROFILE_LABELS[profile]} | fold {fold}")
    print(f"Checkpoint: {checkpoint}")
    print("Training:   NONE (evaluation only)")
    print("=" * 118)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    summary_path = fold_output / "external31_inference_mode_summary.csv"
    rows = read_csv(summary_path)
    for row in rows:
        row["profile"] = PROFILE_LABELS[profile]
        row["profile_key"] = profile
        row["fold"] = fold
        row["checkpoint"] = str(checkpoint)
    return rows


def aggregate_profile(fold_rows):
    out = []
    profiles = sorted({r["profile_key"] for r in fold_rows})
    for profile in profiles:
        for mode in ["STUDENT", "EMA", "ENSEMBLE"]:
            subset = [r for r in fold_rows if r["profile_key"] == profile and r["mode"] == mode]
            if not subset:
                continue

            def vals(key):
                return np.asarray([float(r[key]) for r in subset], dtype=float)

            mean_dice = vals("mean_dice")
            median_dice = vals("median_dice")
            precision = vals("mean_precision")
            recall = vals("mean_recall")
            abs_rve = vals("median_abs_rve_pct")
            fail = vals("failures_dice_lt_070")
            hq = vals("high_quality_dice_gte_080")
            over20 = vals("overseg_gt_20pct")
            over50 = vals("overseg_gt_50pct")
            under20 = vals("underseg_lt_minus20pct")

            out.append({
                "profile": PROFILE_LABELS[profile],
                "mode": mode,
                "n_folds": len(subset),
                "external_cases_per_fold": int(float(subset[0]["n"])),
                "mean_of_fold_mean_dice": float(np.mean(mean_dice)),
                "std_of_fold_mean_dice": float(np.std(mean_dice)),
                "min_fold_mean_dice": float(np.min(mean_dice)),
                "max_fold_mean_dice": float(np.max(mean_dice)),
                "mean_of_fold_median_dice": float(np.mean(median_dice)),
                "mean_precision": float(np.mean(precision)),
                "mean_recall": float(np.mean(recall)),
                "mean_median_abs_rve_pct": float(np.mean(abs_rve)),
                "mean_failures_dice_lt_070": float(np.mean(fail)),
                "min_failures_dice_lt_070": int(np.min(fail)),
                "max_failures_dice_lt_070": int(np.max(fail)),
                "mean_high_quality_dice_gte_080": float(np.mean(hq)),
                "mean_overseg_gt_20pct": float(np.mean(over20)),
                "mean_overseg_gt_50pct": float(np.mean(over50)),
                "mean_underseg_lt_minus20pct": float(np.mean(under20)),
            })
    out.sort(key=lambda r: (r["profile"], MODE_ORDER[r["mode"]]))
    return out


def print_profile_summary(rows):
    print("\n" + "=" * 136)
    print("EXTERNAL31 AUGMENTATION CV SCREEN — EXISTING CHECKPOINTS ONLY")
    print("=" * 136)
    print(
        f"{'profile':<18} {'mode':<10} {'folds':>5} {'meanDice':>9} {'sd':>7} "
        f"{'precision':>10} {'recall':>8} {'med|RVE|':>10} {'mean<.70':>10} {'mean>=.80':>10}"
    )
    for r in rows:
        print(
            f"{r['profile']:<18} {r['mode']:<10} {int(r['n_folds']):>5} "
            f"{float(r['mean_of_fold_mean_dice']):>9.4f} {float(r['std_of_fold_mean_dice']):>7.4f} "
            f"{float(r['mean_precision']):>10.4f} {float(r['mean_recall']):>8.4f} "
            f"{float(r['mean_median_abs_rve_pct']):>9.2f}% "
            f"{float(r['mean_failures_dice_lt_070']):>10.2f} "
            f"{float(r['mean_high_quality_dice_gte_080']):>10.2f}"
        )
    print("=" * 136)
    print("NOTE: profile rows summarize five CV models; they are screening results, not final all-62 models.")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate existing A1/A2 Round-2 CV checkpoints on frozen external31; no training"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    p.add_argument("--profile", choices=["appearance", "mild_affine", "both"], default="both")
    p.add_argument("--fold", default="all", help="0..4 or all; default all")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    p.add_argument("--round2-audit", default=str(DEFAULT_ROUND2_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=31)
    args = p.parse_args()

    if args.gpu is not None and args.gpu < 0:
        p.error("--gpu must be >=0")
    if args.expected_count != 31:
        p.error("This screening runner is intentionally locked to the frozen external31 set")
    if args.fold == "all":
        folds = list(range(5))
    else:
        try:
            fold = int(args.fold)
        except ValueError:
            p.error("--fold must be 0..4 or all")
        if fold not in range(5):
            p.error("--fold must be 0..4 or all")
        folds = [fold]

    if not BENCHMARK_SCRIPT.exists():
        raise FileNotFoundError(BENCHMARK_SCRIPT)
    for required in [Path(args.pool_manifest), Path(args.round2_audit)]:
        if not required.exists():
            raise FileNotFoundError(required)

    profiles = selected_profiles(args.profile)
    print("=" * 118)
    print("ROUND-2 AUGMENTATION EXTERNAL31 SCREEN")
    print(f"Profiles:    {', '.join(PROFILE_LABELS[x] for x in profiles)}")
    print(f"Folds:       {folds}")
    print(f"External n:  {args.expected_count}")
    print("Training:    NONE")
    print("Evaluation:  existing strict native-grid benchmark, threshold 0.50")
    print("=" * 118)

    fold_rows = []
    for profile in profiles:
        for fold in folds:
            fold_rows.extend(benchmark_one(args, profile, fold))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary_path = output_dir / "augmentation_external31_fold_summary.csv"
    write_csv(fold_summary_path, fold_rows)

    profile_rows = aggregate_profile(fold_rows)
    profile_summary_path = output_dir / "augmentation_external31_profile_summary.csv"
    write_csv(profile_summary_path, profile_rows)

    metadata = {
        "version": "round2_augmentation_cv_external31_screen_v1",
        "training_performed": False,
        "profiles": [PROFILE_LABELS[x] for x in profiles],
        "folds": folds,
        "external_count": args.expected_count,
        "threshold": 0.50,
        "resize_size": [128, 128, 128],
        "evaluation": "strict native-grid benchmark via benchmark_final62_student_ema_ensemble_external31.py",
        "interpretation_warning": (
            "Each row summarizes CV checkpoints. Fold models exclude their original source held-out cases from training, "
            "so this is a screening comparison and not an apples-to-apples replacement for the final all-62 A0 model."
        ),
    }
    (output_dir / "augmentation_external31_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print_profile_summary(profile_rows)
    print(f"Fold summary:    {fold_summary_path}")
    print(f"Profile summary: {profile_summary_path}")


if __name__ == "__main__":
    main()
