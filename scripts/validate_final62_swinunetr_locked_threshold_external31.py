#!/usr/bin/env python3
"""Validate the INTERNAL-OOF-locked Final62 SwinUNETR EMA threshold on external31.

This script performs evaluation only. It reads the threshold selected by
calibrate_swinunetr_ema_oof_threshold.py, refuses any manual candidate-threshold override, and
compares the locked value with baseline 0.50 on the frozen external31 set.

The candidate threshold is never selected from external31 inside this script. However external31
had already been inspected in earlier Swin experiments, so interpret this as a diagnostic
post-hoc calibration check rather than a pristine prospective test.
"""

import csv
import inspect
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
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402
from monai.networks.nets import SwinUNETR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
import hassl.training.trainer as trainer_module  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    collect_gt,
    invert_probability_exact,
    normalize_native_probability,
    read_csv,
    read_gt_binary,
    resolve_validation_cases,
)

FEATURE_SIZE = 24
CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12_swinunetr_fs24_v1/"
    "checkpoints/final_checkpoint.pth"
)
LOCK_PATH = Path(
    "experiments/round2_cv_62_translation12_swinunetr_fs24_v1/"
    "oof_threshold_calibration_ema_v1/locked_threshold.json"
)
POOL_MANIFEST = Path("experiments/auto_label_pool_round1_raw_v1/auto_label_manifest.csv")
ROUND2_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
OUTPUT_DIR = Path("experiments/external31_final62_swinunetr_ema_oof_locked_threshold_v1")
EXPECTED_COUNT = 31
BASELINE_THRESHOLD = 0.50

ORIGINAL_BUILD_NETWORK = trainer_module.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_swin_b1(backbone: str, num_classes: int, dropout: float):
    if backbone != "swinunetr_b1_fs24":
        return ORIGINAL_BUILD_NETWORK(backbone, num_classes, dropout)
    if int(num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {num_classes}")
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


def apply_swin(config, resize_size=128, epochs=1):
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    config.unet_backbone = "swinunetr_b1_fs24"
    config.swinunetr_feature_size = FEATURE_SIZE
    config.use_gradient_checkpointing = True


trainer_module.build_network = build_swin_b1
cv.build_network = build_swin_b1


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def summarize(rows, thresholds):
    out = []
    for threshold in thresholds:
        subset = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-8]
        if len(subset) != EXPECTED_COUNT:
            raise RuntimeError(
                f"Threshold {threshold:.3f}: expected {EXPECTED_COUNT} cases, found {len(subset)}"
            )
        dice = np.asarray([float(r["dice"]) for r in subset], dtype=float)
        precision = np.asarray([float(r["precision"]) for r in subset], dtype=float)
        recall = np.asarray([float(r["recall"]) for r in subset], dtype=float)
        signed = np.asarray([float(r["signed_rve_pct"]) for r in subset], dtype=float)
        out.append({
            "source": "EMA",
            "threshold": threshold,
            "n": len(subset),
            "mean_dice": float(np.mean(dice)),
            "std_dice": float(np.std(dice)),
            "median_dice": float(np.median(dice)),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "mean_signed_rve_pct": float(np.mean(signed)),
            "median_signed_rve_pct": float(np.median(signed)),
            "median_abs_rve_pct": float(np.median(np.abs(signed))),
            "overseg_gt_20pct": int(np.sum(signed > 20.0)),
            "overseg_gt_50pct": int(np.sum(signed > 50.0)),
            "underseg_lt_minus20pct": int(np.sum(signed < -20.0)),
            "failures_dice_lt_070": int(np.sum(dice < 0.70)),
            "high_quality_dice_gte_080": int(np.sum(dice >= 0.80)),
        })
    return out


def main():
    p = argparse.ArgumentParser(
        description="Validate the internal-OOF-locked Final62 Swin EMA threshold on frozen external31"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--gt-dir", action="append", required=True)
    args = p.parse_args()

    for path in [CHECKPOINT, LOCK_PATH, POOL_MANIFEST, ROUND2_AUDIT]:
        if not path.exists():
            raise FileNotFoundError(path)

    lock = read_json(LOCK_PATH)
    if lock.get("version") != "swinunetr_b1_ema_oof_threshold_lock_v1":
        raise RuntimeError(f"Unexpected threshold lock version: {lock.get('version')!r}")
    if lock.get("source") != "EMA":
        raise RuntimeError(f"Expected EMA threshold lock, got {lock.get('source')!r}")
    if int(lock.get("n_oof_cases", 0)) != 47:
        raise RuntimeError("Threshold lock is not based on the exact 47-case OOF set")
    if lock.get("external31_read_by_this_script") is not False:
        raise RuntimeError("Threshold lock metadata does not assert internal-only selection")

    candidate = float(lock["locked_threshold"])
    if not 0.0 < candidate < 1.0:
        raise RuntimeError(f"Invalid locked threshold: {candidate}")
    thresholds = [BASELINE_THRESHOLD]
    if abs(candidate - BASELINE_THRESHOLD) > 1e-8:
        thresholds.append(candidate)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Final62 Swin B1 expects prototype student + EMA")
    apply_swin(config, resize_size=128, epochs=1)

    audit = read_json(ROUND2_AUDIT)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit metadata is not marked passing")
    training_ids = set(str(x) for x in audit.get("all_current_human_label_ids", []))

    gt_by_id = collect_gt(args.gt_dir, config.label_suffix)
    pool_rows = read_csv(POOL_MANIFEST)
    cases = resolve_validation_cases(pool_rows, gt_by_id, EXPECTED_COUNT)
    overlap = sorted({str(c["id"]) for c in cases} & training_ids)
    if overlap:
        raise RuntimeError(
            "EXTERNAL/TRAINING LEAKAGE: external31 overlaps Final62 training IDs: " + ", ".join(overlap)
        )
    cases_by_id = {str(c["id"]): c for c in cases}

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    items = [{"image": c["image"], "id": c["id"]} for c in cases]
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, CHECKPOINT, device)
    if teacher is None:
        raise RuntimeError("Final62 Swin checkpoint has no EMA teacher")
    del student
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 124)
    print("FINAL62 SWINUNETR EMA — OOF-LOCKED THRESHOLD EXTERNAL31 DIAGNOSTIC")
    print(f"Cases:                 {len(cases)}")
    print(f"Training overlap:      {len(overlap)}")
    print(f"Baseline threshold:    {BASELINE_THRESHOLD:.2f}")
    print(f"OOF-locked threshold:  {candidate:.2f}")
    print(f"Lock file:             {LOCK_PATH}")
    print("Threshold selection:   INTERNAL original47 OOF only; no retuning here")
    print("Source:                EMA only")
    print("Geometry:              exact native-grid probability inversion")
    print("=" * 124)

    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            raw_id = batch["id"]
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            case = cases_by_id[case_id]
            image_t = batch["image"].to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = cv.main_prediction(inferer(image_t, teacher))
                prob_t = torch.sigmoid(logits)

            native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
            source_img, prob_zyx = normalize_native_probability(native_prob, case["image"])
            gt = read_gt_binary(case["gt_path"], source_img)

            by_thr = {}
            for threshold in thresholds:
                metrics = binary_metrics(prob_zyx > threshold, gt)
                rows.append({
                    "case_id": case_id,
                    "source": "EMA",
                    "threshold": threshold,
                    **metrics,
                })
                by_thr[threshold] = metrics

            b = by_thr[BASELINE_THRESHOLD]["dice"]
            c = by_thr[candidate]["dice"] if candidate in by_thr else b
            print(
                f"[{index:2d}/{len(cases)}] {case_id} | Dice {BASELINE_THRESHOLD:.2f}={b:.4f} -> "
                f"{candidate:.2f}={c:.4f} ({c - b:+.4f})"
            )

    summary = summarize(rows, thresholds)
    base = next(r for r in summary if abs(float(r["threshold"]) - BASELINE_THRESHOLD) < 1e-8)
    cand = next(r for r in summary if abs(float(r["threshold"]) - candidate) < 1e-8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "external31_locked_threshold_case_metrics.csv", rows)
    write_csv(OUTPUT_DIR / "external31_locked_threshold_summary.csv", summary)

    metadata = {
        "version": "final62_swinunetr_ema_oof_locked_threshold_external31_v1",
        "checkpoint": str(CHECKPOINT),
        "threshold_lock": str(LOCK_PATH),
        "source": "EMA",
        "baseline_threshold": BASELINE_THRESHOLD,
        "candidate_threshold": candidate,
        "n_external_cases": len(cases),
        "training_overlap_count": len(overlap),
        "external_gt_usage": "evaluation_only",
        "candidate_selection": "read verbatim from internal-OOF lock file; no external optimization",
        "delta_candidate_minus_0p50": {
            "mean_dice": float(cand["mean_dice"]) - float(base["mean_dice"]),
            "mean_precision": float(cand["mean_precision"]) - float(base["mean_precision"]),
            "mean_recall": float(cand["mean_recall"]) - float(base["mean_recall"]),
            "median_abs_rve_pct": float(cand["median_abs_rve_pct"]) - float(base["median_abs_rve_pct"]),
            "failures_dice_lt_070": int(cand["failures_dice_lt_070"]) - int(base["failures_dice_lt_070"]),
            "high_quality_dice_gte_080": int(cand["high_quality_dice_gte_080"]) - int(base["high_quality_dice_gte_080"]),
        },
        "interpretation_warning": (
            "External31 was already inspected in prior Swin experiments before this follow-up. "
            "Treat this as a diagnostic calibration check, not a pristine prospective validation."
        ),
    }
    (OUTPUT_DIR / "external31_locked_threshold_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 130)
    print("FINAL62 SWIN EMA — EXTERNAL31 LOCKED-THRESHOLD RESULT")
    print("=" * 130)
    print(
        f"{'thr':>5} {'meanDice':>9} {'median':>8} {'precision':>10} {'recall':>8} "
        f"{'medSignedRVE':>13} {'med|RVE|':>10} {'<.70':>6} {'>=.80':>7}"
    )
    for row in summary:
        label = " baseline" if abs(float(row["threshold"]) - BASELINE_THRESHOLD) < 1e-8 else " locked"
        print(
            f"{float(row['threshold']):>5.2f} {float(row['mean_dice']):>9.4f} "
            f"{float(row['median_dice']):>8.4f} {float(row['mean_precision']):>10.4f} "
            f"{float(row['mean_recall']):>8.4f} {float(row['median_signed_rve_pct']):>+12.2f}% "
            f"{float(row['median_abs_rve_pct']):>9.2f}% {int(row['failures_dice_lt_070']):>6d} "
            f"{int(row['high_quality_dice_gte_080']):>7d}{label}"
        )
    print("=" * 130)
    print(
        f"Locked - 0.50: Dice {float(cand['mean_dice']) - float(base['mean_dice']):+.4f} | "
        f"Precision {float(cand['mean_precision']) - float(base['mean_precision']):+.4f} | "
        f"Recall {float(cand['mean_recall']) - float(base['mean_recall']):+.4f}"
    )
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
