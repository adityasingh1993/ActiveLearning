#!/usr/bin/env python3
"""Calibrate one SwinUNETR EMA threshold using INTERNAL OOF cases only.

Purpose
-------
The Final62 SwinUNETR external31 result showed a high-precision / low-recall pattern. This
script asks whether threshold=0.50 is already too conservative for SwinUNETR by using only the
47 original source cases in their exact held-out CV folds.

No training is performed. No external31 image/label is read. For each fold, the already-trained
B1 best checkpoint is loaded, the EMA teacher is run on that fold's original held-out IDs, the
probability map is inverted to the exact native image grid, and a deliberately small pre-defined
threshold grid is evaluated:

    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60

The locked threshold is chosen by highest mean native-grid OOF Dice. Ties are broken by:
1) fewer Dice<0.70 failures,
2) lower median absolute RVE,
3) smaller precision/recall imbalance,
4) closeness to 0.50.

Important interpretation note
-----------------------------
External31 has already been inspected in earlier Swin experiments, so this is a diagnostic
post-hoc calibration study, not a pristine prospective validation. The threshold VALUE itself,
however, is derived without reading external31.
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
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)

CV_DIR = Path("experiments/round2_cv_62_translation12_swinunetr_fs24_v1")
PLAN_PATH = CV_DIR / "round2_cv_plan.json"
OUTPUT_DIR = CV_DIR / "oof_threshold_calibration_ema_v1"
FEATURE_SIZE = 24
EXPECTED_OOF_CASES = 47
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
SOURCE = "EMA"

ORIGINAL_BUILD_NETWORK = trainer_module.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_swin_b1(backbone: str, num_classes: int, dropout: float):
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


def apply_swin(config, resize_size=128, epochs=1):
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {config.num_classes}")
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


def summarize(rows):
    out = []
    for threshold in THRESHOLDS:
        subset = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-8]
        if len(subset) != EXPECTED_OOF_CASES:
            raise RuntimeError(
                f"Threshold {threshold:.2f}: expected {EXPECTED_OOF_CASES} OOF rows, found {len(subset)}"
            )
        dice = np.asarray([float(r["dice"]) for r in subset], dtype=float)
        precision = np.asarray([float(r["precision"]) for r in subset], dtype=float)
        recall = np.asarray([float(r["recall"]) for r in subset], dtype=float)
        signed = np.asarray([float(r["signed_rve_pct"]) for r in subset], dtype=float)
        out.append({
            "source": SOURCE,
            "threshold": threshold,
            "n": len(subset),
            "mean_dice": float(np.mean(dice)),
            "std_dice": float(np.std(dice)),
            "median_dice": float(np.median(dice)),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "precision_recall_gap": float(abs(np.mean(precision) - np.mean(recall))),
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


def choose_threshold(summary_rows):
    # Primary objective is mean OOF Dice. Remaining terms are deterministic tie-breaks only.
    return sorted(
        summary_rows,
        key=lambda r: (
            -float(r["mean_dice"]),
            int(r["failures_dice_lt_070"]),
            float(r["median_abs_rve_pct"]),
            float(r["precision_recall_gap"]),
            abs(float(r["threshold"]) - 0.50),
            float(r["threshold"]),
        ),
    )[0]


def main():
    p = argparse.ArgumentParser(
        description="Choose a SwinUNETR EMA threshold from exact native-grid internal OOF predictions"
    )
    p.add_argument("--config", required=True)
    args = p.parse_args()

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing completed Swin CV plan: {PLAN_PATH}. Run all five B1 folds first."
        )

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Controlled B1 uses prototype student + EMA")
    apply_swin(config, resize_size=128, epochs=1)

    plan = read_json(PLAN_PATH)
    folds = sorted(plan.get("folds", []), key=lambda x: int(x["fold"]))
    if len(folds) != 5:
        raise RuntimeError(f"Expected five CV folds in {PLAN_PATH}, found {len(folds)}")

    held_out = [str(case_id) for fold in folds for case_id in fold.get("val_ids", [])]
    if len(held_out) != EXPECTED_OOF_CASES or len(set(held_out)) != EXPECTED_OOF_CASES:
        raise RuntimeError(
            f"OOF provenance must contain each of the original {EXPECTED_OOF_CASES} cases exactly once"
        )

    cases = {str(c["id"]): c for c in cv.collect_cases(config)}
    missing = sorted(set(held_out) - set(cases))
    if missing:
        raise RuntimeError(f"Current data folder is missing OOF cases: {missing}")

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

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 126)
    print("SWINUNETR B1 EMA — INTERNAL OOF NATIVE-GRID THRESHOLD CALIBRATION")
    print(f"OOF cases:      {EXPECTED_OOF_CASES} original source cases, each held out exactly once")
    print(f"CV directory:   {CV_DIR}")
    print(f"Source:         {SOURCE}")
    print(f"Thresholds:     {', '.join(f'{x:.2f}' for x in THRESHOLDS)}")
    print("Selection:      highest mean OOF Dice; deterministic safety/volume tie-breaks")
    print("External31 I/O: NONE")
    print("Geometry:       probabilities inverted to exact native image grid before thresholding")
    print("=" * 126)

    rows = []
    for fold_spec in folds:
        fold = int(fold_spec["fold"])
        val_ids = [str(x) for x in fold_spec["val_ids"]]
        checkpoint = CV_DIR / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if "teacher" not in state:
            raise RuntimeError(f"Fold {fold} checkpoint has no EMA teacher: {checkpoint}")

        teacher = cv.build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()

        items = [{"image": cases[case_id]["image"], "id": case_id} for case_id in val_ids]
        loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for index, batch in enumerate(loader, start=1):
                raw_id = batch["id"]
                case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
                image_t = batch["image"].to(device)

                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    logits = cv.main_prediction(inferer(image_t, teacher))
                    prob_t = torch.sigmoid(logits)

                native_prob = invert_probability_exact(
                    prob_t, batch, inverse_transform, index=0
                )
                source_img, prob_zyx = normalize_native_probability(
                    native_prob, cases[case_id]["image"]
                )
                gt = read_gt_binary(cases[case_id]["label"], source_img)

                dice_at_050 = None
                for threshold in THRESHOLDS:
                    metrics = binary_metrics(prob_zyx > threshold, gt)
                    rows.append({
                        "fold": fold,
                        "case_id": case_id,
                        "source": SOURCE,
                        "threshold": threshold,
                        "checkpoint": str(checkpoint),
                        "checkpoint_epoch": int(state.get("epoch", 0)),
                        **metrics,
                    })
                    if abs(threshold - 0.50) < 1e-8:
                        dice_at_050 = float(metrics["dice"])

                print(
                    f"[fold {fold} {index:2d}/{len(val_ids)}] {case_id} | EMA native Dice@0.50={dice_at_050:.4f}"
                )

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unique_cases = {str(r["case_id"]) for r in rows}
    if len(unique_cases) != EXPECTED_OOF_CASES:
        raise RuntimeError(f"Expected {EXPECTED_OOF_CASES} unique OOF cases, found {len(unique_cases)}")

    summary = summarize(rows)
    selected = choose_threshold(summary)
    baseline = next(r for r in summary if abs(float(r["threshold"]) - 0.50) < 1e-8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "oof_threshold_case_metrics.csv", rows)
    write_csv(OUTPUT_DIR / "oof_threshold_summary.csv", summary)

    lock = {
        "version": "swinunetr_b1_ema_oof_threshold_lock_v1",
        "architecture": "SwinUNETR",
        "feature_size": FEATURE_SIZE,
        "source": SOURCE,
        "n_oof_cases": EXPECTED_OOF_CASES,
        "threshold_grid": THRESHOLDS,
        "baseline_threshold": 0.50,
        "locked_threshold": float(selected["threshold"]),
        "selection_primary": "highest mean native-grid internal OOF Dice",
        "tie_breaks": [
            "fewer Dice<0.70 failures",
            "lower median absolute RVE",
            "smaller mean precision-recall gap",
            "threshold closer to 0.50",
        ],
        "baseline_metrics": baseline,
        "selected_metrics": selected,
        "delta_vs_0p50": {
            "mean_dice": float(selected["mean_dice"]) - float(baseline["mean_dice"]),
            "mean_precision": float(selected["mean_precision"]) - float(baseline["mean_precision"]),
            "mean_recall": float(selected["mean_recall"]) - float(baseline["mean_recall"]),
            "median_abs_rve_pct": float(selected["median_abs_rve_pct"]) - float(baseline["median_abs_rve_pct"]),
            "failures_dice_lt_070": int(selected["failures_dice_lt_070"]) - int(baseline["failures_dice_lt_070"]),
            "high_quality_dice_gte_080": int(selected["high_quality_dice_gte_080"]) - int(baseline["high_quality_dice_gte_080"]),
        },
        "selection_data": "original47 exact held-out OOF cases only",
        "probability_geometry": "exact native image grid after MONAI inversion",
        "external31_read_by_this_script": False,
        "interpretation_warning": (
            "External31 had already been inspected in prior Swin experiments before this follow-up was proposed. "
            "Therefore this is a post-hoc diagnostic calibration study, not a pristine prospective validation; "
            "the locked threshold value itself is selected without reading external31."
        ),
    }
    lock_path = OUTPUT_DIR / "locked_threshold.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print("\n" + "=" * 132)
    print("INTERNAL OOF EMA THRESHOLD RESULTS — NATIVE GRID")
    print("=" * 132)
    print(
        f"{'thr':>5} {'meanDice':>9} {'median':>8} {'precision':>10} {'recall':>8} "
        f"{'medSignedRVE':>13} {'med|RVE|':>10} {'<.70':>6} {'>=.80':>7}"
    )
    for row in summary:
        marker = "  <-- LOCK" if row is selected else ""
        print(
            f"{float(row['threshold']):>5.2f} {float(row['mean_dice']):>9.4f} "
            f"{float(row['median_dice']):>8.4f} {float(row['mean_precision']):>10.4f} "
            f"{float(row['mean_recall']):>8.4f} {float(row['median_signed_rve_pct']):>+12.2f}% "
            f"{float(row['median_abs_rve_pct']):>9.2f}% {int(row['failures_dice_lt_070']):>6d} "
            f"{int(row['high_quality_dice_gte_080']):>7d}{marker}"
        )
    print("=" * 132)
    print(f"Locked EMA threshold: {float(selected['threshold']):.2f}")
    print(
        f"Mean Dice: {float(baseline['mean_dice']):.4f} @0.50 -> {float(selected['mean_dice']):.4f} "
        f"({float(selected['mean_dice']) - float(baseline['mean_dice']):+.4f})"
    )
    print(
        f"Precision/Recall: {float(selected['mean_precision']):.4f} / {float(selected['mean_recall']):.4f} | "
        f"median |RVE|={float(selected['median_abs_rve_pct']):.2f}%"
    )
    print(f"Lock file: {lock_path}")
    print("Next step: evaluate this locked value on external31 without changing it.")


if __name__ == "__main__":
    main()
