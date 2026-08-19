#!/usr/bin/env python3
"""Audit historical vs freshly reproduced Final62 128^3 results.

The script checks frozen training metadata and compares external31 metrics case-by-case. It is
intended to answer whether the historical 0.7367 EMA baseline reproduces on the same current
code path used for the 160^3 experiment.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_OLD_TRAIN = Path("experiments/final_supervised_round2_62_translation12")
DEFAULT_NEW_TRAIN = Path("experiments/final_supervised_round2_62_translation12_resize128_repro")
DEFAULT_OLD_EVAL = Path("experiments/external31_final62_inference_modes")
DEFAULT_NEW_EVAL = Path("experiments/external31_final62_resize128_repro_inference_modes")


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nested_get(d, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def find_mode(rows, mode):
    for row in rows:
        if row.get("mode") == mode:
            return row
    raise RuntimeError(f"Missing mode {mode}")


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(float(row[key]))


def main():
    p = argparse.ArgumentParser(description="Compare historical and reproduced Final62 128^3 runs")
    p.add_argument("--old-train-dir", default=str(DEFAULT_OLD_TRAIN))
    p.add_argument("--new-train-dir", default=str(DEFAULT_NEW_TRAIN))
    p.add_argument("--old-eval-dir", default=str(DEFAULT_OLD_EVAL))
    p.add_argument("--new-eval-dir", default=str(DEFAULT_NEW_EVAL))
    args = p.parse_args()

    old_train = Path(args.old_train_dir)
    new_train = Path(args.new_train_dir)
    old_eval = Path(args.old_eval_dir)
    new_eval = Path(args.new_eval_dir)

    old_meta = read_json(old_train / "final_training_metadata.json")
    new_meta = read_json(new_train / "final_training_metadata.json")

    print("=" * 118)
    print("FINAL62 128^3 REPRODUCIBILITY AUDIT")
    print("=" * 118)

    if old_meta is None:
        print(f"Historical training metadata: MISSING ({old_train / 'final_training_metadata.json'})")
    if new_meta is None:
        print(f"Reproduction metadata:        MISSING ({new_train / 'final_training_metadata.json'})")

    metadata_mismatch = False
    if old_meta is not None and new_meta is not None:
        checks = [
            ("n_total_human_labels", ("n_total_human_labels",)),
            ("seed", ("seed",)),
            ("final_training_epochs", ("final_training_epochs",)),
            ("median_cv_best_epoch", ("median_cv_best_epoch",)),
            ("prediction_threshold", ("prediction_threshold",)),
            ("prediction_source", ("prediction_source",)),
            ("spatial_size", ("recipe", "spatial_size")),
            ("spacing", ("recipe", "spacing")),
            ("loss", ("recipe", "loss")),
            ("optimizer", ("recipe", "optimizer")),
            ("learning_rate", ("recipe", "learning_rate")),
            ("weight_decay", ("recipe", "weight_decay")),
            ("dropout", ("recipe", "dropout")),
            ("lambda_unsup", ("recipe", "lambda_unsup")),
            ("translation_voxels", ("recipe", "translation_voxels")),
        ]
        print("\nTRAINING METADATA")
        for label, path in checks:
            old_v = nested_get(old_meta, *path)
            new_v = nested_get(new_meta, *path)
            same = old_v == new_v
            metadata_mismatch |= not same
            print(f"  {label:<24} old={old_v!r:<24} new={new_v!r:<24} {'PASS' if same else 'DIFF'}")

        old_ids = old_meta.get("all_human_label_ids")
        new_ids = new_meta.get("all_human_label_ids")
        same_ids = old_ids == new_ids
        metadata_mismatch |= not same_ids
        print(f"  {'all_human_label_ids':<24} {'PASS' if same_ids else 'DIFF'}")

    old_ckpt = old_train / "checkpoints" / "final_checkpoint.pth"
    new_ckpt = new_train / "checkpoints" / "final_checkpoint.pth"
    old_hash = sha256(old_ckpt)
    new_hash = sha256(new_ckpt)
    print("\nCHECKPOINTS")
    print(f"  historical exists: {old_ckpt.exists()} | sha256={old_hash or 'N/A'}")
    print(f"  reproduced exists: {new_ckpt.exists()} | sha256={new_hash or 'N/A'}")
    if old_hash and new_hash:
        print(f"  byte-identical:     {old_hash == new_hash}")
        print("  Note: different hashes alone do not prove a bug; CUDA training can be numerically non-identical.")

    old_summary = read_csv(old_eval / "external31_inference_mode_summary.csv")
    new_summary = read_csv(new_eval / "external31_inference_mode_summary.csv")
    old_cases = read_csv(old_eval / "external31_inference_mode_case_metrics.csv")
    new_cases = read_csv(new_eval / "external31_inference_mode_case_metrics.csv")

    print("\nEXTERNAL31 SUMMARY")
    print(f"{'mode':<10} {'old mean':>10} {'new mean':>10} {'delta':>10} {'old <.70':>10} {'new <.70':>10} {'old >=.80':>10} {'new >=.80':>10}")
    mean_deltas = {}
    for mode in ["STUDENT", "EMA", "ENSEMBLE"]:
        old_r = find_mode(old_summary, mode)
        new_r = find_mode(new_summary, mode)
        delta = f(new_r, "mean_dice") - f(old_r, "mean_dice")
        mean_deltas[mode] = delta
        print(
            f"{mode:<10} {f(old_r, 'mean_dice'):10.4f} {f(new_r, 'mean_dice'):10.4f} {delta:+10.4f} "
            f"{i(old_r, 'failures_dice_lt_070'):10d} {i(new_r, 'failures_dice_lt_070'):10d} "
            f"{i(old_r, 'high_quality_dice_gte_080'):10d} {i(new_r, 'high_quality_dice_gte_080'):10d}"
        )

    old_map = {(r["case_id"], r["mode"]): r for r in old_cases}
    new_map = {(r["case_id"], r["mode"]): r for r in new_cases}
    if set(old_map) != set(new_map):
        missing_new = sorted(set(old_map) - set(new_map))
        extra_new = sorted(set(new_map) - set(old_map))
        raise RuntimeError(
            "Historical/reproduction external case-mode sets differ.\n"
            f"Missing in reproduction: {missing_new}\nExtra in reproduction: {extra_new}"
        )

    ema_deltas = []
    for key in sorted(old_map):
        if key[1] != "EMA":
            continue
        ema_deltas.append((key[0], f(new_map[key], "dice") - f(old_map[key], "dice")))
    abs_ema = [abs(x[1]) for x in ema_deltas]
    mean_abs_ema = sum(abs_ema) / len(abs_ema)
    max_case, max_delta = max(ema_deltas, key=lambda x: abs(x[1]))

    print("\nEMA CASE-LEVEL REPRODUCIBILITY")
    print(f"  mean absolute per-case Dice delta: {mean_abs_ema:.6f}")
    print(f"  largest absolute Dice delta:      {abs(max_delta):.6f} ({max_case}, signed {max_delta:+.6f})")

    ema_mean_delta = mean_deltas["EMA"]
    print("\nVERDICT")
    if metadata_mismatch:
        print("  CONFIG/PROVENANCE MISMATCH DETECTED. Resolve metadata differences before interpreting resolution.")
    elif abs(ema_mean_delta) <= 0.005 and mean_abs_ema <= 0.01:
        print("  PASS: historical 128^3 baseline reproduces closely on the current code path.")
        print("  The 160^3 degradation is therefore much more likely to be a real resolution effect.")
    elif abs(ema_mean_delta) <= 0.015:
        print("  CAUTION: same recipe, but reproduction drift is non-trivial. Inspect case-level deltas before drawing a strong conclusion.")
    else:
        print("  FAIL: 128^3 baseline did not reproduce. Do not attribute the 160^3 difference to resolution yet.")
        print("  Audit code/config/environment differences and stochasticity first.")


if __name__ == "__main__":
    main()
