#!/usr/bin/env python3
"""Audit the expanded labeled dataset and freeze a 5-fold CV manifest.

Run this before the 47-label supervised experiment. It intentionally does not use
<data_dir>/splits.json from the earlier 13-case diagnostic. Instead it:

- discovers every image/label pair visible to the supplied config
- verifies the expected labeled-case count (47 by default)
- rejects duplicate case IDs and empty model-space labels
- runs the same deterministic preprocessing audit used by the diagnostic pipeline
- writes per-case model/native statistics to CSV
- reports GT-size, target-location, intensity, and spacing distributions
- optionally compares a focus case (80d095 by default) with the expanded dataset
- creates a deterministic patient-grouped 5-fold manifest (seed 42 by default)
- refuses to silently change an existing manifest unless --regenerate-splits is used

Outputs default to:
  experiments/cv5_supervised_47_translation12/dataset_audit.csv
  experiments/cv5_supervised_47_translation12/audit_summary.json
  experiments/cv5_supervised_47_translation12/fold_summary.csv
  experiments/cv5_supervised_47_translation12/cv_splits.json

This script does not train a model.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
import scripts.audit_labeled_dataset as audit
import scripts.train_supervised_cv as cv


DEFAULT_OUTPUT_DIR = Path("experiments/cv5_supervised_47_translation12")
DEFAULT_FOCUS_CASE = "80d0955124466d9b82337e7a17a8a2b5de9f4ec9244be0daa6eeb6f5014989d6"


def finite_values(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def stats(rows, key):
    values = finite_values(rows, key)
    if values.size == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def percentile_rank(rows, key, value):
    values = finite_values(rows, key)
    if values.size == 0 or not math.isfinite(float(value)):
        return None
    return float(100.0 * np.mean(values <= float(value)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(summary):
    print("\nEXPANDED DATASET DISTRIBUTION")
    print("=" * 100)
    print(f"{'metric':<28} {'min':>12} {'p25':>12} {'median':>12} {'p75':>12} {'max':>12}")
    print("-" * 100)
    for key, values in summary.items():
        if not isinstance(values, dict) or "median" not in values:
            continue

        def fmt(v):
            return "nan" if v is None else f"{v:.6f}"

        print(
            f"{key:<28} {fmt(values['min']):>12} {fmt(values['p25']):>12} "
            f"{fmt(values['median']):>12} {fmt(values['p75']):>12} {fmt(values['max']):>12}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Audit 47 labeled cases and freeze the 5-fold manifest for translation-only CV"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-cases", type=int, default=47)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--focus-case", default=DEFAULT_FOCUS_CASE)
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Explicitly replace the frozen manifest. Do not use after training has started.",
    )
    args = parser.parse_args()

    if args.expected_cases < 1 or args.folds < 2:
        parser.error("expected-cases must be >=1 and folds must be >=2")

    config = HASSLConfig.from_yaml(args.config)
    cases = cv.collect_cases(config)
    case_ids = [case["id"] for case in cases]

    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate case IDs found: {duplicates}")
    if len(cases) != args.expected_cases:
        raise RuntimeError(
            f"Expected exactly {args.expected_cases} labeled cases, found {len(cases)}. "
            "Resolve missing/extra image-label pairs before freezing the 47-case CV split."
        )

    print("=" * 100)
    print("47-LABEL DATASET PREPARATION")
    print(f"Config:          {args.config}")
    print(f"Labeled cases:   {len(cases)}")
    print(f"CV folds:        {args.folds}")
    print(f"Seed:            {args.seed}")
    print(f"Output:          {args.output_dir}")
    print("=" * 100)

    native_t = audit._native_transform()
    spaced_t = audit._post_spacing_transform(config)
    model_t = get_base_transforms(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases)}] auditing {case['id']}")
        rows.append(audit._row_for_case(case, "all", native_t, spaced_t, model_t))

    empty = [row["case_id"] for row in rows if int(row["model_gt_vox"]) <= 0]
    if empty:
        raise RuntimeError(
            "Empty labels after deterministic model preprocessing: " + ", ".join(empty)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "dataset_audit.csv"
    write_csv(audit_path, rows)

    distribution_keys = [
        "native_gt_fraction",
        "post_spacing_gt_fraction",
        "model_gt_fraction",
        "centroid_d_norm",
        "centroid_h_norm",
        "centroid_w_norm",
        "native_intensity_p1",
        "native_intensity_p50",
        "native_intensity_p99",
        "model_intensity_p1",
        "model_intensity_p50",
        "model_intensity_p99",
        "native_spacing_x",
        "native_spacing_y",
        "native_spacing_z",
    ]
    distributions = {key: stats(rows, key) for key in distribution_keys}

    focus_summary = None
    focus = next((row for row in rows if row["case_id"] == args.focus_case), None)
    if focus is not None:
        focus_keys = [
            "model_gt_fraction",
            "centroid_d_norm",
            "centroid_h_norm",
            "centroid_w_norm",
            "model_intensity_p50",
            "model_intensity_p99",
        ]
        focus_summary = {
            "case_id": args.focus_case,
            "metrics": {
                key: {
                    "value": float(focus[key]),
                    "percentile_within_all_labels": percentile_rank(rows, key, focus[key]),
                    "dataset_min": distributions[key]["min"],
                    "dataset_median": distributions[key]["median"],
                    "dataset_max": distributions[key]["max"],
                }
                for key in focus_keys
            },
        }

    manifest_path = output_dir / "cv_splits.json"
    if manifest_path.exists() and not args.regenerate_splits:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cv.validate_manifest(manifest, cases, args.folds, manifest_path)
        print(f"\nReusing existing frozen manifest: {manifest_path}")
    else:
        manifest = cv.create_manifest(
            cases,
            args.folds,
            args.seed,
            getattr(config, "patient_id_regex", None),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        action = "Regenerated" if args.regenerate_splits else "Created"
        print(f"\n{action} frozen manifest: {manifest_path}")

    fold_rows = []
    for fold in manifest["folds"]:
        fold_rows.append({
            "fold": int(fold["fold"]),
            "train_cases": len(fold["train_ids"]),
            "held_out_cases": len(fold["val_ids"]),
            "held_out_case_ids": ";".join(fold["val_ids"]),
            "held_out_patients": ";".join(fold.get("val_patients", [])),
        })
    write_csv(output_dir / "fold_summary.csv", fold_rows)

    summary = {
        "expected_cases": args.expected_cases,
        "found_cases": len(cases),
        "empty_model_labels": empty,
        "folds": args.folds,
        "seed": args.seed,
        "patient_id_regex": getattr(config, "patient_id_regex", None),
        "distributions": distributions,
        "focus_case": focus_summary,
        "manifest": str(manifest_path),
        "audit_csv": str(audit_path),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print_distribution(distributions)
    print("\nFOLD BALANCE")
    print("=" * 72)
    for row in fold_rows:
        print(
            f"Fold {row['fold']}: train={row['train_cases']} | held-out={row['held_out_cases']}"
        )

    if focus_summary is not None:
        print(f"\nFOCUS CASE WITHIN EXPANDED {len(rows)}-LABEL DISTRIBUTION")
        print("=" * 88)
        for key, item in focus_summary["metrics"].items():
            print(
                f"{key:<24} value={item['value']:.6f} | "
                f"percentile={item['percentile_within_all_labels']:.1f}% | "
                f"range={item['dataset_min']:.6f}..{item['dataset_max']:.6f}"
            )
    else:
        print(f"\nFocus case {args.focus_case} is not present in the current 47 labeled cases.")

    print("\nQUALITY GATE: PASS")
    print(f"Audit CSV:       {audit_path}")
    print(f"Audit summary:   {output_dir / 'audit_summary.json'}")
    print(f"Fold summary:    {output_dir / 'fold_summary.csv'}")
    print(f"Frozen manifest: {manifest_path}")
    print("Do not regenerate this manifest after any fold training has started.")


if __name__ == "__main__":
    main()
