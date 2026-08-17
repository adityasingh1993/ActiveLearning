#!/usr/bin/env python3
"""Controlled Round-1 CV with label counts derived from passing audit metadata.

The original 47 held-out folds are reused exactly. Every newly audited human label is added
only to each fold's training set. Label counts and experiment naming come from the audit,
so the current 47 + 8 = 55 case round is represented correctly without hard-coded 56-label
metadata.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_active_learning_round1_cv as helpers
from scripts.audit_round1_labels import discover_round1_cases
from scripts.train_supervised_cv_47_translation import translation_only_transform


DEFAULT_SOURCE_CV_DIR = Path("experiments/cv5_supervised_47_translation12")
EXPECTED_SOURCE_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Train controlled Round-1 CV using the exact new-label set recorded by a passing audit"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", required=True)
    parser.add_argument("--fold", default="all", help="0..4 or all")
    parser.add_argument("--source-cv-dir", default=str(DEFAULT_SOURCE_CV_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--eval-source", choices=["student", "teacher", "ensemble"], default="ensemble")
    parser.add_argument("--eval-threshold", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >=1")
    if not 0.0 < args.eval_threshold < 1.0:
        parser.error("--eval-threshold must be in (0,1)")

    source_cv_dir = Path(args.source_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    baseline_results_path = source_cv_dir / "cv_results.csv"
    audit_metadata_path = Path(args.audit_metadata)

    source_manifest = read_json(source_manifest_path)
    baseline_rows = cv.read_results(baseline_results_path)
    if len(source_manifest.get("all_case_ids", [])) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Source CV manifest is not the frozen {EXPECTED_SOURCE_CASES}-case manifest")
    if len({str(row["case_id"]) for row in baseline_rows}) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Round-0 cv_results.csv does not contain {EXPECTED_SOURCE_CASES} unique cases")

    audit = read_json(audit_metadata_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Audit metadata does not record a passing label audit")

    new_ids = sorted(str(x) for x in audit.get("new_human_label_ids", []))
    expected_new = int(audit.get("expected_new_human_labels", len(new_ids)))
    if expected_new != len(new_ids):
        raise RuntimeError(
            "Audit metadata is internally inconsistent: "
            f"expected_new_human_labels={expected_new}, discovered={len(new_ids)}"
        )
    if expected_new < 1:
        raise RuntimeError("Round-1 audit contains no new human labels")

    expected_total = EXPECTED_SOURCE_CASES + expected_new
    reported_total = int(audit.get("n_current_valid_human_labels", expected_total))
    if reported_total != expected_total:
        raise RuntimeError(
            "Audit total does not equal frozen source + new labels: "
            f"reported={reported_total}, expected={expected_total}"
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"experiments/round1_cv_{expected_total}_translation12")
    )

    base_config = HASSLConfig.from_yaml(args.config)
    if base_config.compute_mode != "prototype":
        raise RuntimeError("Frozen Round-0 experiment used prototype student + EMA teacher")

    _, source_ids, by_id, current_new_ids = discover_round1_cases(base_config, source_manifest_path)
    if sorted(current_new_ids) != new_ids:
        raise RuntimeError(
            "Current label folder no longer matches the audited Round-1 label set. "
            f"Audited={new_ids} Current={sorted(current_new_ids)}. Re-run the audit before training."
        )
    if len(by_id) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total valid labels (47+{expected_new}), found {len(by_id)}"
        )

    try:
        selected_folds = cv.parse_fold(args.fold, 5)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    cv.spatial_aug_transform = translation_only_transform
    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=int(args.seed),
        resize_size=int(args.resize_size),
        epochs=int(args.epochs),
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source=args.eval_source,
        eval_threshold=float(args.eval_threshold),
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=12.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(baseline_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    fold_specs = []
    for original in source_manifest["folds"]:
        fold_idx = int(original["fold"])
        original_train = [str(x) for x in original["train_ids"]]
        original_val = [str(x) for x in original["val_ids"]]
        train_ids = sorted(set(original_train) | set(new_ids))
        val_ids = sorted(original_val)

        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: train/held-out overlap detected")
        if set(new_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: a new AL label leaked into held-out evaluation")
        if not set(original_train).issubset(train_ids):
            raise RuntimeError(f"Fold {fold_idx}: original training cases were lost")

        fold_specs.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "original_train_ids": sorted(original_train),
            "new_round1_train_ids": new_ids,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_label = f"round1_{expected_total}labels_translation12"
    plan = {
        "version": f"round1_cv_{expected_total}_translation12_v1",
        "source_cv_dir": str(source_cv_dir),
        "source_manifest": str(source_manifest_path),
        "round0_results": str(baseline_results_path),
        "audit_metadata": str(audit_metadata_path),
        "n_frozen_source_labels": len(source_ids),
        "n_new_human_labels": len(new_ids),
        "n_total_human_labels": len(by_id),
        "new_human_label_ids": new_ids,
        "selected_unlabeled_ids": list(audit.get("selected_unlabeled_ids", [])),
        "folds": fold_specs,
        "recipe": {
            "epochs": int(args.epochs),
            "resize_size": [int(args.resize_size)] * 3,
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "rotation": 0.0,
            "scale": 0.0,
            "flips": False,
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": args.eval_source,
            "eval_threshold": float(args.eval_threshold),
            "checkpoint_for_evaluation": "best_checkpoint.pth",
        },
        "comparison_rule": (
            "Exact original val_ids are reused. Every audited new human label is appended only to each fold's training IDs."
        ),
    }

    plan_path = output_dir / "round1_cv_plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise RuntimeError(
                f"Existing Round-1 plan differs from current provenance: {plan_path}. "
                "Use a fresh output directory rather than mixing experiments."
            )
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(fold["fold"]): fold for fold in fold_specs}
    print("=" * 104)
    print("ROUND-1 ACTIVE-LEARNING CONTROLLED CV")
    print(f"Frozen source labels:  {len(source_ids)}")
    print(f"New human labels:      {len(new_ids)}")
    print(f"Total human labels:    {len(by_id)}")
    print(f"Running folds:         {selected_folds}")
    print(f"New IDs in TRAIN ONLY: {', '.join(new_ids)}")
    print("Held-out IDs: exact original Round-0 folds; no new AL label is evaluated as held-out here.")
    print("Recipe: frozen Round-0 CV | 100 epochs default | translation +/-12 p=0.8 | ensemble @0.50")
    print(f"Plan: {plan_path}")
    print("=" * 104)

    new_rows = []
    for fold_idx in selected_folds:
        rows = cv.run_fold(runtime_args, fold_map[fold_idx], output_dir)
        for row in rows:
            row["experiment"] = experiment_label
        new_rows.extend(rows)

    results_path = output_dir / "cv_results.csv"
    existing_rows = cv.read_results(results_path)
    selected_set = set(selected_folds)
    existing_rows = [row for row in existing_rows if int(row["fold"]) not in selected_set]
    merged = existing_rows + new_rows
    merged.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    cv.write_results(results_path, merged)
    print(f"\nRound-1 results saved: {results_path}")

    completed_folds = {int(row["fold"]) for row in merged}
    if completed_folds == set(range(5)):
        if len({str(row["case_id"]) for row in merged}) != EXPECTED_SOURCE_CASES:
            raise RuntimeError("Completed Round-1 CV does not cover the exact original 47 held-out cases")
        cv.print_complete_summary(merged, 5)
        helpers.compare_to_round0(baseline_rows, merged, output_dir)
    else:
        print(f"Completed folds so far: {sorted(completed_folds)} / [0, 1, 2, 3, 4]")
        print("Run the remaining folds; the paired Round-0 comparison is emitted only after all five complete.")


if __name__ == "__main__":
    main()
