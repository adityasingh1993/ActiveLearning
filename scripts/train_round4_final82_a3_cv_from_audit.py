#!/usr/bin/env python3
"""Controlled Final82 A3 CV from the live central HUMAN_GOLD label directory.

Round-4 corrected masks are assumed to have already been copied directly into the central
``labels/`` directory used by HASSL. No Round-4 annotation-audit or promotion artifact is
required by this runner.

Before any training starts, the script performs an inline, read-only Final82 validation:
- the frozen original47 split manifest is unchanged,
- the prior Final72 audit contributes exactly 72 known HUMAN_GOLD IDs,
- ``round4_annotation_batch.csv`` contributes exactly 10 unique ANNOTATE IDs,
- the 10 Round-4 IDs do not overlap the prior 72,
- the live label directory contains exactly prior72 + Round4-10 = 82 labels,
- every live image/label pair is readable, non-empty, and geometry matched,
- all 35 post-original47 HUMAN_GOLD cases are TRAIN ONLY in every fold.

Evaluation remains the exact frozen original47 held-out cases in their original five folds.
Comparison is Final72-A3 -> Final82-A3; the only intended change is HUMAN_GOLD 72 -> 82.

Frozen A3 recipe
----------------
- DynUNet, binary, resize 128^3
- DiceCE, AdamW lr=1e-4, dropout=0, lambda_unsup=0
- paired translation +/-4 vox with p=0.5
- left/right flip after RAS on spatial axis 0 with p=0.5
- no AP/SI flip, rotation, scale, gamma, noise, elastic, or LCC
- random init seed42, 100 epochs
- raw Student+EMA 50/50 ensemble at threshold 0.50
- external31 never accessed
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv
import scripts.train_active_learning_round2_cv_from_audit as r2
import scripts.train_final72_screen_spatial_folds12 as spatial
from scripts.audit_round1_labels import audit_case, discover_round1_cases

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
BASELINE_A3 = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
FINAL72_AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
ROUND4_BATCH = Path("experiments/round4_active_a3_committee_v1/round4_annotation_batch.csv")
OUTPUT = Path("experiments/round4_cv_82_a3")

EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 72
EXPECTED_ROUND4 = 10
EXPECTED_TOTAL = 82

A3_VARIANT = {
    "name": "translation4_p05_lrflip_p05",
    "spatial_aug": True,
    "translate_voxels": 4.0,
    "translate_prob": 0.5,
    "lr_flip": True,
    "lr_flip_prob": 0.5,
}


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_live_final82(
    config,
    source_manifest_path: Path,
    final72_audit_path: Path,
    round4_batch_path: Path,
    output_dir: Path,
):
    """Validate the live central label directory and return controlled Final82 state."""
    prior = read_json(final72_audit_path)
    if not prior.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Prior Final72 audit is not marked passing")
    prior_ids = sorted(str(x) for x in prior.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR or len(set(prior_ids)) != EXPECTED_PRIOR:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_PRIOR} unique prior Final72 HUMAN_GOLD IDs, "
            f"found {len(set(prior_ids))}"
        )

    batch = r2.read_csv(round4_batch_path)
    if len(batch) != EXPECTED_ROUND4:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_ROUND4} Round4 selected rows, found {len(batch)}"
        )
    if "case_id" not in batch[0]:
        raise RuntimeError("Round4 batch is missing case_id")
    if "round4_state" not in batch[0]:
        raise RuntimeError("Round4 batch is missing round4_state")

    bad_states = [
        str(row.get("case_id", ""))
        for row in batch
        if str(row.get("round4_state", "")).strip().upper() != "ANNOTATE"
    ]
    if bad_states:
        raise RuntimeError(
            "Round4 annotation batch contains non-ANNOTATE rows: " + ", ".join(bad_states)
        )

    round4_ids = sorted(str(row["case_id"]).strip() for row in batch)
    if any(not x for x in round4_ids) or len(set(round4_ids)) != EXPECTED_ROUND4:
        raise RuntimeError("Round4 annotation batch contains empty or duplicate case IDs")

    overlap = sorted(set(prior_ids) & set(round4_ids))
    if overlap:
        raise RuntimeError(
            "Round4 selected IDs overlap prior Final72 HUMAN_GOLD: " + ", ".join(overlap)
        )

    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest_path)
    source_ids = sorted(str(x) for x in source_ids)
    if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
        raise RuntimeError("Live discovery does not preserve exact frozen original47 source IDs")

    current_ids = sorted(str(x) for x in by_id)
    expected_ids = sorted(set(prior_ids) | set(round4_ids))
    unexpected = sorted(set(current_ids) - set(expected_ids))
    missing = sorted(set(expected_ids) - set(current_ids))
    if unexpected or missing:
        raise RuntimeError(
            "Live label directory is not exact Final72 + Round4 selected10.\n"
            f"Unexpected labels ({len(unexpected)}): {unexpected}\n"
            f"Missing expected labels ({len(missing)}): {missing}"
        )
    if len(current_ids) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TOTAL} live HUMAN_GOLD labels, found {len(current_ids)}"
        )

    audit_rows = []
    failures = []
    source_set = set(source_ids)
    prior_set = set(prior_ids)
    round4_set = set(round4_ids)
    for case_id in current_ids:
        row = audit_case(by_id[case_id])
        if case_id in round4_set:
            status = "ROUND4_NEW_HUMAN_GOLD"
        elif case_id in prior_set:
            status = "PRIOR_FINAL72_HUMAN_GOLD"
        else:
            status = "UNEXPECTED"
        row = {"status": status, **row}
        audit_rows.append(row)
        if not int(row.get("audit_ok", 0)):
            failures.append(f"{case_id}: {row.get('audit_error', 'audit failed')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "final82_live_label_audit.csv"
    audit_json = output_dir / "final82_live_label_audit.json"
    r2.write_csv(audit_csv, audit_rows)

    audit_metadata = {
        "version": "final82_live_label_pretraining_audit_v1",
        "source_manifest": str(source_manifest_path),
        "prior_final72_audit": str(final72_audit_path),
        "round4_annotation_batch": str(round4_batch_path),
        "n_frozen_source": len(source_ids),
        "n_prior_final72": len(prior_ids),
        "n_round4_new": len(round4_ids),
        "n_total_human_gold": len(current_ids),
        "prior_final72_human_label_ids": prior_ids,
        "round4_new_human_label_ids": round4_ids,
        "all_current_human_label_ids": current_ids,
        "unexpected_label_ids": unexpected,
        "missing_expected_label_ids": missing,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "selection_provenance_enforced": True,
        "label_source": "live central label directory; Round4 masks copied there directly by user",
        "external31_access": False,
    }
    audit_json.write_text(json.dumps(audit_metadata, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError(
            "Final82 live label audit FAILED. Training will not start.\n"
            + "\n".join(failures[:20])
        )

    extra_ids = sorted(set(current_ids) - source_set)
    if len(extra_ids) != EXPECTED_TOTAL - EXPECTED_SOURCE:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TOTAL - EXPECTED_SOURCE} train-only HUMAN_GOLD cases, "
            f"found {len(extra_ids)}"
        )
    if set(round4_ids) - set(extra_ids):
        raise RuntimeError("A Round4 selected case is not in the train-only extra set")

    print("=" * 118)
    print("FINAL82 LIVE LABEL PRETRAINING AUDIT — PASS")
    print(f"Frozen original source:  {len(source_ids)}")
    print(f"Prior Final72 labels:    {len(prior_ids)}")
    print(f"Round4 new labels:       {len(round4_ids)}")
    print(f"Total HUMAN_GOLD:        {len(current_ids)}")
    print(f"Train-only extras:       {len(extra_ids)}")
    print("Geometry/non-empty:      PASS for all")
    print("Unexpected labels:       0")
    print("Missing expected labels: 0")
    print(f"Audit CSV:               {audit_csv}")
    print(f"Audit metadata:          {audit_json}")
    print("=" * 118)

    return prior_ids, round4_ids, current_ids, source_ids, by_id, extra_ids, audit_json


def compare_final72a3_final82(baseline_rows, final82_rows, output_dir: Path):
    old = {str(x["case_id"]): r2.enrich_volume_metrics(x) for x in baseline_rows}
    new = {str(x["case_id"]): r2.enrich_volume_metrics(x) for x in final82_rows}
    if len(old) != EXPECTED_SOURCE or len(new) != EXPECTED_SOURCE or set(old) != set(new):
        raise RuntimeError("Final72-A3/Final82 paired comparison must contain exact original47 IDs")

    paired = []
    for case_id in sorted(old):
        a, b = old[case_id], new[case_id]
        if int(a["fold"]) != int(b["fold"]):
            raise RuntimeError(f"Frozen fold changed for {case_id}")
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "final72_a3_dice": float(a["dice"]),
            "final82_a3_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "final72_a3_precision": float(a["precision"]),
            "final82_a3_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "final72_a3_recall": float(a["recall"]),
            "final82_a3_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "final72_a3_signed_rve": float(a["signed_rve"]),
            "final82_a3_signed_rve": float(b["signed_rve"]),
            "delta_signed_rve": float(b["signed_rve"]) - float(a["signed_rve"]),
            "final72_a3_hd95": float(a["hd95"]),
            "final82_a3_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
        })

    s0 = r2.summarize(list(old.values()))
    s1 = r2.summarize(list(new.values()))
    delta = np.asarray([x["delta_dice"] for x in paired], dtype=float)
    summary = {
        "version": "final82_a3_vs_final72_a3_fixed_original47_v1",
        "final72_a3": s0,
        "final82_a3": s1,
        "delta": {
            "mean_dice": s1["mean_dice"] - s0["mean_dice"],
            "mean_precision": s1["mean_precision"] - s0["mean_precision"],
            "mean_recall": s1["mean_recall"] - s0["mean_recall"],
            "median_signed_rve_pct": s1["median_signed_rve_pct"] - s0["median_signed_rve_pct"],
            "median_abs_rve_pct": s1["median_abs_rve_pct"] - s0["median_abs_rve_pct"],
            "dice_lt_0p70": s1["dice_lt_0p70"] - s0["dice_lt_0p70"],
            "dice_lt_0p50": s1["dice_lt_0p50"] - s0["dice_lt_0p50"],
            "dice_ge_0p80": s1["dice_ge_0p80"] - s0["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(delta > 1e-6)),
            "worsened": int(np.sum(delta < -1e-6)),
            "unchanged": int(np.sum(np.abs(delta) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(delta >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(delta <= -0.05)),
        },
        "interpretation": (
            "Paired internal development comparison on exact frozen original47 held-out cases. "
            "Architecture/recipe are A3 in both arms; intended change is 10 Round4 HUMAN_GOLD labels."
        ),
    }
    r2.write_csv(output_dir / "final82_a3_vs_final72_a3_case_comparison.csv", paired)
    (output_dir / "final82_a3_vs_final72_a3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("FINAL72-A3 vs FINAL82-A3 — FIXED ORIGINAL47 HELD-OUT COMPARISON")
    print(
        f"Mean Dice:          {s0['mean_dice']:.4f} -> {s1['mean_dice']:.4f} "
        f"({summary['delta']['mean_dice']:+.4f})"
    )
    print(f"Median Dice:        {s0['median_dice']:.4f} -> {s1['median_dice']:.4f}")
    print(
        f"Precision:          {s0['mean_precision']:.4f} -> {s1['mean_precision']:.4f} "
        f"({summary['delta']['mean_precision']:+.4f})"
    )
    print(
        f"Recall:             {s0['mean_recall']:.4f} -> {s1['mean_recall']:.4f} "
        f"({summary['delta']['mean_recall']:+.4f})"
    )
    print(
        f"Median signed RVE:  {s0['median_signed_rve_pct']:+.2f}% -> "
        f"{s1['median_signed_rve_pct']:+.2f}%"
    )
    print(
        f"Median |RVE|:       {s0['median_abs_rve_pct']:.2f}% -> "
        f"{s1['median_abs_rve_pct']:.2f}%"
    )
    print(f"Dice <0.70:         {s0['dice_lt_0p70']} -> {s1['dice_lt_0p70']}")
    print(f"Dice <0.50:         {s0['dice_lt_0p50']} -> {s1['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {s0['dice_ge_0p80']} -> {s1['dice_ge_0p80']}")
    print(
        "Case effects:       "
        f"improved={summary['case_effects']['improved']} | "
        f"worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | "
        f"<=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print("=" * 120)


def main():
    p = argparse.ArgumentParser(description="Controlled Final82 A3 CV from live central labels")
    p.add_argument("--config", required=True)
    p.add_argument("--final72-audit", default=str(FINAL72_AUDIT))
    p.add_argument("--round4-batch", default=str(ROUND4_BATCH))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--baseline-a3-dir", default=str(BASELINE_A3))
    p.add_argument("--output-dir", default=str(OUTPUT))
    p.add_argument("--fold", default="all", help="0..4 or all")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source_cv_dir = Path(args.source_cv_dir)
    baseline_dir = Path(args.baseline_a3_dir)
    output_dir = Path(args.output_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    baseline_results_path = baseline_dir / "cv_results.csv"

    source_manifest = read_json(source_manifest_path)
    source_manifest_ids = sorted(str(x) for x in source_manifest.get("all_case_ids", []))
    if len(source_manifest_ids) != EXPECTED_SOURCE or len(set(source_manifest_ids)) != EXPECTED_SOURCE:
        raise RuntimeError("Source CV manifest is not exact frozen original47")

    baseline_rows = r2.read_csv(baseline_results_path)
    if len({str(x["case_id"]) for x in baseline_rows}) != EXPECTED_SOURCE:
        raise RuntimeError("Final72 A3 baseline must contain exact 47 held-out source cases")
    if {int(x["fold"]) for x in baseline_rows} != set(range(5)):
        raise RuntimeError("Final72 A3 baseline does not contain all five frozen folds")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Controlled Final82 A3 requires prototype DynUNet Student+EMA mode")

    (
        prior_ids,
        round4_ids,
        current_ids,
        source_ids,
        _by_id,
        extra_ids,
        live_audit_json,
    ) = validate_live_final82(
        config=config,
        source_manifest_path=source_manifest_path,
        final72_audit_path=Path(args.final72_audit),
        round4_batch_path=Path(args.round4_batch),
        output_dir=output_dir,
    )

    source_set = set(source_ids)
    selected_folds = cv.parse_fold(args.fold, 5)

    fold_specs = []
    for original in source_manifest["folds"]:
        fold_idx = int(original["fold"])
        val_ids = sorted(str(x) for x in original["val_ids"])
        train_ids = sorted(set(str(x) for x in original["train_ids"]) | set(extra_ids))
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: train/validation overlap")
        if set(extra_ids) & set(val_ids):
            raise RuntimeError(f"Fold {fold_idx}: added HUMAN_GOLD leaked into held-out original47")
        fold_specs.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "original_train_ids": sorted(str(x) for x in original["train_ids"]),
            "prior_final72_train_only_ids": sorted(set(prior_ids) - source_set),
            "round4_train_only_ids": round4_ids,
        })

    spatial.ACTIVE_VARIANT = dict(A3_VARIANT)
    cv.spatial_aug_transform = spatial.screening_spatial_transform
    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=42,
        resize_size=128,
        epochs=100,
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source="ensemble",
        eval_threshold=0.50,
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=4.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(baseline_results_path),
    )
    cv.install_cv_loader_hook(runtime_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "round4_final82_a3_controlled_cv_live_labels_v2",
        "source_manifest": str(source_manifest_path),
        "baseline_final72_a3": str(baseline_dir),
        "prior_final72_audit": str(args.final72_audit),
        "round4_annotation_batch": str(args.round4_batch),
        "live_final82_audit": str(live_audit_json),
        "label_source": "live central label directory",
        "n_frozen_source": EXPECTED_SOURCE,
        "n_prior_final72": EXPECTED_PRIOR,
        "n_round4_new": EXPECTED_ROUND4,
        "n_total_human_gold": EXPECTED_TOTAL,
        "n_train_only_extra": len(extra_ids),
        "all_train_only_extra_ids": extra_ids,
        "round4_new_human_label_ids": round4_ids,
        "all_current_human_label_ids": current_ids,
        "folds": fold_specs,
        "recipe": {
            "architecture": "DynUNet",
            "epochs": 100,
            "resize_size": [128, 128, 128],
            "loss": "dice_ce",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip": True,
            "lr_flip_probability": 0.5,
            "lr_flip_axis_after_ras": 0,
            "rotation": "off",
            "scale": "off",
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "external31_access": False,
    }
    plan_path = output_dir / "round4_final82_a3_cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        if args.overwrite:
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        else:
            raise RuntimeError(
                f"Existing plan differs: {plan_path}. Use --overwrite intentionally or a fresh output directory."
            )
    elif not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    fold_map = {int(x["fold"]): x for x in fold_specs}
    print("=" * 118)
    print("ROUND-4 / FINAL82 A3 DYNUNET CONTROLLED CV — LIVE LABEL MODE")
    print(f"Prior HUMAN_GOLD:       {EXPECTED_PRIOR}")
    print(f"Round4 HUMAN_GOLD:      {EXPECTED_ROUND4}")
    print(f"Total HUMAN_GOLD:       {EXPECTED_TOTAL}")
    print(f"Train-only extras:      {len(extra_ids)}")
    print(f"Running folds:          {selected_folds}")
    print("Held-out evaluation:    exact original47 / exact frozen folds")
    print("Recipe:                 A3 unchanged; only HUMAN_GOLD 72 -> 82")
    print("A3 spatial:             translation +/-4 p=.5 + LR flip p=.5")
    print("Evaluation:             raw Student+EMA 50/50 @ .50, no LCC")
    print("Round4 label source:    live central labels/ directory")
    print("External31:             NOT ACCESSED")
    print("=" * 118)

    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    kept = [r for r in existing if int(r["fold"]) not in selected_folds]
    combined = kept + new_rows
    combined.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    cv.write_results(results_path, combined)

    completed_folds = {int(r["fold"]) for r in combined}
    if completed_folds == set(range(5)) and len({str(r["case_id"]) for r in combined}) == EXPECTED_SOURCE:
        compare_final72a3_final82(baseline_rows, combined, output_dir)
    else:
        print(
            f"Partial CV complete ({sorted(completed_folds)}); "
            "full paired summary waits for all five folds."
        )

    print(f"\nResults:    {results_path}")
    print(f"Plan:       {plan_path}")
    print(f"Live audit: {live_audit_json}")


if __name__ == "__main__":
    main()
