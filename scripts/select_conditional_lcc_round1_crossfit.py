#!/usr/bin/env python3
"""Cross-fit conditional-LCC threshold selection on Round-1 OOF case decisions.

For each original segmentation fold k:
  1. use the other four folds as development rows,
  2. choose RAW or one dominance threshold using only those development rows,
  3. apply that selected rule to fold k,
  4. aggregate the held-out decisions across all five folds.

Selection priority on the four development folds:
  - reject candidates that catastrophically zero any previously non-zero-Dice case,
  - maximize mean Dice,
  - then minimize Dice<0.70 failures,
  - then maximize mean recall,
  - then prefer the more conservative rule (RAW, otherwise larger threshold).

This avoids choosing one global threshold on all 47 cases and scoring it on those same cases.
It is still development evidence because the segmentation model itself was developed in this
project and the eight new AL cases were train-only in every fold.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DECISIONS = Path(
    "experiments/round1_conditional_lcc_oof/conditional_lcc_case_decisions.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/round1_conditional_lcc_crossfit")


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
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(float(row[key]))


def build_case_table(rows):
    thresholds = sorted({float(row["dominance_threshold"]) for row in rows})
    cases = {}
    for row in rows:
        case_id = str(row["case_id"])
        fold = i(row, "fold")
        threshold = float(row["dominance_threshold"])
        entry = cases.setdefault(
            case_id,
            {
                "fold": fold,
                "raw": {
                    "dice": f(row, "raw_dice"),
                    "precision": f(row, "raw_precision"),
                    "recall": f(row, "raw_recall"),
                    "rve": f(row, "raw_rve"),
                    "hd95": f(row, "raw_hd95"),
                },
                "thresholds": {},
            },
        )
        if entry["fold"] != fold:
            raise RuntimeError(f"Case {case_id} appears in multiple folds")
        entry["thresholds"][threshold] = {
            "apply_lcc": i(row, "apply_lcc"),
            "dice": f(row, "final_dice"),
            "precision": f(row, "final_precision"),
            "recall": f(row, "final_recall"),
            "rve": f(row, "final_rve"),
            "hd95": f(row, "final_hd95"),
            "delta_dice": f(row, "delta_dice"),
        }

    for case_id, entry in cases.items():
        missing = sorted(set(thresholds) - set(entry["thresholds"]))
        if missing:
            raise RuntimeError(f"Case {case_id} missing thresholds: {missing}")
    return cases, thresholds


def metrics_for_rule(entry, rule):
    if rule == "RAW":
        return entry["raw"], 0
    threshold = float(rule)
    chosen = entry["thresholds"][threshold]
    return chosen, int(chosen["apply_lcc"])


def evaluate_rule(cases, case_ids, rule):
    metrics = []
    applied = 0
    catastrophic = 0
    for case_id in case_ids:
        entry = cases[case_id]
        selected, applied_here = metrics_for_rule(entry, rule)
        raw = entry["raw"]
        metrics.append(selected)
        applied += applied_here
        if raw["dice"] > 1e-6 and selected["dice"] <= 1e-6:
            catastrophic += 1

    dice = np.asarray([m["dice"] for m in metrics], dtype=float)
    recall = np.asarray([m["recall"] for m in metrics], dtype=float)
    precision = np.asarray([m["precision"] for m in metrics], dtype=float)
    rve = np.asarray([m["rve"] for m in metrics], dtype=float)
    hd95 = np.asarray([m["hd95"] for m in metrics], dtype=float)
    finite_hd = hd95[np.isfinite(hd95)]
    return {
        "n": len(case_ids),
        "n_lcc_applied": int(applied),
        "n_catastrophic": int(catastrophic),
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_rve_pct": float(np.median(rve)),
        "median_hd95_mm": float(np.median(finite_hd)) if finite_hd.size else float("nan"),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def conservative_rank(rule):
    # RAW is most conservative. Among LCC rules, larger thresholds apply less often.
    return 2.0 if rule == "RAW" else float(rule)


def choose_rule(cases, development_ids, thresholds):
    candidates = ["RAW"] + [float(x) for x in thresholds]
    scored = []
    for rule in candidates:
        summary = evaluate_rule(cases, development_ids, rule)
        scored.append((rule, summary))

    safe = [(rule, summary) for rule, summary in scored if summary["n_catastrophic"] == 0]
    pool = safe if safe else scored
    pool.sort(
        key=lambda item: (
            item[1]["mean_dice"],
            -item[1]["dice_lt_0p70"],
            item[1]["mean_recall"],
            conservative_rank(item[0]),
        ),
        reverse=True,
    )
    return pool[0], scored


def summarize_final(rows):
    dice = np.asarray([float(row["final_dice"]) for row in rows], dtype=float)
    precision = np.asarray([float(row["final_precision"]) for row in rows], dtype=float)
    recall = np.asarray([float(row["final_recall"]) for row in rows], dtype=float)
    rve = np.asarray([float(row["final_rve"]) for row in rows], dtype=float)
    hd95 = np.asarray([float(row["final_hd95"]) for row in rows], dtype=float)
    finite_hd = hd95[np.isfinite(hd95)]
    return {
        "n_cases": len(rows),
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "median_rve_pct": float(np.median(rve)),
        "median_hd95_mm": float(np.median(finite_hd)) if finite_hd.size else float("nan"),
        "dice_lt_0p20": int(np.sum(dice < 0.20)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_lt_0p60": int(np.sum(dice < 0.60)),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-fit Round-1 conditional-LCC threshold selection by original segmentation fold"
    )
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    output_dir = Path(args.output_dir)
    rows = read_csv(decisions_path)
    cases, thresholds = build_case_table(rows)
    folds = sorted({int(entry["fold"]) for entry in cases.values()})
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected original folds [0,1,2,3,4], found {folds}")
    if len(cases) != 47:
        raise RuntimeError(f"Expected 47 OOF cases, found {len(cases)}")

    all_ids = sorted(cases)
    fold_selection_rows = []
    final_rows = []

    print("=" * 112)
    print("ROUND-1 CROSS-FITTED CONDITIONAL-LCC POLICY")
    print("Each held-out fold receives a rule selected using only the other four folds.")
    print("Candidate rules: RAW, " + ", ".join(f"LCC@{x:.2f}" for x in thresholds))
    print("=" * 112)

    for heldout_fold in folds:
        heldout_ids = [case_id for case_id in all_ids if cases[case_id]["fold"] == heldout_fold]
        development_ids = [case_id for case_id in all_ids if cases[case_id]["fold"] != heldout_fold]
        (selected_rule, dev_summary), all_scores = choose_rule(cases, development_ids, thresholds)

        held_summary = evaluate_rule(cases, heldout_ids, selected_rule)
        fold_selection_rows.append({
            "heldout_fold": heldout_fold,
            "selected_rule": "RAW" if selected_rule == "RAW" else "CONDITIONAL_LCC",
            "selected_threshold": "" if selected_rule == "RAW" else float(selected_rule),
            "development_n": len(development_ids),
            "development_mean_dice": dev_summary["mean_dice"],
            "development_failures_lt_0p70": dev_summary["dice_lt_0p70"],
            "development_mean_recall": dev_summary["mean_recall"],
            "development_catastrophic": dev_summary["n_catastrophic"],
            "heldout_n": len(heldout_ids),
            "heldout_lcc_applied": held_summary["n_lcc_applied"],
            "heldout_mean_dice": held_summary["mean_dice"],
            "heldout_failures_lt_0p70": held_summary["dice_lt_0p70"],
        })

        rule_text = "RAW" if selected_rule == "RAW" else f"LCC@{float(selected_rule):.2f}"
        print(
            f"Fold {heldout_fold}: selected {rule_text:8s} on n={len(development_ids)} dev | "
            f"dev Dice={dev_summary['mean_dice']:.4f} failures={dev_summary['dice_lt_0p70']} | "
            f"held-out Dice={held_summary['mean_dice']:.4f} failures={held_summary['dice_lt_0p70']}"
        )

        for case_id in heldout_ids:
            entry = cases[case_id]
            selected, applied = metrics_for_rule(entry, selected_rule)
            raw = entry["raw"]
            final_rows.append({
                "fold": heldout_fold,
                "case_id": case_id,
                "selected_rule": "RAW" if selected_rule == "RAW" else "CONDITIONAL_LCC",
                "selected_threshold": "" if selected_rule == "RAW" else float(selected_rule),
                "lcc_applied": int(applied),
                "raw_dice": raw["dice"],
                "final_dice": selected["dice"],
                "delta_dice": selected["dice"] - raw["dice"],
                "raw_precision": raw["precision"],
                "final_precision": selected["precision"],
                "raw_recall": raw["recall"],
                "final_recall": selected["recall"],
                "raw_rve": raw["rve"],
                "final_rve": selected["rve"],
                "raw_hd95": raw["hd95"],
                "final_hd95": selected["hd95"],
            })

    final_rows.sort(key=lambda row: (int(row["fold"]), str(row["case_id"])))
    raw_rows = []
    for case_id in all_ids:
        raw = cases[case_id]["raw"]
        raw_rows.append({
            "final_dice": raw["dice"],
            "final_precision": raw["precision"],
            "final_recall": raw["recall"],
            "final_rve": raw["rve"],
            "final_hd95": raw["hd95"],
        })

    raw_summary = summarize_final(raw_rows)
    crossfit_summary = summarize_final(final_rows)
    deltas = np.asarray([float(row["delta_dice"]) for row in final_rows], dtype=float)
    summary = {
        "version": "round1_conditional_lcc_crossfit_v1",
        "source_decisions": str(decisions_path),
        "selection_method": (
            "For each held-out original fold: choose RAW or threshold on other four folds; zero-catastrophic first, "
            "then mean Dice, fewer Dice<0.70 failures, higher recall, more conservative rule."
        ),
        "raw": raw_summary,
        "crossfit": crossfit_summary,
        "delta": {
            "mean_dice": crossfit_summary["mean_dice"] - raw_summary["mean_dice"],
            "mean_precision": crossfit_summary["mean_precision"] - raw_summary["mean_precision"],
            "mean_recall": crossfit_summary["mean_recall"] - raw_summary["mean_recall"],
            "median_rve_pct": crossfit_summary["median_rve_pct"] - raw_summary["median_rve_pct"],
            "median_hd95_mm": crossfit_summary["median_hd95_mm"] - raw_summary["median_hd95_mm"],
            "dice_lt_0p70": crossfit_summary["dice_lt_0p70"] - raw_summary["dice_lt_0p70"],
            "dice_lt_0p50": crossfit_summary["dice_lt_0p50"] - raw_summary["dice_lt_0p50"],
            "dice_ge_0p80": crossfit_summary["dice_ge_0p80"] - raw_summary["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(deltas > 1e-6)),
            "worsened": int(np.sum(deltas < -1e-6)),
            "unchanged": int(np.sum(np.abs(deltas) <= 1e-6)),
            "improved_ge_0p05": int(np.sum(deltas >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(deltas <= -0.05)),
        },
        "warning": (
            "Cross-fitting removes direct threshold-selection leakage across the 47 original OOF cases, but this remains "
            "development evidence rather than an independent external validation set."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "fold_rule_selection.csv"
    case_path = output_dir / "crossfit_case_results.csv"
    summary_path = output_dir / "crossfit_summary.json"
    write_csv(fold_path, fold_selection_rows)
    write_csv(case_path, final_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("CROSS-FITTED ROUND-1 LCC RESULT")
    print(f"Mean Dice:      {raw_summary['mean_dice']:.4f} -> {crossfit_summary['mean_dice']:.4f} "
          f"({summary['delta']['mean_dice']:+.4f})")
    print(f"Precision:      {raw_summary['mean_precision']:.4f} -> {crossfit_summary['mean_precision']:.4f}")
    print(f"Recall:         {raw_summary['mean_recall']:.4f} -> {crossfit_summary['mean_recall']:.4f}")
    print(f"Median RVE:     {raw_summary['median_rve_pct']:.1f}% -> {crossfit_summary['median_rve_pct']:.1f}%")
    print(f"Median HD95:    {raw_summary['median_hd95_mm']:.2f} -> {crossfit_summary['median_hd95_mm']:.2f} mm")
    print(f"Dice <0.70:     {raw_summary['dice_lt_0p70']} -> {crossfit_summary['dice_lt_0p70']}")
    print(f"Dice <0.50:     {raw_summary['dice_lt_0p50']} -> {crossfit_summary['dice_lt_0p50']}")
    print(f"Dice >=0.80:    {raw_summary['dice_ge_0p80']} -> {crossfit_summary['dice_ge_0p80']}")
    print(
        f"Case effects:   improved={summary['case_effects']['improved']} | "
        f"worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | "
        f"<=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print(f"Fold rules:     {fold_path}")
    print(f"Case results:   {case_path}")
    print(f"Summary:        {summary_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
