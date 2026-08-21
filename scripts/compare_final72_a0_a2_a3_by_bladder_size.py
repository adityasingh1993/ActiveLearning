#!/usr/bin/env python3
"""Compare Final72 A0/A2/A3 OOF performance by fixed bladder-size group.

This is a diagnostic only. Size groups come from the already-computed Final72 HUMAN_GOLD
physical-volume terciles in:
    experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv

Models
------
A0 : Final72, translation +/-12 vox p=.8, no flip, DiceCE, 128^3
A2 : Final72, translation +/-4 vox p=.5, no flip, DiceCE, 128^3
A3 : Final72, translation +/-4 vox p=.5 + LR flip p=.5, DiceCE, 128^3

All comparisons are on the same frozen original47 held-out OOF cases. No external31 data is used.

Outputs
-------
experiments/final72_size_stratified_a0_a2_a3/
    case_level_joined.csv
    size_group_model_summary.csv
    paired_size_group_deltas.csv
    summary.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_SIZE_PROFILE = Path(
    "experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv"
)
DEFAULT_A0 = Path("experiments/round3_cv_72_translation12/cv_results.csv")
DEFAULT_A2 = Path("experiments/final72_screen_a2_translation4_p05/cv_results.csv")
DEFAULT_A3 = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05/cv_results.csv")
DEFAULT_OUTPUT = Path("experiments/final72_size_stratified_a0_a2_a3")
EXPECTED_OOF = 47
MODELS = ("A0", "A2", "A3")
SIZE_ORDER = ("SMALL", "MEDIUM", "LARGE")


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


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return float(default)


def normalize_size_group(row):
    for key in ("size_group", "bladder_size_group", "physical_volume_group"):
        value = str(row.get(key, "")).strip().upper()
        if value in SIZE_ORDER:
            return value
    raise RuntimeError(
        "Size profile has no recognized SMALL/MEDIUM/LARGE column. "
        "Expected one of: size_group, bladder_size_group, physical_volume_group"
    )


def index_metrics(rows, name):
    out = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id in out:
            raise RuntimeError(f"Duplicate {name} case_id: {case_id}")
        out[case_id] = row
    if len(out) != EXPECTED_OOF:
        raise RuntimeError(f"{name} must contain exact original47 OOF rows; found {len(out)}")
    return out


def summarize(rows):
    dice = np.asarray([f(r, "dice") for r in rows], dtype=float)
    precision = np.asarray([f(r, "precision") for r in rows], dtype=float)
    recall = np.asarray([f(r, "recall") for r in rows], dtype=float)
    hd95 = np.asarray([f(r, "hd95") for r in rows], dtype=float)
    rve = np.asarray([f(r, "rve") for r in rows], dtype=float)
    return {
        "n": len(rows),
        "mean_dice": float(np.nanmean(dice)),
        "median_dice": float(np.nanmedian(dice)),
        "mean_precision": float(np.nanmean(precision)),
        "mean_recall": float(np.nanmean(recall)),
        "mean_hd95_mm": float(np.nanmean(hd95)),
        "median_hd95_mm": float(np.nanmedian(hd95)),
        "median_abs_rve_pct": float(np.nanmedian(np.abs(rve))),
        "dice_lt_0p70": int(np.sum(dice < 0.70)),
        "dice_lt_0p50": int(np.sum(dice < 0.50)),
        "dice_ge_0p80": int(np.sum(dice >= 0.80)),
    }


def main():
    p = argparse.ArgumentParser(description="Size-stratified A0/A2/A3 Final72 OOF comparison")
    p.add_argument("--size-profile", default=str(DEFAULT_SIZE_PROFILE))
    p.add_argument("--a0-results", default=str(DEFAULT_A0))
    p.add_argument("--a2-results", default=str(DEFAULT_A2))
    p.add_argument("--a3-results", default=str(DEFAULT_A3))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    size_rows = read_csv(Path(args.size_profile))
    size_by_id = {}
    for row in size_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        size_by_id[case_id] = {
            **row,
            "size_group": normalize_size_group(row),
        }

    model_rows = {
        "A0": index_metrics(read_csv(Path(args.a0_results)), "A0"),
        "A2": index_metrics(read_csv(Path(args.a2_results)), "A2"),
        "A3": index_metrics(read_csv(Path(args.a3_results)), "A3"),
    }
    ids = set(model_rows["A0"])
    if set(model_rows["A2"]) != ids or set(model_rows["A3"]) != ids:
        raise RuntimeError("A0/A2/A3 do not contain the exact same original47 IDs")
    missing_size = sorted(ids - set(size_by_id))
    if missing_size:
        raise RuntimeError(f"Missing size profile for OOF cases: {missing_size}")

    joined = []
    for case_id in sorted(ids):
        size = size_by_id[case_id]
        for model in MODELS:
            row = model_rows[model][case_id]
            joined.append({
                "case_id": case_id,
                "fold": int(row["fold"]),
                "model": model,
                "size_group": size["size_group"],
                "physical_volume_ml": size.get("physical_volume_ml", size.get("bladder_volume_ml", "")),
                "native_fg_fraction": size.get("native_fg_fraction", ""),
                "post128_fg_fraction": size.get("post128_fg_fraction", size.get("post_resize_fg_fraction", "")),
                "dice": f(row, "dice"),
                "precision": f(row, "precision"),
                "recall": f(row, "recall"),
                "rve": f(row, "rve"),
                "hd95": f(row, "hd95"),
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_level_joined.csv", joined)

    summary_rows = []
    summary_map = {}
    for group in SIZE_ORDER:
        for model in MODELS:
            subset = [r for r in joined if r["size_group"] == group and r["model"] == model]
            s = summarize(subset)
            row = {"size_group": group, "model": model, **s}
            summary_rows.append(row)
            summary_map[(group, model)] = s
    write_csv(output_dir / "size_group_model_summary.csv", summary_rows)

    paired_rows = []
    for group in SIZE_ORDER:
        for candidate in ("A2", "A3"):
            base = summary_map[(group, "A0")]
            cand = summary_map[(group, candidate)]
            paired_rows.append({
                "size_group": group,
                "candidate": candidate,
                "n": cand["n"],
                "delta_mean_dice": cand["mean_dice"] - base["mean_dice"],
                "delta_mean_precision": cand["mean_precision"] - base["mean_precision"],
                "delta_mean_recall": cand["mean_recall"] - base["mean_recall"],
                "delta_mean_hd95_mm": cand["mean_hd95_mm"] - base["mean_hd95_mm"],
                "delta_median_abs_rve_pct": cand["median_abs_rve_pct"] - base["median_abs_rve_pct"],
                "delta_dice_lt_0p70": cand["dice_lt_0p70"] - base["dice_lt_0p70"],
                "delta_dice_lt_0p50": cand["dice_lt_0p50"] - base["dice_lt_0p50"],
                "delta_dice_ge_0p80": cand["dice_ge_0p80"] - base["dice_ge_0p80"],
            })
    write_csv(output_dir / "paired_size_group_deltas.csv", paired_rows)

    payload = {
        "version": "final72_a0_a2_a3_size_stratified_v1",
        "n_oof": EXPECTED_OOF,
        "models": {
            "A0": "translation +/-12 vox p=.8; no flip; DiceCE; 128^3",
            "A2": "translation +/-4 vox p=.5; no flip; DiceCE; 128^3",
            "A3": "translation +/-4 vox p=.5; LR flip p=.5; DiceCE; 128^3",
        },
        "summary": summary_rows,
        "paired_deltas_vs_a0": paired_rows,
        "external31_access": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 124)
    print("FINAL72 A0 vs A2 vs A3 — SIZE-STRATIFIED ORIGINAL47 OOF")
    for group in SIZE_ORDER:
        print(f"\n{group}")
        for model in MODELS:
            s = summary_map[(group, model)]
            print(
                f"  {model}: n={s['n']:2d} | Dice={s['mean_dice']:.4f} | "
                f"Prec={s['mean_precision']:.4f} | Rec={s['mean_recall']:.4f} | "
                f"HD95={s['mean_hd95_mm']:.2f}mm | med|RVE|={s['median_abs_rve_pct']:.1f}% | "
                f"Dice<.70={s['dice_lt_0p70']}"
            )
        a0 = summary_map[(group, "A0")]
        for candidate in ("A2", "A3"):
            c = summary_map[(group, candidate)]
            print(
                f"    {candidate}-A0: Dice={c['mean_dice']-a0['mean_dice']:+.4f} | "
                f"Prec={c['mean_precision']-a0['mean_precision']:+.4f} | "
                f"Rec={c['mean_recall']-a0['mean_recall']:+.4f}"
            )
    print("\nKey question: does A2/A3 improve SMALL-bladder precision without sacrificing recall?")
    print(f"Outputs: {output_dir}")
    print("=" * 124)


if __name__ == "__main__":
    main()
