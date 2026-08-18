#!/usr/bin/env python3
"""Build the final all-62 predicted-ROI dataset from leakage-safe OOF EMA maps.

This is the fair 62-case counterpart of build_guided_refiner_oof47_roi_dataset.py.
Every case uses a coarse EMA probability map produced by the fold model that held that case out.
GT is cropped with the predicted ROI and is never allowed to move, enlarge, or repair the ROI.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("build_guided_refiner_oof62_roi_dataset.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_predicted_roi_refinement_external31 import (
    build_training_target_prior,
    choose_localization_candidate,
    crop_gt_coverage,
    expanded_roi_from_candidate,
    parse_thresholds,
)
from scripts.oracle_roi_utils import geometry_equal

DEFAULT_OOF_MANIFEST = Path(
    "experiments/guided_refiner_oof62_probabilities_v1/oof62_ema_probability_manifest.csv"
)
DEFAULT_AUDIT_CSV = Path(
    "experiments/round2_supervised_62_translation12/round2_label_audit.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/guided_refiner_oof62_roi_dataset_v1")
EXPECTED_CASES = 62


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    p = argparse.ArgumentParser(description="Build guided-refiner OOF62 predicted-ROI dataset")
    p.add_argument("--oof-manifest", default=str(DEFAULT_OOF_MANIFEST))
    p.add_argument("--round2-audit-csv", default=str(DEFAULT_AUDIT_CSV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--candidate-thresholds", default="0.30,0.50,0.70")
    p.add_argument("--min-component-voxels", type=int, default=8)
    p.add_argument("--roi-margin-fraction", type=float, default=0.75)
    p.add_argument("--min-target-multiplier", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    thresholds = parse_thresholds(args.candidate_thresholds)
    if args.min_component_voxels < 1:
        p.error("--min-component-voxels must be >=1")
    if args.roi_margin_fraction < 0:
        p.error("--roi-margin-fraction must be >=0")
    if args.min_target_multiplier <= 0:
        p.error("--min-target-multiplier must be >0")

    oof_manifest = Path(args.oof_manifest)
    audit_csv = Path(args.round2_audit_csv)
    output_dir = Path(args.output_dir)
    source_rows = read_csv(oof_manifest)
    if len(source_rows) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} true-OOF rows, found {len(source_rows)}")
    if len({str(r["case_id"]) for r in source_rows}) != EXPECTED_CASES:
        raise RuntimeError("OOF62 manifest contains duplicate case IDs")

    # Frozen localization prior from all 62 HUMAN_GOLD cases. This never sees external GT.
    prior = build_training_target_prior(audit_csv, expected_count=62)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    image_dir = output_dir / "images"
    coarse_dir = output_dir / "coarse"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, source in enumerate(sorted(source_rows, key=lambda r: str(r["case_id"])), start=1):
        case_id = str(source["case_id"])
        image = sitk.ReadImage(str(source["image_path"]))
        label = sitk.ReadImage(str(source["label_path"]))
        coarse = sitk.ReadImage(str(source["probability_path"]))
        if not geometry_equal(image, label):
            raise RuntimeError(f"Image/label geometry mismatch for {case_id}")
        if not geometry_equal(image, coarse):
            raise RuntimeError(f"Image/OOF probability geometry mismatch for {case_id}")

        coarse_arr = np.asarray(sitk.GetArrayFromImage(coarse), dtype=np.float32)
        gt = np.asarray(sitk.GetArrayFromImage(label)) > 0
        candidate, n_candidates = choose_localization_candidate(
            coarse_arr,
            thresholds=thresholds,
            prior=prior,
            min_component_voxels=args.min_component_voxels,
        )
        start_xyz, size_xyz = expanded_roi_from_candidate(
            candidate,
            image.GetSize(),
            prior,
            margin_fraction=args.roi_margin_fraction,
            min_target_multiplier=args.min_target_multiplier,
        )
        coverage, gt_inside, gt_total = crop_gt_coverage(gt, start_xyz, size_xyz)

        image_crop = sitk.RegionOfInterest(image, size=size_xyz, index=start_xyz)
        coarse_crop = sitk.RegionOfInterest(coarse, size=size_xyz, index=start_xyz)
        label_crop = sitk.RegionOfInterest(label, size=size_xyz, index=start_xyz)

        image_out = image_dir / f"{case_id}.mha"
        coarse_out = coarse_dir / f"{case_id}.ema_prob.mha"
        label_out = label_dir / f"{case_id}.seg.nrrd"
        sitk.WriteImage(image_crop, str(image_out), useCompression=True)
        sitk.WriteImage(coarse_crop, str(coarse_out), useCompression=True)
        sitk.WriteImage(label_crop, str(label_out), useCompression=True)

        full_vox = max(int(np.prod(image.GetSize())), 1)
        crop_vox = int(np.prod(size_xyz))
        rows.append({
            "case_id": case_id,
            "fold": int(source["fold"]),
            "image_path": str(image_out),
            "coarse_probability_path": str(coarse_out),
            "label_path": str(label_out),
            "source_image_path": str(source["image_path"]),
            "source_label_path": str(source["label_path"]),
            "source_probability_path": str(source["probability_path"]),
            "oof_ema_dice_at_050": float(source["oof_ema_dice_at_050"]),
            "roi_start_x": int(start_xyz[0]),
            "roi_start_y": int(start_xyz[1]),
            "roi_start_z": int(start_xyz[2]),
            "roi_size_x": int(size_xyz[0]),
            "roi_size_y": int(size_xyz[1]),
            "roi_size_z": int(size_xyz[2]),
            "roi_volume_fraction_of_full": float(crop_vox / full_vox),
            "gt_coverage_by_predicted_roi": float(coverage),
            "gt_voxels_inside_predicted_roi": int(gt_inside),
            "gt_voxels_total": int(gt_total),
            "n_localization_candidates": int(n_candidates),
            "selected_candidate_threshold": float(candidate["threshold"]),
            "selected_candidate_score": float(candidate["score"]),
            "selected_component_voxels": int(candidate["voxels"]),
            "selected_component_mean_probability": float(candidate["mean_probability"]),
        })
        print(
            f"[{index:2d}/{EXPECTED_CASES}] {case_id} | fold={source['fold']} | "
            f"crop/full={crop_vox/full_vox:.3f} | GT coverage(eval)={coverage:.3f} | "
            f"OOF Dice={float(source['oof_ema_dice_at_050']):.4f}"
        )

    manifest_path = output_dir / "guided_refiner_oof62_roi_manifest.csv"
    write_csv(manifest_path, rows)
    coverages = np.asarray([float(r["gt_coverage_by_predicted_roi"]) for r in rows], dtype=float)
    crop_fracs = np.asarray([float(r["roi_volume_fraction_of_full"]) for r in rows], dtype=float)
    metadata = {
        "version": "guided_refiner_oof62_predicted_roi_dataset_v1",
        "n_cases": len(rows),
        "source_oof_manifest": str(oof_manifest),
        "round2_audit_csv_for_global_prior": str(audit_csv),
        "candidate_thresholds": thresholds,
        "roi_margin_fraction_each_side": float(args.roi_margin_fraction),
        "min_target_multiplier": float(args.min_target_multiplier),
        "median_gt_coverage": float(np.median(coverages)),
        "gt_coverage_lt_090": int(np.sum(coverages < 0.90)),
        "gt_coverage_lt_050": int(np.sum(coverages < 0.50)),
        "median_crop_fraction_of_full": float(np.median(crop_fracs)),
        "gt_usage_rule": (
            "GT is cropped with the OOF-predicted ROI and used only as the supervised target/coverage diagnostic. "
            "GT never moves, enlarges, or repairs the proposed ROI."
        ),
        "comparison_note": (
            "This all-62 dataset removes the prior 47-vs-62 refiner training-size mismatch. Every refiner "
            "input still uses leakage-safe held-out coarse guidance."
        ),
    }
    metadata_path = output_dir / "guided_refiner_oof62_roi_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("GUIDED REFINER OOF62 PREDICTED-ROI DATASET COMPLETE")
    print(f"Cases:                 {len(rows)}")
    print(f"Median GT coverage:    {np.median(coverages):.3f}")
    print(f"Coverage < .90:        {int(np.sum(coverages < 0.90))}/{len(rows)}")
    print(f"Coverage < .50:        {int(np.sum(coverages < 0.50))}/{len(rows)}")
    print(f"Median crop/full:      {np.median(crop_fracs):.3f}")
    print(f"Manifest:              {manifest_path}")
    print(f"Metadata:              {metadata_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
