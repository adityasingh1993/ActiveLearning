#!/usr/bin/env python3
"""Build an app-compatible native review pack for Final136 External31 RAW Dice < 0.80.

The script reads the completed locked External31 metrics, selects RAW Dice < the requested
threshold, reruns only those cases through the frozen Final136 CenterNet -> DynUNet pipeline,
and verifies the reproduced RAW/LCC Dice. It exports full-native-extent image, normalized GT,
RAW, LCC, and CenterNet ROI files with Slicer segment metadata plus HASSL provenance fields.
It does not retrain, tune, or modify source data.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nrrd
import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry
import scripts.benchmark_final91_a3_external31 as benchmark
import scripts.train_supervised_cv as cv
from scripts.build_final91_external31_failure_review_pack import (
    copy_exact,
    read_ground_truth_segment_metadata,
    verify_saved_segment_metadata,
    write_csv,
)
from scripts.build_final91_qc_two_stage_external31_failure_review_pack import infer_two_stage
from scripts.validate_external_threshold_31 import read_gt_binary


EVALUATION_DIR = Path("experiments/external31_final136_two_stage_all136_locked")
STAGE1_CHECKPOINT = Path(
    "experiments/final136_two_stage_all136/stage1/final_centernet3d.pth"
)
STAGE2_CHECKPOINT = Path(
    "experiments/final136_two_stage_all136/stage2/final_stage2_dynunet.pth"
)
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
EXPECTED_EXTERNAL = 31
MODEL_THRESHOLD = 0.50
DEFAULT_APP_SEGMENT_TAGS = "Segmentation.Status:inprogress|"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def app_metadata(case_id, content, postprocessing, source_sha256=""):
    result = {
        "HASSL_Content": content,
        "HASSL_CaseID": str(case_id),
        "HASSL_PredictionRole": "external31_review_only",
        "HASSL_Model": "final136_two_stage_all136",
        "HASSL_Threshold": f"{MODEL_THRESHOLD:.2f}",
        "HASSL_Postprocessing": postprocessing,
        "HASSL_External31Access": "evaluation_only",
    }
    if source_sha256:
        result["HASSL_SourceSegmentationSHA256"] = str(source_sha256)
    return result


def verify_app_metadata(path, expected):
    header = nrrd.read_header(str(path))
    mismatches = {
        key: {"expected": str(value), "actual": header.get(key)}
        for key, value in expected.items()
        if str(header.get(key)) != str(value)
    }
    if mismatches:
        raise RuntimeError(f"Application metadata mismatch for {path}: {mismatches}")


def write_segmentation(destination, mask, image_path, metadata, extra_metadata):
    write_mask_with_spatial_geometry(
        str(destination),
        np.asarray(mask, dtype=np.uint8) * int(metadata["label_value"]),
        reference_image_path=str(image_path),
        segment_name=metadata["segment_name"],
        segment_id=metadata["segment_id"],
        label_value=metadata["label_value"],
        segment_color=metadata["color"],
        segment_layer=metadata["layer"],
        segment_tags=metadata["tags"],
        extra_metadata=extra_metadata,
    )
    verify_saved_segment_metadata(destination, metadata)
    verify_app_metadata(destination, extra_metadata)
    saved = sitk.ReadImage(str(destination))
    reference = sitk.ReadImage(str(image_path))
    geometry_checks = {
        "size": saved.GetSize() == reference.GetSize(),
        "spacing": np.allclose(saved.GetSpacing(), reference.GetSpacing(), atol=1e-7, rtol=0),
        "origin": np.allclose(saved.GetOrigin(), reference.GetOrigin(), atol=1e-6, rtol=0),
        "direction": np.allclose(saved.GetDirection(), reference.GetDirection(), atol=1e-7, rtol=0),
    }
    if not all(geometry_checks.values()):
        raise RuntimeError(
            f"Saved segmentation does not exactly match native image geometry: "
            f"{destination}: {geometry_checks}"
        )


def ensure_app_segment_tags(metadata):
    """Preserve real source tags; add the app-required status tag when GT has none."""
    result = dict(metadata)
    tags = str(result.get("tags", "")).strip()
    if tags in {"", "|"}:
        result["tags"] = DEFAULT_APP_SEGMENT_TAGS
    return result


def diagnostic_hint(raw, lcc):
    dice = float(raw["dice"])
    precision = float(raw["precision"])
    recall = float(raw["recall"])
    lcc_gain = float(lcc["dice"]) - dice
    if dice < 0.10:
        return "CATASTROPHIC_LOCALIZATION_OR_STAGE2_FAILURE"
    if lcc_gain >= 0.02:
        return "DETACHED_FALSE_POSITIVE_COMPONENT"
    if recall < 0.65 and precision >= 0.75:
        return "UNDERSEGMENTATION"
    if precision < 0.65 and recall >= 0.75:
        return "OVERSEGMENTATION"
    if precision < 0.65 and recall < 0.65:
        return "SEVERE_MIXED_FAILURE"
    return "BOUNDARY_SMALL_TARGET_OR_AMBIGUOUS_APPEARANCE"


def main():
    parser = argparse.ArgumentParser(
        description="Build Final136 External31 RAW Dice<0.80 app review pack"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--evaluation-dir", default=str(EVALUATION_DIR))
    parser.add_argument("--stage1-checkpoint", default=str(STAGE1_CHECKPOINT))
    parser.add_argument("--stage2-checkpoint", default=str(STAGE2_CHECKPOINT))
    parser.add_argument("--dice-threshold", type=float, default=0.80)
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.dice_threshold < 1.0:
        parser.error("--dice-threshold must be between 0 and 1")

    evaluation_dir = Path(args.evaluation_dir)
    threshold_token = f"{args.dice_threshold:.2f}".replace(".", "p")
    output_dir = Path(args.output_dir) if args.output_dir else (
        evaluation_dir / f"raw_dice_lt_{threshold_token}_app_review_pack"
    )
    raw_rows = read_csv(evaluation_dir / "external31_final136_two_stage_raw_case_metrics.csv")
    lcc_rows = read_csv(evaluation_dir / "external31_final136_two_stage_lcc_case_metrics.csv")
    raw_by_id = {str(row["case_id"]): row for row in raw_rows}
    lcc_by_id = {str(row["case_id"]): row for row in lcc_rows}
    if (
        len(raw_by_id) != EXPECTED_EXTERNAL
        or len(lcc_by_id) != EXPECTED_EXTERNAL
        or set(raw_by_id) != set(lcc_by_id)
    ):
        raise RuntimeError("Expected identical 31-case Final136 RAW and LCC metrics")

    selected_ids = sorted(
        (case_id for case_id, row in raw_by_id.items()
         if float(row["dice"]) < args.dice_threshold),
        key=lambda case_id: (float(raw_by_id[case_id]["dice"]), case_id),
    )
    if not selected_ids:
        raise RuntimeError("No Final136 RAW cases are below the requested Dice threshold")
    if abs(args.dice_threshold - 0.80) < 1e-8 and len(selected_ids) != 17:
        raise RuntimeError(
            f"Expected 17/31 cases below 0.80 from the reported result, found {len(selected_ids)}"
        )

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=1)
    images = benchmark.collect_exact(Path(args.image_dir), config.image_suffix)
    labels = benchmark.collect_exact(Path(args.gt_dir), config.label_suffix)
    if set(images) != set(raw_by_id) or set(labels) != set(raw_by_id):
        raise RuntimeError("External image/GT directories do not exactly match recorded External31")

    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    segment_metadata = {}
    raw_destinations, lcc_destinations, roi_destinations = {}, {}, {}
    extra_metadata = {}
    for case_id in selected_ids:
        case_dir = output_dir / case_id
        copy_exact(images[case_id], case_dir / "image.mha")
        gt_metadata = ensure_app_segment_tags(
            read_ground_truth_segment_metadata(labels[case_id])
        )
        segment_metadata[case_id] = gt_metadata
        source_hash = sha256(labels[case_id])
        reference = sitk.ReadImage(str(images[case_id]))
        gt = read_gt_binary(labels[case_id], reference)
        gt_extra = app_metadata(case_id, "ground_truth", "human_ground_truth", source_hash)
        write_segmentation(
            case_dir / "ground_truth" / "ground_truth.seg.nrrd",
            gt, images[case_id], gt_metadata, gt_extra,
        )
        raw_destinations[case_id] = (
            case_dir / "final136_two_stage_raw_pred" / "final136_two_stage_raw_pred.seg.nrrd"
        )
        lcc_destinations[case_id] = (
            case_dir / "final136_two_stage_lcc_pred" / "final136_two_stage_lcc_pred.seg.nrrd"
        )
        roi_destinations[case_id] = (
            case_dir / "final136_centernet_roi" / "final136_centernet_roi.seg.nrrd"
        )
        extra_metadata[case_id] = {
            "raw": app_metadata(case_id, "prediction", "raw_no_lcc"),
            "lcc": app_metadata(case_id, "prediction", "largest_26_connected_component"),
            "roi": app_metadata(case_id, "localizer_roi", "centernet_box_50pct_margin"),
        }

    print("Final136 selected-case inference")
    computed = infer_two_stage(
        config, selected_ids, images, labels,
        Path(args.stage1_checkpoint), Path(args.stage2_checkpoint),
        raw_destinations, lcc_destinations, roi_destinations, segment_metadata,
        {case_id: raw_by_id[case_id]["dice"] for case_id in selected_ids},
        {case_id: lcc_by_id[case_id]["dice"] for case_id in selected_ids},
        extra_metadata=extra_metadata,
    )
    for case_id in selected_ids:
        for kind, destination in (
            ("raw", raw_destinations[case_id]),
            ("lcc", lcc_destinations[case_id]),
            ("roi", roi_destinations[case_id]),
        ):
            verify_app_metadata(destination, extra_metadata[case_id][kind])

    manifest = []
    for order, case_id in enumerate(selected_ids, start=1):
        raw, lcc, rerun = raw_by_id[case_id], lcc_by_id[case_id], computed[case_id]
        row = {
            "review_order": order,
            "case_id": case_id,
            "selection_reason": f"FINAL136_RAW_DICE_LT_{args.dice_threshold:.2f}",
            "final136_raw_dice": float(raw["dice"]),
            "final136_lcc_dice": float(lcc["dice"]),
            "lcc_minus_raw_dice": float(lcc["dice"]) - float(raw["dice"]),
            "raw_precision": float(raw["precision"]),
            "raw_recall": float(raw["recall"]),
            "raw_signed_rve_pct": float(raw["signed_rve_pct"]),
            "raw_abs_rve_pct": float(raw["abs_rve_pct"]),
            "raw_hd95_mm": float(raw["hd95_mm"]),
            "raw_component_count": int(float(raw["component_count"])),
            "centernet_roi_gt_coverage": rerun["centernet_roi_gt_coverage"],
            "centernet_confidence": rerun["center_confidence"],
            "centernet_peak_margin": rerun["center_peak_margin"],
            "centernet_crop_fraction_model_grid": rerun["crop_fraction_model_grid"],
            "gt_vox": int(float(raw["gt_vox"])),
            "raw_pred_vox": int(rerun["raw"]["pred_vox"]),
            "embedded_segment_id": segment_metadata[case_id]["segment_id"],
            "embedded_segment_name": segment_metadata[case_id]["segment_name"],
            "embedded_segment_tags": segment_metadata[case_id]["tags"],
            "diagnostic_hint_not_ground_truth": diagnostic_hint(raw, lcc),
            "review_notes": "",
        }
        manifest.append(row)
        case_dir = output_dir / case_id
        (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_csv(case_dir / "metrics.csv", [row])

    manifest_path = output_dir / "final136_external31_dice_lt_0p80_review_manifest.csv"
    write_csv(manifest_path, manifest)
    metadata = {
        "version": "final136_external31_dice_lt_0p80_app_review_pack_v1",
        "selection": f"Final136 External31 RAW Dice < {args.dice_threshold:.2f}",
        "n_external": EXPECTED_EXTERNAL,
        "n_selected": len(selected_ids),
        "selected_case_ids_in_ascending_raw_dice_order": selected_ids,
        "contents": [
            "native image", "normalized native-grid human ground truth",
            "Final136 two-stage RAW prediction", "Final136 two-stage LCC prediction",
            "Final136 CenterNet ROI", "metrics and review hints",
        ],
        "segmentation_metadata": (
            "GT, RAW, and LCC preserve identical source segment ID, name, label value, layer, "
            "color, and Segment0_Tags. All segmentations have full native extent, Slicer binary "
            "labelmap/reference metadata, and verified HASSL provenance fields."
        ),
        "source_data_modified": False,
        "external_role": "review only; no tuning or model selection",
    }
    (output_dir / "pack_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "FINAL136 TWO-STAGE EXTERNAL31 — RAW DICE < 0.80 APP REVIEW PACK\n\n"
        "Each case contains image.mha plus app-compatible native-grid GT, RAW, LCC, and "
        "CenterNet ROI .seg.nrrd files. GT/RAW/LCC use the same source segment identity and "
        "tags. Cases are ordered by ascending RAW Dice in the manifest.\n",
        encoding="utf-8",
    )
    archive_path = None
    if args.archive:
        archive_path = Path(shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        ))
    print("=" * 116)
    print("FINAL136 EXTERNAL31 DICE<0.80 APP REVIEW PACK — COMPLETE")
    print(f"Selected:  {len(selected_ids)}/{EXPECTED_EXTERNAL}")
    print(f"Pack:      {output_dir}")
    print(f"Manifest:  {manifest_path}")
    if archive_path:
        print(f"ZIP:       {archive_path}")
    print("=" * 116)


if __name__ == "__main__":
    main()
