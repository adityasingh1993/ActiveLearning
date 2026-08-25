#!/usr/bin/env python3
"""Build a native Slicer review pack from held-out QC90 two-stage predictions.

Every included prediction comes from the fold that excluded that case from both CenterNet and
DynUNet training. Selection is the union of RAW Dice below a threshold and the bottom-K OOF cases.
Low OOF agreement is a review signal, not proof that the HUMAN_GOLD annotation is wrong.
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
import torch
from monai.data import DataLoader, Dataset
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from scripts.audit_round1_labels import discover_round1_cases  # noqa: E402
from scripts.build_final91_external31_failure_review_pack import (  # noqa: E402
    copy_exact,
    read_ground_truth_segment_metadata,
    verify_saved_segment_metadata,
    write_csv,
)
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)


AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
STAGE1_DIR = Path("experiments/final91_label_qc90_centernet3d_oof")
STAGE2_DIR = Path("experiments/final91_label_qc90_two_stage_oof")
OUTPUT = STAGE2_DIR / "bottom15_and_dice_lt_0p70_review_pack"
EXPECTED_LIVE = 91
EXPECTED_QC90 = 90
QUARANTINED_CASE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def derived_segment_metadata(gt_metadata, name, segment_id, color):
    result = dict(gt_metadata)
    result.update({"segment_name": name, "segment_id": segment_id, "color": color})
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def app_metadata(case_id, fold, content, postprocessing, source_sha256=""):
    result = {
        "HASSL_Content": content,
        "HASSL_CaseID": str(case_id),
        "HASSL_FrozenOOFFold": str(int(fold)),
        "HASSL_PredictionRole": "heldout_label_qc_only",
        "HASSL_Threshold": "0.50",
        "HASSL_Postprocessing": postprocessing,
        "HASSL_External31Access": "false",
    }
    if source_sha256:
        result["HASSL_SourceSegmentationSHA256"] = str(source_sha256)
    return result


def verify_app_metadata(destination, expected):
    header = nrrd.read_header(str(destination))
    mismatches = {
        key: {"expected": str(value), "actual": header.get(key)}
        for key, value in expected.items()
        if str(header.get(key)) != str(value)
    }
    if mismatches:
        raise RuntimeError(f"Application metadata mismatch for {destination}: {mismatches}")


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


def label_features(gt, reference):
    _, components = ndimage.label(gt, structure=np.ones((3, 3, 3), dtype=np.uint8))
    faces = (
        gt[0].any(), gt[-1].any(), gt[:, 0].any(), gt[:, -1].any(),
        gt[:, :, 0].any(), gt[:, :, -1].any(),
    )
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    return {
        "gt_components_26": int(components),
        "gt_touches_native_border": int(any(faces)),
        "gt_border_faces_touched": int(sum(bool(value) for value in faces)),
        "gt_foreground_fraction": float(gt.mean()),
        "gt_volume_mm3": float(gt.sum() * np.prod(spacing)),
    }


def diagnostic_hint(row):
    if int(row["qc_full_grid_fallback"]):
        return "OOF_LOCALIZER_MISS_REVIEW_IMAGE_AND_GT_LOCATION"
    if int(row["gt_components_26"]) > 1:
        return "POSSIBLE_DISCONNECTED_GT_COMPONENT"
    if float(row["raw_dice"]) < 0.10:
        return "CATASTROPHIC_OOF_DISAGREEMENT_OR_AMBIGUOUS_TARGET"
    if float(row["lcc_delta_dice"]) >= 0.02:
        return "DETACHED_FALSE_POSITIVE_COMPONENT"
    precision, recall = float(row["raw_precision"]), float(row["raw_recall"])
    if precision >= 0.75 and recall < 0.65:
        return "OOF_UNDERSEGMENTATION_OR_GT_TOO_EXTENSIVE"
    if precision < 0.65 and recall >= 0.75:
        return "OOF_OVERSEGMENTATION_OR_GT_TOO_CONSERVATIVE"
    if precision < 0.65 and recall < 0.65:
        return "SEVERE_MIXED_OOF_DISAGREEMENT"
    return "BOUNDARY_SMALL_TARGET_OR_DIFFICULT_APPEARANCE"


def invert_model_grid(mask, batch, inverse, image_path):
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    native = invert_probability_exact(tensor, batch, inverse, index=0)
    reference, probability = normalize_native_probability(native, image_path)
    return reference, probability > 0.5


def main():
    parser = argparse.ArgumentParser(description="Build Final91 QC90 OOF label-review pack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--stage1-dir", default=str(STAGE1_DIR))
    parser.add_argument("--stage2-dir", default=str(STAGE2_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--bottom-k", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 < args.failure_dice < 1:
        parser.error("--failure-dice must be between 0 and 1")
    if not 1 <= args.bottom_k <= EXPECTED_QC90:
        parser.error(f"--bottom-k must be in [1,{EXPECTED_QC90}]")

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(value) for value in audit.get("all_current_human_label_ids", []))
    if (
        not audit.get("all_visible_labels_passed_audit", False)
        or not audit.get("selection_provenance_enforced", False)
        or len(audited_ids) != EXPECTED_LIVE
        or QUARANTINED_CASE_ID not in audited_ids
    ):
        raise RuntimeError("Final91 live-label audit/quarantine provenance is invalid")
    qc_ids = sorted(set(audited_ids) - {QUARANTINED_CASE_ID})

    stage1_dir, stage2_dir = Path(args.stage1_dir), Path(args.stage2_dir)
    stage1_summary = read_json(stage1_dir / "centernet3d_gate_summary.json")
    stage2_summary = read_json(stage2_dir / "stage2_qc90_oof_summary.json")
    if not stage1_summary.get("complete_qc90_oof", False):
        raise RuntimeError("Complete Stage-1 QC90 OOF predictions are required")
    if not stage2_summary.get("complete_qc90_oof", False):
        raise RuntimeError("Complete Stage-2 QC90 OOF predictions are required")
    if stage1_summary.get("external31_access") is not False or stage2_summary.get("external31_access") is not False:
        raise RuntimeError("QC provenance unexpectedly indicates external31 access")

    stage1_rows = read_csv(stage1_dir / "centernet3d_oof_metrics.csv")
    stage2_rows = read_csv(stage2_dir / "stage2_oof_metrics.csv")
    stage1_by_id = {str(row["case_id"]): row for row in stage1_rows}
    stage2_by_id = {str(row["case_id"]): row for row in stage2_rows}
    if len(stage1_rows) != EXPECTED_QC90 or set(stage1_by_id) != set(qc_ids):
        raise RuntimeError("Stage-1 OOF metrics do not contain the exact QC90 population")
    if len(stage2_rows) != EXPECTED_QC90 or set(stage2_by_id) != set(qc_ids):
        raise RuntimeError("Stage-2 OOF metrics do not contain the exact QC90 population")
    if any(int(stage1_by_id[case_id]["fold"]) != int(float(stage2_by_id[case_id]["fold"])) for case_id in qc_ids):
        raise RuntimeError("Stage-1/Stage-2 OOF fold mismatch")

    ordered = sorted(stage2_rows, key=lambda row: (float(row["raw_dice"]), str(row["case_id"])))
    threshold_ids = {
        str(row["case_id"]) for row in ordered if float(row["raw_dice"]) < args.failure_dice
    }
    bottom_ids = {str(row["case_id"]) for row in ordered[:args.bottom_k]}
    selected_ids = sorted(
        threshold_ids | bottom_ids,
        key=lambda case_id: (float(stage2_by_id[case_id]["raw_dice"]), case_id),
    )

    config = HASSLConfig.from_yaml(args.config)
    config.preprocessing_mode = "resize"
    config.spatial_size = (128, 128, 128)
    _, source_ids, by_id, _ = discover_round1_cases(config, Path(args.source_manifest))
    if len(source_ids) != 47 or sorted(by_id) != audited_ids:
        raise RuntimeError("Live labels or frozen original47 source changed after audit")

    mask_dir = stage2_dir / "oof_model_grid_masks"
    missing_masks = [case_id for case_id in selected_ids if not (mask_dir / f"{case_id}.npz").exists()]
    if missing_masks:
        raise RuntimeError(
            "OOF masks are missing. Rerun the relevant Stage-2 folds with --save-oof-masks. "
            f"Missing: {missing_masks}"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=True, to_tensor=True,
    )
    loader = DataLoader(
        Dataset(
            [{"id": case_id, "image": str(by_id[case_id]["image"])} for case_id in selected_ids],
            transform=transform,
        ),
        batch_size=1, shuffle=False, num_workers=0,
    )

    manifest = []
    for order, batch in enumerate(loader, start=1):
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
        saved = np.load(mask_dir / f"{case_id}.npz")
        reference, raw = invert_model_grid(saved["raw"], batch, inverse, by_id[case_id]["image"])
        _, lcc = invert_model_grid(saved["lcc"], batch, inverse, by_id[case_id]["image"])
        stage1 = stage1_by_id[case_id]
        roi_grid = np.zeros((128, 128, 128), dtype=np.uint8)
        bounds = [int(float(stage1[key])) for key in (
            "crop_z0", "crop_z1", "crop_y0", "crop_y1", "crop_x0", "crop_x1"
        )]
        z0, z1, y0, y1, x0, x1 = bounds
        roi_grid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1
        _, roi = invert_model_grid(roi_grid, batch, inverse, by_id[case_id]["image"])
        gt = read_gt_binary(by_id[case_id]["label"], reference)
        native_raw_metrics = binary_metrics(raw, gt)
        if abs(float(native_raw_metrics["dice"]) - float(stage2_by_id[case_id]["raw_dice"])) > 0.01:
            raise RuntimeError(f"Native/model-grid Dice mismatch exceeds tolerance for {case_id}")

        case_dir = output_dir / case_id
        copy_exact(Path(by_id[case_id]["image"]), case_dir / "image.mha")
        source_gt = Path(by_id[case_id]["label"])
        source_gt_sha256 = file_sha256(source_gt)
        row_fold = int(float(stage2_by_id[case_id]["fold"]))
        gt_metadata = read_ground_truth_segment_metadata(source_gt)
        # Match the proven application-compatible prediction schema: RAW and LCC preserve the
        # exact Segment0 ID, Name, LabelValue, Layer, Color and Tags from HUMAN_GOLD.
        raw_metadata = dict(gt_metadata)
        lcc_metadata = dict(gt_metadata)
        roi_metadata = derived_segment_metadata(
            gt_metadata, "QC90 OOF CenterNet ROI", "QC90OOFROI", "1.00 0.55 0.05"
        )
        write_segmentation(
            case_dir / "ground_truth" / "ground_truth.seg.nrrd",
            gt, by_id[case_id]["image"], gt_metadata,
            app_metadata(
                case_id, row_fold,
                "Final91 QC90 HUMAN_GOLD reference", "human_gold_reference",
                source_gt_sha256,
            ),
        )
        write_segmentation(
            case_dir / "oof_two_stage_raw_pred" / "oof_two_stage_raw_pred.seg.nrrd",
            raw, by_id[case_id]["image"], raw_metadata,
            app_metadata(
                case_id, row_fold, "Final91 QC90 OOF two-stage RAW prediction", "raw"
            ),
        )
        write_segmentation(
            case_dir / "oof_two_stage_lcc_pred" / "oof_two_stage_lcc_pred.seg.nrrd",
            lcc, by_id[case_id]["image"], lcc_metadata,
            app_metadata(
                case_id, row_fold, "Final91 QC90 OOF two-stage LCC prediction",
                "largest_connected_component",
            ),
        )
        write_segmentation(
            case_dir / "oof_centernet_roi" / "oof_centernet_roi.seg.nrrd",
            roi, by_id[case_id]["image"], roi_metadata,
            app_metadata(
                case_id, row_fold, "Final91 QC90 OOF CenterNet ROI", "centernet_roi"
            ),
        )

        row = dict(stage2_by_id[case_id])
        row.update({
            "review_order": order,
            "selected_by_dice_threshold": int(case_id in threshold_ids),
            "selected_by_bottom_k": int(case_id in bottom_ids),
            "selection_reason": "+".join(filter(None, (
                "OOF_RAW_DICE_LT_THRESHOLD" if case_id in threshold_ids else "",
                f"BOTTOM_{args.bottom_k}_OOF_RAW_DICE" if case_id in bottom_ids else "",
            ))),
            "stage1_predicted_gt_coverage": float(stage1["gt_crop_coverage"]),
            "stage1_predicted_crop_fraction": float(stage1["crop_fraction"]),
            "stage1_center_confidence": float(stage1["center_confidence"]),
            "stage1_center_peak_margin": float(stage1["center_peak_margin"]),
            "source_ground_truth_sha256": source_gt_sha256,
            "embedded_segment_id": gt_metadata["segment_id"],
            "embedded_segment_name": gt_metadata["segment_name"],
            "embedded_label_value": gt_metadata["label_value"],
            "embedded_layer": gt_metadata["layer"],
            "embedded_color": gt_metadata["color"],
            "embedded_tags": gt_metadata["tags"],
            **label_features(gt, reference),
            "diagnostic_hint_not_ground_truth": "",
            "review_notes": "",
        })
        row["diagnostic_hint_not_ground_truth"] = diagnostic_hint(row)
        manifest.append(row)
        (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_csv(case_dir / "metrics.csv", [row])
        print(
            f"{order:02d}/{len(selected_ids)} {case_id} | fold={row['fold']} | "
            f"OOF RAW/LCC={float(row['raw_dice']):.4f}/{float(row['lcc_dice']):.4f}"
        )

    write_csv(output_dir / "label_qc90_oof_review_manifest.csv", manifest)
    summary = {
        "version": "final91_label_qc90_oof_review_pack_v2",
        "role": "heldout annotation diagnosis only; low OOF Dice is not proof of label error",
        "n_qc_clean_oof": EXPECTED_QC90,
        "quarantined_not_scored": QUARANTINED_CASE_ID,
        "failure_threshold": args.failure_dice,
        "bottom_k": args.bottom_k,
        "n_below_threshold": len(threshold_ids),
        "n_selected_union": len(selected_ids),
        "selected_case_ids_in_review_order": [row["case_id"] for row in manifest],
        "seg_nrrd_metadata": {
            "ground_truth_raw_lcc": (
                "preserve exact Segment0 ID, Name, LabelValue, Layer, Color and Tags from "
                "source HUMAN_GOLD"
            ),
            "roi": "derived ROI ID/Name/Color while preserving HUMAN_GOLD Tags",
            "required_slicer_fields": [
                "Segmentation_ContainedRepresentationNames",
                "Segmentation_MasterRepresentation",
                "Segmentation_ReferenceImageExtentOffset",
                "Segment0_ID", "Segment0_Name", "Segment0_LabelValue", "Segment0_Layer",
                "Segment0_Color", "Segment0_Extent", "Segment0_Tags",
            ],
            "application_provenance_prefix": "HASSL_",
            "verified_after_write": True,
        },
        "external31_access": False,
    }
    (output_dir / "label_qc90_oof_review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "FINAL91 FIVE-FOLD OOF LABEL-QC REVIEW PACK\n\n"
        "Every prediction was produced by models that did not train on that case.\n"
        "Low OOF Dice is a review signal, not proof that HUMAN_GOLD is wrong.\n"
        "Inspect the image and ground truth independently before viewing predictions.\n"
        "All .seg.nrrd files contain verified Slicer segment tags, full native extent and HASSL provenance.\n"
        "Ground truth, RAW and LCC preserve the same application-compatible Segment0 identity.\n"
        "A QC full-grid fallback means CenterNet missed/truncated GT; it is recorded in metrics.\n"
        f"The uncertain case {QUARANTINED_CASE_ID} remains quarantined and is not OOF-scored here.\n",
        encoding="utf-8",
    )
    archive_path = None
    if args.archive:
        archive_path = Path(shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        ))
    print("\n" + "=" * 120)
    print("FINAL91 FIVE-FOLD LABEL-QC REVIEW PACK — COMPLETE")
    print(f"QC-clean OOF cases:       {EXPECTED_QC90}")
    print(f"RAW Dice < {args.failure_dice:.2f}:       {len(threshold_ids)}")
    print(f"Bottom-K requested:       {args.bottom_k}")
    print(f"Selected union:           {len(selected_ids)}")
    print(f"Pack:                     {output_dir}")
    if archive_path is not None:
        print(f"ZIP:                      {archive_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
