#!/usr/bin/env python3
"""Build an app-compatible review pack for single-stage Final91 RAW Dice < 0.80.

The source result is the locked full-volume Final91 A3 DynUNet evaluation on External31.
Cases are selected only from the recorded raw Student+EMA ensemble Dice. The frozen model is
rerun for those cases so the pack contains native-grid predictions, and the reproduced Dice must
match the recorded value. Each case contains the source image, normalized ground truth, raw
single-stage prediction, metrics, and a self-contained metadata sidecar. Slicer segmentation and
HASSL provenance fields are embedded in both .seg.nrrd files for application compatibility.

This script does not use the Final136 CenterNet -> DynUNet pipeline, does not apply LCC, and does
not retrain, tune, or modify the source data.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def _consume_option(argv, name):
    value = None
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return value, cleaned


GPU, CLEAN_ARGV = _consume_option(sys.argv, "--gpu")
if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.argv = CLEAN_ARGV

import numpy as np  # noqa: E402
import nrrd  # noqa: E402
import SimpleITK as sitk  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
import scripts.benchmark_final91_a3_external31 as benchmark  # noqa: E402
from scripts.build_final91_external31_failure_review_pack import (  # noqa: E402
    copy_exact,
    infer_checkpoint,
    read_ground_truth_segment_metadata,
    verify_saved_segment_metadata,
    write_csv,
)
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.validate_external_threshold_31 import read_gt_binary  # noqa: E402


BENCHMARK_DIR = Path("experiments/external31_final91_a3_locked")
FINAL91_CHECKPOINT = Path("experiments/final91_a3_all91/checkpoints/final_checkpoint.pth")
TRAINING_METADATA = Path("experiments/final91_a3_all91/final_training_metadata.json")
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
EXPECTED_EXTERNAL = 31
EXPECTED_BELOW_DEFAULT_THRESHOLD = 13
MODEL_THRESHOLD = 0.50
DEFAULT_DICE_THRESHOLD = 0.80
DEFAULT_APP_SEGMENT_TAGS = "Segmentation.Status:inprogress|"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_app_segment_tags(metadata):
    """Preserve source tags and add the app status tag only when the source has none."""
    result = dict(metadata)
    tags = str(result.get("tags", "")).strip()
    if tags in {"", "|"}:
        result["tags"] = DEFAULT_APP_SEGMENT_TAGS
    return result


def app_metadata(case_id, content, postprocessing, checkpoint="", source_sha256=""):
    result = {
        "HASSL_Content": str(content),
        "HASSL_CaseID": str(case_id),
        "HASSL_PredictionRole": "external31_review_only",
        "HASSL_Model": "final91_a3_all91_single_stage_dynunet",
        "HASSL_Threshold": f"{MODEL_THRESHOLD:.2f}",
        "HASSL_Postprocessing": str(postprocessing),
        "HASSL_External31Access": "evaluation_only",
    }
    if checkpoint:
        result["HASSL_Checkpoint"] = str(checkpoint)
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


def native_geometry(path):
    image = sitk.ReadImage(str(path))
    return {
        "size_xyz": [int(value) for value in image.GetSize()],
        "spacing_xyz": [float(value) for value in image.GetSpacing()],
        "origin_xyz": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }


def verify_native_geometry(segmentation_path, image_path):
    actual = sitk.ReadImage(str(segmentation_path))
    reference = sitk.ReadImage(str(image_path))
    checks = {
        "size": actual.GetSize() == reference.GetSize(),
        "spacing": np.allclose(actual.GetSpacing(), reference.GetSpacing(), atol=1e-7, rtol=0),
        "origin": np.allclose(actual.GetOrigin(), reference.GetOrigin(), atol=1e-6, rtol=0),
        "direction": np.allclose(actual.GetDirection(), reference.GetDirection(), atol=1e-7, rtol=0),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Saved segmentation does not match native image geometry: "
            f"{segmentation_path}: {checks}"
        )


def write_segmentation(destination, mask, image_path, segment_metadata, extra_metadata):
    write_mask_with_spatial_geometry(
        str(destination),
        np.asarray(mask, dtype=np.uint8) * int(segment_metadata["label_value"]),
        reference_image_path=str(image_path),
        segment_name=segment_metadata["segment_name"],
        segment_id=segment_metadata["segment_id"],
        label_value=segment_metadata["label_value"],
        segment_color=segment_metadata["color"],
        segment_layer=segment_metadata["layer"],
        segment_tags=segment_metadata["tags"],
        extra_metadata=extra_metadata,
    )
    verify_saved_segment_metadata(destination, segment_metadata)
    verify_app_metadata(destination, extra_metadata)
    verify_native_geometry(destination, image_path)


def select_cases(metric_rows, dice_threshold):
    ensemble_rows = [
        row for row in metric_rows if str(row.get("mode", "")).upper() == "ENSEMBLE"
    ]
    by_id = {str(row["case_id"]): row for row in ensemble_rows}
    if len(ensemble_rows) != EXPECTED_EXTERNAL or len(by_id) != EXPECTED_EXTERNAL:
        raise RuntimeError("Expected exactly 31 unique Final91 ENSEMBLE metric rows")
    selected = sorted(
        (case_id for case_id, row in by_id.items() if float(row["dice"]) < dice_threshold),
        key=lambda case_id: (float(by_id[case_id]["dice"]), case_id),
    )
    return by_id, selected


def diagnostic_hint(row):
    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    hd95 = float(row["hd95_mm"])
    if dice < 0.10:
        return "CATASTROPHIC_LOCALIZATION_OR_EMPTY"
    if precision < 0.60 and recall < 0.60:
        return "SEVERE_MIXED_OR_LOCALIZATION"
    if precision < 0.65 and recall >= 0.75:
        return "OVERSEGMENTATION_FALSE_POSITIVE"
    if recall < 0.65 and precision >= 0.75:
        return "UNDERSEGMENTATION_MISSED_FOREGROUND"
    if np.isfinite(hd95) and hd95 > 5.0:
        return "DETACHED_COMPONENT_OR_BOUNDARY_OUTLIER"
    return "BOUNDARY_SMALL_TARGET_OR_AMBIGUOUS_APPEARANCE"


def main():
    parser = argparse.ArgumentParser(
        description="Build single-stage Final91 External31 RAW Dice<0.80 app review pack"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--checkpoint", default=str(FINAL91_CHECKPOINT))
    parser.add_argument("--training-metadata", default=str(TRAINING_METADATA))
    parser.add_argument("--dice-threshold", type=float, default=DEFAULT_DICE_THRESHOLD)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.dice_threshold < 1.0:
        parser.error("--dice-threshold must be between 0 and 1")

    benchmark_dir = Path(args.benchmark_dir)
    checkpoint = Path(args.checkpoint)
    training_metadata_path = Path(args.training_metadata)
    metric_path = benchmark_dir / "external31_case_metrics.csv"
    evaluation_metadata_path = benchmark_dir / "external31_metadata.json"
    training_metadata = read_json(training_metadata_path)
    evaluation_metadata = read_json(evaluation_metadata_path)
    if int(training_metadata.get("n_total_human_labels", -1)) != 91:
        raise RuntimeError("Training metadata is not the single-stage all91 Final91 model")
    if Path(str(training_metadata.get("deployment_checkpoint", ""))) != checkpoint:
        raise RuntimeError("Checkpoint differs from the Final91 deployment checkpoint")
    if Path(str(evaluation_metadata.get("checkpoint", ""))) != checkpoint:
        raise RuntimeError("External31 metrics were not produced by the requested Final91 checkpoint")
    if str(evaluation_metadata.get("primary_mode", "")).upper() != "ENSEMBLE":
        raise RuntimeError("Final91 External31 primary mode is not the locked ensemble")
    if abs(float(evaluation_metadata.get("threshold", -1.0)) - MODEL_THRESHOLD) > 1e-8:
        raise RuntimeError("Final91 External31 metrics are not from locked threshold 0.50")
    if str(evaluation_metadata.get("primary_postprocessing", "")) != "raw_no_lcc":
        raise RuntimeError("Final91 External31 metrics are not the locked RAW result")

    current, selected_ids = select_cases(read_csv(metric_path), args.dice_threshold)
    if not selected_ids:
        raise RuntimeError("No single-stage Final91 RAW cases are below the Dice threshold")
    if (
        abs(args.dice_threshold - DEFAULT_DICE_THRESHOLD) < 1e-8
        and len(selected_ids) != EXPECTED_BELOW_DEFAULT_THRESHOLD
    ):
        raise RuntimeError(
            "Expected 13/31 locked Final91 single-stage cases below Dice 0.80, "
            f"found {len(selected_ids)}; refusing to build from mismatched metrics"
        )

    threshold_token = f"{args.dice_threshold:.2f}".replace(".", "p")
    output_dir = Path(args.output_dir) if args.output_dir else (
        benchmark_dir / f"raw_dice_lt_{threshold_token}_app_review_pack"
    )
    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Review pack requires the prototype single-stage DynUNet recipe")
    cv.apply_baseline(config, resize_size=128, epochs=1)
    images = benchmark.collect_exact(Path(args.image_dir), config.image_suffix)
    labels = benchmark.collect_exact(Path(args.gt_dir), config.label_suffix)
    if set(images) != set(current) or set(labels) != set(current):
        raise RuntimeError("External image/GT directories do not exactly match recorded External31")

    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    segment_metadata = {}
    prediction_destinations = {}
    prediction_extra_metadata = {}
    gt_extra_metadata = {}
    source_hashes = {}
    for case_id in selected_ids:
        case_dir = output_dir / case_id
        copy_exact(images[case_id], case_dir / "image.mha")
        source_hashes[case_id] = {
            "image_sha256": sha256(images[case_id]),
            "ground_truth_sha256": sha256(labels[case_id]),
        }
        metadata = ensure_app_segment_tags(read_ground_truth_segment_metadata(labels[case_id]))
        segment_metadata[case_id] = metadata
        reference = sitk.ReadImage(str(images[case_id]))
        gt = read_gt_binary(labels[case_id], reference)
        gt_extra_metadata[case_id] = app_metadata(
            case_id,
            "ground_truth",
            "human_ground_truth",
            source_sha256=source_hashes[case_id]["ground_truth_sha256"],
        )
        write_segmentation(
            case_dir / "ground_truth" / "ground_truth.seg.nrrd",
            gt,
            images[case_id],
            metadata,
            gt_extra_metadata[case_id],
        )
        prediction_destinations[case_id] = (
            case_dir / "final91_pred" / "final91_pred.seg.nrrd"
        )
        prediction_extra_metadata[case_id] = app_metadata(
            case_id,
            "prediction",
            "raw_no_lcc",
            checkpoint=str(checkpoint),
        )

    print("Single-stage Final91 selected-case inference")
    computed = infer_checkpoint(
        config,
        checkpoint,
        selected_ids,
        images,
        labels,
        prediction_destinations,
        {case_id: current[case_id]["dice"] for case_id in selected_ids},
        segment_metadata,
        extra_metadata=prediction_extra_metadata,
    )

    manifest = []
    for review_order, case_id in enumerate(selected_ids, start=1):
        prediction_path = prediction_destinations[case_id]
        verify_app_metadata(prediction_path, prediction_extra_metadata[case_id])
        verify_native_geometry(prediction_path, images[case_id])
        row = current[case_id]
        metric_row = {
            "review_order": review_order,
            "case_id": case_id,
            "selection_reason": f"FINAL91_SINGLE_STAGE_RAW_DICE_LT_{threshold_token.upper()}",
            "final91_raw_dice": float(row["dice"]),
            "final91_raw_precision": float(row["precision"]),
            "final91_raw_recall": float(row["recall"]),
            "final91_raw_signed_rve_pct": float(row["signed_rve_pct"]),
            "final91_raw_abs_rve_pct": float(row["abs_rve_pct"]),
            "final91_raw_hd95_mm": float(row["hd95_mm"]),
            "final91_raw_component_count": int(float(row["component_count"])),
            "gt_vox": int(float(row["gt_vox"])),
            "pred_vox": int(computed[case_id]["pred_vox"]),
            "embedded_segment_id": segment_metadata[case_id]["segment_id"],
            "embedded_segment_name": segment_metadata[case_id]["segment_name"],
            "embedded_segment_tags": segment_metadata[case_id]["tags"],
            "diagnostic_hint_not_ground_truth": diagnostic_hint(row),
            "review_notes": "",
        }
        manifest.append(metric_row)
        case_dir = output_dir / case_id
        (case_dir / "metrics.json").write_text(
            json.dumps(metric_row, indent=2), encoding="utf-8"
        )
        write_csv(case_dir / "metrics.csv", [metric_row])
        case_metadata = {
            "version": "final91_single_stage_external31_review_case_v1",
            "case_id": case_id,
            "selection": {
                "metric": "Final91 single-stage RAW ensemble Dice",
                "operator": "<",
                "threshold": float(args.dice_threshold),
                "value": float(row["dice"]),
            },
            "files": {
                "image": "image.mha",
                "ground_truth": "ground_truth/ground_truth.seg.nrrd",
                "prediction": "final91_pred/final91_pred.seg.nrrd",
                "metrics_json": "metrics.json",
                "metrics_csv": "metrics.csv",
            },
            "sources": {
                "image": str(images[case_id]),
                "ground_truth": str(labels[case_id]),
                "metrics": str(metric_path),
                "evaluation_metadata": str(evaluation_metadata_path),
                "training_metadata": str(training_metadata_path),
                "config": str(args.config),
                "checkpoint": str(checkpoint),
                **source_hashes[case_id],
            },
            "native_geometry": native_geometry(images[case_id]),
            "prediction_definition": {
                "architecture": "DynUNet",
                "pipeline": "single_stage_full_volume",
                "probability": "Student+EMA 50/50 ensemble",
                "threshold": MODEL_THRESHOLD,
                "postprocessing": "raw_no_lcc",
            },
            "segment_metadata": segment_metadata[case_id],
            "embedded_ground_truth_metadata": gt_extra_metadata[case_id],
            "embedded_prediction_metadata": prediction_extra_metadata[case_id],
            "metrics": metric_row,
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(case_metadata, indent=2), encoding="utf-8"
        )

    manifest_path = output_dir / f"final91_external31_dice_lt_{threshold_token}_manifest.csv"
    write_csv(manifest_path, manifest)
    pack_metadata = {
        "version": "final91_single_stage_external31_dice_threshold_app_review_pack_v1",
        "selection": f"Final91 single-stage RAW ensemble Dice < {args.dice_threshold:.2f}",
        "n_external": EXPECTED_EXTERNAL,
        "n_selected": len(selected_ids),
        "selected_case_ids_in_ascending_raw_dice_order": selected_ids,
        "model": "Final91 A3 all91 single-stage full-volume DynUNet",
        "checkpoint": str(checkpoint),
        "prediction_definition": "Student+EMA 50/50 @ 0.50; RAW; no LCC",
        "per_case_layout": [
            "image.mha",
            "ground_truth/ground_truth.seg.nrrd",
            "final91_pred/final91_pred.seg.nrrd",
            "metrics.json",
            "metrics.csv",
            "metadata.json",
        ],
        "segmentation_metadata": (
            "GT and prediction preserve the source segment ID, name, label value, layer, color, "
            "and tags; both use the full native image extent and verified size, spacing, origin, "
            "direction, Slicer binary-labelmap metadata, and HASSL provenance fields."
        ),
        "source_data_modified": False,
        "external_role": "review only; no tuning or model selection",
    }
    (output_dir / "pack_metadata.json").write_text(
        json.dumps(pack_metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "FINAL91 SINGLE-STAGE EXTERNAL31 — RAW DICE BELOW REVIEW THRESHOLD\n\n"
        "Each case contains image.mha, app-compatible native-grid ground truth, the raw "
        "single-stage Final91 DynUNet prediction, metrics, and metadata.json. Open image.mha "
        "with both .seg.nrrd files. No CenterNet, crop, LCC, or Final136 prediction is included.\n",
        encoding="utf-8",
    )

    archive_path = None
    if args.archive:
        archive_path = Path(
            shutil.make_archive(
                str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
            )
        )
    print("=" * 112)
    print("FINAL91 SINGLE-STAGE EXTERNAL31 APP REVIEW PACK — COMPLETE")
    print(f"Selection: RAW Dice < {args.dice_threshold:.2f}")
    print(f"Selected:  {len(selected_ids)}/{EXPECTED_EXTERNAL}")
    print(f"Pack:      {output_dir}")
    print(f"Manifest:  {manifest_path}")
    if archive_path:
        print(f"ZIP:       {archive_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
