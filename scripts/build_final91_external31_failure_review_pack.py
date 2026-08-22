#!/usr/bin/env python3
"""Build a downloadable visual-review pack for Final91 External31 failures.

Selection is the union of:
  * Final91 ensemble Dice < 0.70
  * Final91 - Final62 ensemble Dice <= -0.05

The script reruns the two locked Student+EMA 50/50 checkpoints only for the selected
cases, exports native-geometry Slicer segmentations, copies the source image and GT,
and writes joined case metrics. It never changes source data and never tunes a model,
threshold, or post-processing rule on External31.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


def _consume_option(argv, name):
    value = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
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
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
import scripts.benchmark_final91_a3_external31 as b91  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)

FINAL91_CHECKPOINT = Path("experiments/final91_a3_all91/checkpoints/final_checkpoint.pth")
FINAL62_CHECKPOINT = Path(
    "experiments/final_supervised_round2_62_translation12/checkpoints/final_checkpoint.pth"
)
BENCHMARK_DIR = Path("experiments/external31_final91_a3_locked")
OUTPUT_DIR = BENCHMARK_DIR / "failure_review_pack"
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
THRESHOLD = 0.50


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
            writer.writerow({key: row.get(key, "") for key in fields})


def copy_exact(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")


def read_ground_truth_segment_metadata(path: Path):
    """Read the single foreground segment identity that should be preserved in predictions."""
    header = nrrd.read_header(str(path))
    prefixes = sorted(
        key.rsplit("_", 1)[0]
        for key in header
        if key.startswith("Segment") and key.endswith("_Name")
    )
    if len(prefixes) > 1:
        raise RuntimeError(f"Expected one foreground segment in {path}, found {prefixes}")
    prefix = prefixes[0] if prefixes else "Segment0"
    name = str(header.get(f"{prefix}_Name", "Bladder"))
    segment_id = str(header.get(f"{prefix}_ID", name))
    try:
        label_value = int(header.get(f"{prefix}_LabelValue", 1))
        layer = int(header.get(f"{prefix}_Layer", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Slicer label/layer metadata in {path}") from exc
    if label_value <= 0:
        raise RuntimeError(f"Foreground Segment0_LabelValue must be positive in {path}")
    return {
        "segment_id": segment_id,
        "segment_name": name,
        "label_value": label_value,
        "layer": layer,
        "color": str(header.get(f"{prefix}_Color", "0.0 1.0 0.0")),
        "tags": str(header.get(f"{prefix}_Tags", "|")),
    }


def verify_saved_segment_metadata(path: Path, expected):
    """Fail if the saved prediction lost its embedded 3D Slicer segmentation metadata."""
    header = nrrd.read_header(str(path))
    checks = {
        "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ReferenceImageExtentOffset": "0 0 0",
        "Segment0_ID": expected["segment_id"],
        "Segment0_Name": expected["segment_name"],
        "Segment0_LabelValue": str(expected["label_value"]),
        "Segment0_Layer": str(expected["layer"]),
        "Segment0_Color": expected["color"],
        "Segment0_Tags": expected["tags"],
    }
    mismatches = {
        key: {"expected": value, "actual": header.get(key)}
        for key, value in checks.items()
        if str(header.get(key)) != str(value)
    }
    if mismatches:
        raise RuntimeError(f"Embedded .seg.nrrd metadata mismatch for {path}: {mismatches}")
    if "Segment0_Extent" not in header:
        raise RuntimeError(f"Saved .seg.nrrd has no Segment0_Extent: {path}")
    return checks


def diagnostic_hint(row):
    dice = float(row["final91_dice"])
    precision = float(row["final91_precision"])
    recall = float(row["final91_recall"])
    hd95 = float(row["final91_hd95_mm"])
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
    return "BOUNDARY_OR_SMALL_TARGET_REVIEW"


def infer_checkpoint(
    config,
    checkpoint,
    selected_ids,
    images,
    labels,
    destinations,
    expected_dice,
    segment_metadata,
):
    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    items = [{"id": case_id, "image": str(images[case_id])} for case_id in selected_ids]
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError(f"Checkpoint has no EMA teacher: {checkpoint}")
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    computed = {}
    with torch.no_grad():
        for batch in loader:
            raw_id = batch["id"]
            case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
            image_t = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                s_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                t_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                ensemble = 0.5 * (s_prob + t_prob)
            native = invert_probability_exact(ensemble, batch, inverse_transform, index=0)
            reference, probability = normalize_native_probability(native, images[case_id])
            gt = read_gt_binary(labels[case_id], reference)
            pred = probability > THRESHOLD
            metrics = binary_metrics(pred, gt)
            if abs(float(metrics["dice"]) - float(expected_dice[case_id])) > 1e-4:
                raise RuntimeError(
                    f"{case_id}: rerun Dice {metrics['dice']:.6f} differs from recorded "
                    f"{float(expected_dice[case_id]):.6f} for {checkpoint}"
                )
            embedded = segment_metadata[case_id]
            prediction_with_label_value = pred.astype(np.uint8) * int(embedded["label_value"])
            write_mask_with_spatial_geometry(
                str(destinations[case_id]),
                prediction_with_label_value,
                reference_image_path=str(images[case_id]),
                segment_name=embedded["segment_name"],
                segment_id=embedded["segment_id"],
                label_value=embedded["label_value"],
                segment_color=embedded["color"],
                segment_layer=embedded["layer"],
                segment_tags=embedded["tags"],
            )
            verify_saved_segment_metadata(destinations[case_id], embedded)
            computed[case_id] = metrics
            print(f"  {case_id}: Dice={metrics['dice']:.4f} -> {destinations[case_id]}")
    del student, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return computed


def main():
    parser = argparse.ArgumentParser(description="Build Final91 External31 failure visual-review pack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--final91-checkpoint", default=str(FINAL91_CHECKPOINT))
    parser.add_argument("--final62-checkpoint", default=str(FINAL62_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--regression-delta", type=float, default=-0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true", help="Also create a ZIP beside the pack")
    args = parser.parse_args()

    if not 0.0 < args.failure_dice < 1.0:
        parser.error("--failure-dice must be between 0 and 1")
    if args.regression_delta >= 0.0:
        parser.error("--regression-delta must be negative")

    benchmark_dir = Path(args.benchmark_dir)
    metric_rows = [
        row for row in read_csv(benchmark_dir / "external31_case_metrics.csv")
        if str(row.get("mode", "")).upper() == "ENSEMBLE"
    ]
    comparison_rows = read_csv(benchmark_dir / "final91_vs_final62_external31_case_comparison.csv")
    current = {str(row["case_id"]): row for row in metric_rows}
    paired = {str(row["case_id"]): row for row in comparison_rows}
    if len(current) != 31 or set(current) != set(paired):
        raise RuntimeError("Expected identical 31-case Final91 metrics and Final62 comparison files")

    selected_ids = sorted(
        case_id for case_id in current
        if float(current[case_id]["dice"]) < args.failure_dice
        or float(paired[case_id]["delta_dice"]) <= args.regression_delta
    )
    if not selected_ids:
        raise RuntimeError("No cases satisfy the requested failure/regression rules")

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=1)
    images = b91.collect_exact(Path(args.image_dir), config.image_suffix)
    labels = b91.collect_exact(Path(args.gt_dir), config.label_suffix)
    if set(images) != set(current) or set(labels) != set(current):
        raise RuntimeError("External image/GT directories do not exactly match recorded External31 IDs")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild intentionally")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = []
    final91_destinations = {}
    final62_destinations = {}
    segment_metadata = {}
    for case_id in selected_ids:
        case_dir = output_dir / case_id
        copy_exact(images[case_id], case_dir / "image.mha")
        copy_exact(labels[case_id], case_dir / "ground_truth" / "ground_truth.seg.nrrd")
        segment_metadata[case_id] = read_ground_truth_segment_metadata(labels[case_id])
        final91_destinations[case_id] = case_dir / "final91_pred" / "final91_pred.seg.nrrd"
        final62_destinations[case_id] = case_dir / "final62_pred" / "final62_pred.seg.nrrd"

    print("Final91 selected-case inference")
    final91_computed = infer_checkpoint(
        config, Path(args.final91_checkpoint), selected_ids, images, labels,
        final91_destinations, {case_id: current[case_id]["dice"] for case_id in selected_ids},
        segment_metadata,
    )
    print("Final62 selected-case inference")
    final62_computed = infer_checkpoint(
        config, Path(args.final62_checkpoint), selected_ids, images, labels,
        final62_destinations,
        {case_id: paired[case_id]["final62_ensemble_dice"] for case_id in selected_ids},
        segment_metadata,
    )

    for index, case_id in enumerate(selected_ids, start=1):
        now = current[case_id]
        old = paired[case_id]
        reasons = []
        if float(now["dice"]) < args.failure_dice:
            reasons.append("FINAL91_DICE_LT_0P70")
        if float(old["delta_dice"]) <= args.regression_delta:
            reasons.append("FINAL91_REGRESSION_LE_MINUS_0P05")
        row = {
            "review_order": index,
            "case_id": case_id,
            "selection_reason": "+".join(reasons),
            "diagnostic_hint_not_ground_truth": "",
            "final62_dice": float(old["final62_ensemble_dice"]),
            "final91_dice": float(now["dice"]),
            "delta_dice": float(old["delta_dice"]),
            "final91_precision": float(now["precision"]),
            "final91_recall": float(now["recall"]),
            "final91_signed_rve_pct": float(now["signed_rve_pct"]),
            "final91_abs_rve_pct": float(now["abs_rve_pct"]),
            "final91_hd95_mm": float(now["hd95_mm"]),
            "gt_vox": int(float(now["gt_vox"])),
            "final62_pred_vox": int(final62_computed[case_id]["pred_vox"]),
            "final91_pred_vox": int(final91_computed[case_id]["pred_vox"]),
            "embedded_segment_id": segment_metadata[case_id]["segment_id"],
            "embedded_segment_name": segment_metadata[case_id]["segment_name"],
            "embedded_label_value": segment_metadata[case_id]["label_value"],
            "embedded_layer": segment_metadata[case_id]["layer"],
            "embedded_color": segment_metadata[case_id]["color"],
            "embedded_tags": segment_metadata[case_id]["tags"],
            "review_notes": "",
        }
        row["diagnostic_hint_not_ground_truth"] = diagnostic_hint(row)
        manifest.append(row)
        case_dir = output_dir / case_id
        (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_csv(case_dir / "metrics.csv", [row])

    write_csv(output_dir / "failure_review_manifest.csv", manifest)
    metadata = {
        "version": "final91_external31_failure_review_pack_v1",
        "selection": {
            "union": True,
            "final91_ensemble_dice_lt": args.failure_dice,
            "final91_minus_final62_ensemble_dice_lte": args.regression_delta,
        },
        "n_selected": len(selected_ids),
        "selected_case_ids": selected_ids,
        "prediction_definition": "Student+EMA 50/50 raw probability ensemble @ 0.50; no LCC",
        "seg_nrrd_metadata": (
            "Final91 and Final62 predictions preserve Segment0 ID, Name, LabelValue, Layer, "
            "Color and Tags from each ground truth; Slicer representation, reference offset "
            "and full native Segment0_Extent are written and verified"
        ),
        "source_external_role": "frozen_failure_diagnosis_only; no tuning or selection",
        "source_data_modified": False,
    }
    (output_dir / "pack_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "FINAL91 EXTERNAL31 FAILURE REVIEW PACK\n\n"
        "Load each image, ground truth, Final62 prediction and Final91 prediction together in 3D Slicer.\n"
        "The diagnostic hint is only a triage aid; confirm the phenotype visually.\n"
        "Do not use these External31 labels to tune thresholds, post-processing or checkpoint selection.\n",
        encoding="utf-8",
    )

    archive_path = None
    if args.archive:
        archive_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name))
    print("=" * 112)
    print("FINAL91 EXTERNAL31 FAILURE REVIEW PACK — COMPLETE")
    print(f"Selected cases: {len(selected_ids)}")
    print(f"Pack:           {output_dir}")
    print(f"Manifest:       {output_dir / 'failure_review_manifest.csv'}")
    if archive_path is not None:
        print(f"Download ZIP:   {archive_path}")
    print("External31 remains diagnosis-only; no model rule was changed.")
    print("=" * 112)


if __name__ == "__main__":
    main()
