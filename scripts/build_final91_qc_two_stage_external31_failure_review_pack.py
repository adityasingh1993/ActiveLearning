#!/usr/bin/env python3
"""Build a Slicer-ready review pack for final all90 two-stage External31 failures.

Selection is locked to final all90 two-stage RAW Dice < 0.70. For every selected case the script
reruns only the already-frozen models and exports the native image, GT, full-volume Final91
prediction, final two-stage RAW/LCC predictions, and the predicted CenterNet ROI. The ROI/GT
coverage is diagnosis-only and makes crop misses directly visible. No model rule is changed.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _consume_option(argv, name):
    value, cleaned, index = None, [argv[0]], 1
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value, index = argv[index + 1], index + 2
            continue
        if token.startswith(name + "="):
            value, index = token.split("=", 1)[1], index + 1
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from hassl.models.centernet3d import CenterNet3D, decode_peak  # noqa: E402
from hassl.training.trainer import build_network  # noqa: E402
import scripts.benchmark_final91_a3_external31 as benchmark  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.build_final91_external31_failure_review_pack import (  # noqa: E402
    copy_exact,
    infer_checkpoint,
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


TWO_STAGE_DIR = Path("experiments/external31_final91_qc_two_stage_all90_locked")
FULL_VOLUME_DIR = Path("experiments/external31_final91_a3_locked")
STAGE1_FINAL = Path("experiments/final91_qc_two_stage_all90/stage1/final_centernet3d.pth")
STAGE2_FINAL = Path("experiments/final91_qc_two_stage_all90/stage2/final_stage2_dynunet.pth")
FULL_VOLUME_CHECKPOINT = Path("experiments/final91_a3_all91/checkpoints/final_checkpoint.pth")
OUTPUT = TWO_STAGE_DIR / "dice_lt_0p70_failure_review_pack"
DEFAULT_IMAGE_DIR = Path("/data/v1/compressed/image")
DEFAULT_GT_DIR = Path("/data/v1/compressed/label")
THRESHOLD = 0.50
SAFETY_MARGIN = 0.50


def read_csv(path):
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def crop_bounds(center, size):
    shape = np.asarray((128, 128, 128), dtype=float)
    center_voxels = np.asarray(center, dtype=float) * (shape - 1.0)
    expanded = np.asarray(size, dtype=float) * shape * (1.0 + 2.0 * SAFETY_MARGIN)
    lo = np.maximum(np.floor(center_voxels - 0.5 * expanded).astype(int), 0)
    hi = np.minimum(np.ceil(center_voxels + 0.5 * expanded).astype(int), 127)
    return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))


def crop_image(image, bounds):
    z0, z1, y0, y1, x0, x1 = bounds
    value = image[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    return F.interpolate(value, size=(128, 128, 128), mode="trilinear", align_corners=False)


def paste_probability(probability, bounds):
    z0, z1, y0, y1, x0, x1 = bounds
    value = F.interpolate(
        probability.float(), size=(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1),
        mode="trilinear", align_corners=False,
    )
    full = torch.zeros(
        (probability.shape[0], probability.shape[1], 128, 128, 128),
        device=probability.device, dtype=probability.dtype,
    )
    full[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = value
    return full


def main_prediction(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if torch.is_tensor(output) and output.ndim == 6:
        return output[:, 0]
    return output


def diagnostic_hint(row):
    if float(row["centernet_roi_gt_coverage"]) < 0.99:
        return "LOCALIZER_CROP_MISS_OR_TRUNCATION"
    if float(row["two_stage_raw_dice"]) < 0.10:
        return "CATASTROPHIC_STAGE2_OR_WRONG_LOCALIZATION"
    if float(row["two_stage_lcc_delta_dice"]) >= 0.02:
        return "DETACHED_FALSE_POSITIVE_COMPONENT"
    precision, recall = float(row["two_stage_raw_precision"]), float(row["two_stage_raw_recall"])
    if recall < 0.65 and precision >= 0.75:
        return "UNDERSEGMENTATION_WITH_SAFE_ROI"
    if precision < 0.65 and recall >= 0.75:
        return "OVERSEGMENTATION_INSIDE_ROI"
    if precision < 0.65 and recall < 0.65:
        return "SEVERE_MIXED_STAGE2_FAILURE"
    return "BOUNDARY_SMALL_TARGET_OR_AMBIGUOUS_BLADDER"


def infer_two_stage(
    config, selected_ids, images, labels, stage1_checkpoint, stage2_checkpoint,
    raw_destinations, lcc_destinations, roi_destinations, segment_metadata,
    expected_raw, expected_lcc, extra_metadata=None,
):
    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_linear = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=False, to_tensor=True,
    )
    inverse_nearest = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=True, to_tensor=True,
    )
    items = [{"id": case_id, "image": str(images[case_id])} for case_id in selected_ids]
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Two-stage failure-pack inference requires CUDA")
    first = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    second = torch.load(stage2_checkpoint, map_location="cpu", weights_only=False)
    localizer = CenterNet3D().to(device)
    student = build_network("dynunet", 1, 0.0).to(device)
    teacher = build_network("dynunet", 1, 0.0).to(device)
    localizer.load_state_dict(first["model_state"])
    student.load_state_dict(second["net_A"])
    teacher.load_state_dict(second["teacher"])
    localizer.eval(); student.eval(); teacher.eval()
    results = {}
    roi_metadata = {
        "segment_id": "CenterNet_ROI", "segment_name": "CenterNet ROI",
        "label_value": 1, "layer": 0, "color": "1.0 0.5 0.0", "tags": "|",
    }
    with torch.no_grad():
        for batch in loader:
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            image = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=True):
                outputs = localizer(image)
                center, size, confidence, peak_margin = decode_peak(outputs)
            bounds = crop_bounds(center[0].float().cpu().numpy(), size[0].float().cpu().numpy())
            crop = crop_image(image, bounds)
            with torch.amp.autocast(device.type, enabled=True):
                student_probability = torch.sigmoid(main_prediction(student(crop)))
                teacher_probability = torch.sigmoid(main_prediction(teacher(crop)))
                probability = 0.5 * (student_probability + teacher_probability)
            model_grid_probability = paste_probability(probability, bounds)
            traced_probability = image.clone()
            traced_probability.zero_()
            traced_probability.copy_(model_grid_probability.to(dtype=traced_probability.dtype))
            native_probability = invert_probability_exact(
                traced_probability, batch, inverse_linear, index=0
            )
            reference, probability_zyx = normalize_native_probability(
                native_probability, images[case_id]
            )
            gt = read_gt_binary(labels[case_id], reference)
            raw = probability_zyx > THRESHOLD
            lcc, component_count = benchmark.largest_connected_component(raw)

            z0, z1, y0, y1, x0, x1 = bounds
            roi_model = image.clone()
            roi_model.zero_()
            roi_model[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1.0
            native_roi = invert_probability_exact(roi_model, batch, inverse_nearest, index=0)
            _, roi_zyx = normalize_native_probability(native_roi, images[case_id])
            roi = roi_zyx > 0.5

            raw_metrics, lcc_metrics = binary_metrics(raw, gt), binary_metrics(lcc, gt)
            if abs(float(raw_metrics["dice"]) - float(expected_raw[case_id])) > 1e-4:
                raise RuntimeError(f"{case_id}: rerun RAW Dice differs from recorded result")
            if abs(float(lcc_metrics["dice"]) - float(expected_lcc[case_id])) > 1e-4:
                raise RuntimeError(f"{case_id}: rerun LCC Dice differs from recorded result")
            gt_coverage = float(np.logical_and(roi, gt).sum() / max(int(gt.sum()), 1))
            embedded = segment_metadata[case_id]
            for destination, mask, metadata in (
                (raw_destinations[case_id], raw, embedded),
                (lcc_destinations[case_id], lcc, embedded),
                (roi_destinations[case_id], roi, roi_metadata),
            ):
                content = (
                    "raw" if destination == raw_destinations[case_id]
                    else "lcc" if destination == lcc_destinations[case_id]
                    else "roi"
                )
                write_mask_with_spatial_geometry(
                    str(destination), mask.astype(np.uint8) * int(metadata["label_value"]),
                    reference_image_path=str(images[case_id]),
                    segment_name=metadata["segment_name"], segment_id=metadata["segment_id"],
                    label_value=metadata["label_value"], segment_color=metadata["color"],
                    segment_layer=metadata["layer"], segment_tags=metadata["tags"],
                    extra_metadata=(extra_metadata or {}).get(case_id, {}).get(content),
                )
                verify_saved_segment_metadata(destination, metadata)
            results[case_id] = {
                "raw": raw_metrics, "lcc": lcc_metrics,
                "component_count": int(component_count),
                "center_confidence": float(confidence[0].item()),
                "center_peak_margin": float(peak_margin[0].item()),
                "crop_bounds_zyx": list(bounds),
                "crop_fraction_model_grid": float(
                    ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) / (128 ** 3)
                ),
                "centernet_roi_gt_coverage": gt_coverage,
            }
            print(
                f"  {case_id}: RAW/LCC={raw_metrics['dice']:.4f}/{lcc_metrics['dice']:.4f} "
                f"ROI coverage={gt_coverage:.4f} components={component_count}"
            )
    del localizer, student, teacher
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build final all90 two-stage External31 Dice<0.70 review pack"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--two-stage-dir", default=str(TWO_STAGE_DIR))
    parser.add_argument("--full-volume-dir", default=str(FULL_VOLUME_DIR))
    parser.add_argument("--stage1-checkpoint", default=str(STAGE1_FINAL))
    parser.add_argument("--stage2-checkpoint", default=str(STAGE2_FINAL))
    parser.add_argument("--full-volume-checkpoint", default=str(FULL_VOLUME_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 < args.failure_dice < 1:
        parser.error("--failure-dice must be between 0 and 1")

    two_stage_dir, full_volume_dir = Path(args.two_stage_dir), Path(args.full_volume_dir)
    raw_rows = read_csv(two_stage_dir / "external31_all90_two_stage_raw_case_metrics.csv")
    lcc_rows = read_csv(two_stage_dir / "external31_all90_two_stage_lcc_case_metrics.csv")
    full_rows = [
        row for row in read_csv(full_volume_dir / "external31_case_metrics.csv")
        if str(row.get("mode", "")).upper() == "ENSEMBLE"
    ]
    raw_by_id = {str(row["case_id"]): row for row in raw_rows}
    lcc_by_id = {str(row["case_id"]): row for row in lcc_rows}
    full_by_id = {str(row["case_id"]): row for row in full_rows}
    if len(raw_by_id) != 31 or set(raw_by_id) != set(lcc_by_id) or set(raw_by_id) != set(full_by_id):
        raise RuntimeError("Expected identical 31-case raw/LCC/full-volume metrics")
    selected_ids = sorted(
        case_id for case_id, row in raw_by_id.items()
        if float(row["dice"]) < args.failure_dice
    )
    if not selected_ids:
        raise RuntimeError("No final two-stage RAW cases satisfy Dice < failure threshold")

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=1)
    images = benchmark.collect_exact(Path(args.image_dir), config.image_suffix)
    labels = benchmark.collect_exact(Path(args.gt_dir), config.label_suffix)
    if set(images) != set(raw_by_id) or set(labels) != set(raw_by_id):
        raise RuntimeError("External image/GT directories do not exactly match recorded External31")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild intentionally")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    segment_metadata, full_destinations = {}, {}
    raw_destinations, lcc_destinations, roi_destinations = {}, {}, {}
    for case_id in selected_ids:
        case_dir = output_dir / case_id
        copy_exact(images[case_id], case_dir / "image.mha")
        copy_exact(labels[case_id], case_dir / "ground_truth" / "ground_truth.seg.nrrd")
        segment_metadata[case_id] = read_ground_truth_segment_metadata(labels[case_id])
        full_destinations[case_id] = (
            case_dir / "fullvolume_final91_pred" / "fullvolume_final91_pred.seg.nrrd"
        )
        raw_destinations[case_id] = (
            case_dir / "all90_two_stage_raw_pred" / "all90_two_stage_raw_pred.seg.nrrd"
        )
        lcc_destinations[case_id] = (
            case_dir / "all90_two_stage_lcc_pred" / "all90_two_stage_lcc_pred.seg.nrrd"
        )
        roi_destinations[case_id] = case_dir / "centernet_roi" / "centernet_roi.seg.nrrd"

    print("Full-volume Final91 selected-case inference")
    full_computed = infer_checkpoint(
        config, Path(args.full_volume_checkpoint), selected_ids, images, labels,
        full_destinations,
        {case_id: full_by_id[case_id]["dice"] for case_id in selected_ids},
        segment_metadata,
    )
    print("Final all90 two-stage selected-case inference")
    two_stage = infer_two_stage(
        config, selected_ids, images, labels,
        Path(args.stage1_checkpoint), Path(args.stage2_checkpoint),
        raw_destinations, lcc_destinations, roi_destinations, segment_metadata,
        {case_id: raw_by_id[case_id]["dice"] for case_id in selected_ids},
        {case_id: lcc_by_id[case_id]["dice"] for case_id in selected_ids},
    )

    manifest = []
    for order, case_id in enumerate(selected_ids, start=1):
        raw, lcc, full, rerun = (
            raw_by_id[case_id], lcc_by_id[case_id], full_by_id[case_id], two_stage[case_id]
        )
        row = {
            "review_order": order, "case_id": case_id,
            "selection_reason": "ALL90_TWO_STAGE_RAW_DICE_LT_0P70",
            "diagnostic_hint_not_ground_truth": "",
            "fullvolume_final91_dice": float(full["dice"]),
            "two_stage_raw_dice": float(raw["dice"]),
            "two_stage_lcc_dice": float(lcc["dice"]),
            "two_stage_minus_fullvolume_dice": float(raw["dice"]) - float(full["dice"]),
            "two_stage_lcc_delta_dice": float(lcc["dice"]) - float(raw["dice"]),
            "two_stage_raw_precision": float(raw["precision"]),
            "two_stage_raw_recall": float(raw["recall"]),
            "two_stage_raw_signed_rve_pct": float(raw["signed_rve_pct"]),
            "two_stage_raw_abs_rve_pct": float(raw["abs_rve_pct"]),
            "two_stage_raw_hd95_mm": float(raw["hd95_mm"]),
            "two_stage_raw_component_count": int(float(raw["component_count"])),
            "centernet_roi_gt_coverage": rerun["centernet_roi_gt_coverage"],
            "centernet_confidence": rerun["center_confidence"],
            "centernet_peak_margin": rerun["center_peak_margin"],
            "centernet_crop_fraction_model_grid": rerun["crop_fraction_model_grid"],
            "centernet_crop_bounds_zyx": rerun["crop_bounds_zyx"],
            "gt_vox": int(float(raw["gt_vox"])),
            "fullvolume_pred_vox": int(full_computed[case_id]["pred_vox"]),
            "two_stage_raw_pred_vox": int(rerun["raw"]["pred_vox"]),
            "embedded_segment_id": segment_metadata[case_id]["segment_id"],
            "embedded_segment_name": segment_metadata[case_id]["segment_name"],
            "review_notes": "",
        }
        row["diagnostic_hint_not_ground_truth"] = diagnostic_hint(row)
        manifest.append(row)
        case_dir = output_dir / case_id
        (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_csv(case_dir / "metrics.csv", [row])

    write_csv(output_dir / "failure_review_manifest.csv", manifest)
    metadata = {
        "version": "final91_qc_two_stage_external31_failure_review_pack_v1",
        "selection": "final all90 two-stage RAW Dice < 0.70",
        "failure_dice": args.failure_dice,
        "n_selected": len(selected_ids), "selected_case_ids": selected_ids,
        "contents": [
            "native image", "human ground truth", "full-volume Final91 prediction",
            "final all90 two-stage RAW prediction", "final all90 two-stage LCC prediction",
            "final all90 CenterNet ROI", "joined metrics and diagnostic hint",
        ],
        "seg_nrrd_metadata": (
            "Prediction segment identity is copied from GT; CenterNet ROI uses its own orange "
            "segment identity. Slicer representation, reference offset and native extent are verified."
        ),
        "external_role": "failure diagnosis only; no tuning or model selection",
        "source_data_modified": False,
    }
    (output_dir / "pack_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "FINAL91-QC FINAL TWO-STAGE EXTERNAL31 DICE<0.70 REVIEW PACK\n\n"
        "Load image.mha plus the GT, full-volume prediction, two-stage RAW/LCC predictions, "
        "and orange CenterNet ROI in 3D Slicer.\n"
        "If the orange ROI cuts the GT, localization/crop coverage failed. If ROI covers GT but "
        "the RAW prediction is incomplete, Stage 2 under-segmented.\n"
        "Diagnostic hints are triage aids only. External31 remains diagnosis-only.\n",
        encoding="utf-8",
    )
    archive_path = None
    if args.archive:
        archive_path = Path(shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        ))
    print("=" * 116)
    print("FINAL91-QC TWO-STAGE EXTERNAL31 FAILURE REVIEW PACK — COMPLETE")
    print(f"Selected Dice<0.70: {len(selected_ids)}")
    print(f"Pack:                {output_dir}")
    print(f"Manifest:            {output_dir / 'failure_review_manifest.csv'}")
    if archive_path is not None:
        print(f"ZIP:                 {archive_path}")
    print("=" * 116)


if __name__ == "__main__":
    main()
