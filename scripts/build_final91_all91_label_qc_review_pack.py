#!/usr/bin/env python3
"""Run final two-stage inference on all 91 labels and build a label-QC review pack.

This is deliberately same-data diagnosis, not model evaluation: 90 cases were used to train the
final two-stage models and one case was quarantined. The script selects two-stage RAW Dice < 0.70
and always includes the documented quarantine, then exports native Slicer overlays and evidence
useful for spotting label mistakes.
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
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402
from scipy import ndimage  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from hassl.models.centernet3d import CenterNet3D, decode_peak  # noqa: E402
from hassl.training.trainer import build_network  # noqa: E402
from scripts.audit_round1_labels import discover_round1_cases  # noqa: E402
from scripts.build_final91_external31_failure_review_pack import (  # noqa: E402
    copy_exact,
    read_ground_truth_segment_metadata,
    verify_saved_segment_metadata,
    write_csv,
)
from scripts.build_final91_qc_two_stage_external31_failure_review_pack import (  # noqa: E402
    crop_bounds,
    crop_image,
    main_prediction,
    paste_probability,
)
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
import scripts.benchmark_final91_a3_external31 as benchmark  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)


AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
STAGE1_CHECKPOINT = Path("experiments/final91_qc_two_stage_all90/stage1/final_centernet3d.pth")
STAGE2_CHECKPOINT = Path("experiments/final91_qc_two_stage_all90/stage2/final_stage2_dynunet.pth")
FULL_VOLUME_CHECKPOINT = Path("experiments/final91_a3_all91/checkpoints/final_checkpoint.pth")
OUTPUT = Path("experiments/final91_qc_two_stage_all90/all91_label_qc_dice_lt_0p70_review_pack")
THRESHOLD = 0.50
EXPECTED_LIVE = 91
QUARANTINED_CASE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_segmentation(destination, mask, image_path, metadata):
    write_mask_with_spatial_geometry(
        str(destination), mask.astype(np.uint8) * int(metadata["label_value"]),
        reference_image_path=str(image_path),
        segment_name=metadata["segment_name"], segment_id=metadata["segment_id"],
        label_value=metadata["label_value"], segment_color=metadata["color"],
        segment_layer=metadata["layer"], segment_tags=metadata["tags"],
    )
    verify_saved_segment_metadata(destination, metadata)


def dice_between(first, second):
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    return float(
        (2.0 * np.logical_and(first, second).sum() + 1e-8)
        / (first.sum() + second.sum() + 1e-8)
    )


def label_features(gt, reference):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, components = ndimage.label(gt, structure=structure)
    faces = [
        gt[0].any(), gt[-1].any(), gt[:, 0].any(), gt[:, -1].any(),
        gt[:, :, 0].any(), gt[:, :, -1].any(),
    ]
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    return {
        "gt_components_26": int(components),
        "gt_touches_native_border": int(any(faces)),
        "gt_border_faces_touched": int(sum(bool(x) for x in faces)),
        "gt_foreground_fraction": float(gt.mean()),
        "gt_volume_mm3": float(gt.sum() * np.prod(spacing)),
    }


def diagnostic_hint(row):
    if row["case_id"] == QUARANTINED_CASE_ID:
        return "KNOWN_QUARANTINED_GT_AND_VISIBILITY_REVIEW"
    if int(row["gt_components_26"]) > 1:
        return "POSSIBLE_LABEL_DISCONNECTED_COMPONENT"
    if float(row["centernet_roi_gt_coverage"]) < 0.99:
        return "LOCALIZER_CROP_MISS_NOT_AUTOMATICALLY_LABEL_ERROR"
    if (
        float(row["fullvolume_dice"]) < 0.70
        and float(row["two_stage_raw_dice"]) < 0.70
        and float(row["fullvolume_vs_two_stage_agreement_dice"]) >= 0.80
    ):
        return "POSSIBLE_GT_MISMATCH_BOTH_MODELS_AGREE"
    if float(row["fullvolume_dice"]) >= 0.80 and float(row["two_stage_raw_dice"]) < 0.70:
        return "STAGE2_OR_CROP_FAILURE_GT_SUPPORTED_BY_FULLVOLUME"
    if float(row["two_stage_raw_recall"]) < 0.65 and float(row["two_stage_raw_precision"]) >= 0.75:
        return "TWO_STAGE_UNDERSEGMENTATION_WITH_SAFE_ROI"
    if float(row["two_stage_lcc_delta_dice"]) >= 0.02:
        return "DETACHED_FALSE_POSITIVE_COMPONENT"
    if float(row["fullvolume_dice"]) < 0.70 and float(row["two_stage_raw_dice"]) < 0.70:
        return "HARD_OR_AMBIGUOUS_CASE_REVIEW_GT_CAREFULLY"
    return "BOUNDARY_SMALL_TARGET_OR_MODEL_SPECIFIC_FAILURE"


def infer_full_volume(config, case_ids, by_id, checkpoint, destinations, segment_metadata):
    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"],
        nearest_interp=False, to_tensor=True,
    )
    loader = DataLoader(
        Dataset([
            {"id": case_id, "image": str(by_id[case_id]["image"])} for case_id in case_ids
        ], transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError("Full-volume Final91 checkpoint has no EMA teacher")
    student.eval(); teacher.eval()
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    results = {}
    with torch.no_grad():
        for batch in loader:
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            image = batch["image"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_probability = torch.sigmoid(cv.main_prediction(inferer(image, student)))
                teacher_probability = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                probability = 0.5 * (student_probability + teacher_probability)
            native = invert_probability_exact(probability, batch, inverse, index=0)
            reference, probability_zyx = normalize_native_probability(
                native, by_id[case_id]["image"]
            )
            gt = read_gt_binary(by_id[case_id]["label"], reference)
            pred = probability_zyx > THRESHOLD
            metrics = binary_metrics(pred, gt)
            write_segmentation(
                destinations[case_id], pred, by_id[case_id]["image"], segment_metadata[case_id]
            )
            results[case_id] = {"metrics": metrics, "mask": pred}
            print(f"  {case_id}: full-volume Dice={metrics['dice']:.4f}")
    del student, teacher
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Build all91 same-data label-QC review pack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--stage1-checkpoint", default=str(STAGE1_CHECKPOINT))
    parser.add_argument("--stage2-checkpoint", default=str(STAGE2_CHECKPOINT))
    parser.add_argument("--full-volume-checkpoint", default=str(FULL_VOLUME_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--failure-dice", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 < args.failure_dice < 1:
        parser.error("--failure-dice must be between 0 and 1")

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if (
        not audit.get("all_visible_labels_passed_audit", False)
        or not audit.get("selection_provenance_enforced", False)
        or len(audited_ids) != EXPECTED_LIVE
        or QUARANTINED_CASE_ID not in audited_ids
    ):
        raise RuntimeError("Final91 audit/quarantine provenance is not valid")
    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=1)
    _, source_ids, by_id, _ = discover_round1_cases(config, Path(args.source_manifest))
    if len(source_ids) != 47 or sorted(by_id) != audited_ids:
        raise RuntimeError("Live label directory or frozen original47 source changed")

    stage1_state = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    stage2_state = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    expected_train_ids = sorted(set(audited_ids) - {QUARANTINED_CASE_ID})
    if stage1_state.get("train_ids") != expected_train_ids or stage2_state.get("train_ids") != expected_train_ids:
        raise RuntimeError("Final two-stage checkpoints are not trained on exact QC90")
    if stage1_state.get("recipe", {}).get("external31_access") is not False:
        raise RuntimeError("Stage-1 checkpoint provenance indicates external access")
    if stage2_state.get("recipe", {}).get("external31_access") is not False:
        raise RuntimeError("Stage-2 checkpoint provenance indicates external access")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild intentionally")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

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
    loader = DataLoader(
        Dataset([
            {"id": case_id, "image": str(by_id[case_id]["image"])} for case_id in audited_ids
        ], transform=transform),
        batch_size=1, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("All91 two-stage label-QC inference requires CUDA")
    localizer = CenterNet3D().to(device)
    student = build_network("dynunet", 1, 0.0).to(device)
    teacher = build_network("dynunet", 1, 0.0).to(device)
    localizer.load_state_dict(stage1_state["model_state"])
    student.load_state_dict(stage2_state["net_A"])
    teacher.load_state_dict(stage2_state["teacher"])
    localizer.eval(); student.eval(); teacher.eval()

    roi_metadata = {
        "segment_id": "CenterNet_ROI", "segment_name": "CenterNet ROI",
        "label_value": 1, "layer": 0, "color": "1.0 0.5 0.0", "tags": "|",
    }
    all_rows, selected = [], {}
    print("=" * 120)
    print("FINAL91-QC ALL91 SAME-DATA LABEL-QC INFERENCE")
    print("Role: annotation diagnosis only; 90/91 cases are training cases")
    print(f"Selection: RAW Dice < {args.failure_dice:.2f} plus documented quarantine")
    print("=" * 120)
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
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
            pasted = paste_probability(probability, bounds)
            traced = image.clone(); traced.zero_(); traced.copy_(pasted.to(dtype=traced.dtype))
            native = invert_probability_exact(traced, batch, inverse_linear, index=0)
            reference, probability_zyx = normalize_native_probability(
                native, by_id[case_id]["image"]
            )
            gt = read_gt_binary(by_id[case_id]["label"], reference)
            raw = probability_zyx > THRESHOLD
            lcc, component_count = benchmark.largest_connected_component(raw)
            raw_metrics, lcc_metrics = binary_metrics(raw, gt), binary_metrics(lcc, gt)
            z0, z1, y0, y1, x0, x1 = bounds
            roi_model = image.clone(); roi_model.zero_()
            roi_model[:, :, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1.0
            native_roi = invert_probability_exact(roi_model, batch, inverse_nearest, index=0)
            _, roi_zyx = normalize_native_probability(native_roi, by_id[case_id]["image"])
            roi = roi_zyx > 0.5
            roi_coverage = float(np.logical_and(roi, gt).sum() / max(int(gt.sum()), 1))
            status = "QUARANTINED_NOT_TRAINED" if case_id == QUARANTINED_CASE_ID else "QC90_TRAINING_CASE"
            row = {
                "case_id": case_id, "dataset_status": status,
                "two_stage_raw_dice": raw_metrics["dice"],
                "two_stage_lcc_dice": lcc_metrics["dice"],
                "two_stage_lcc_delta_dice": lcc_metrics["dice"] - raw_metrics["dice"],
                "two_stage_raw_precision": raw_metrics["precision"],
                "two_stage_raw_recall": raw_metrics["recall"],
                "two_stage_raw_signed_rve_pct": raw_metrics["signed_rve_pct"],
                "two_stage_raw_abs_rve_pct": raw_metrics["abs_rve_pct"],
                "two_stage_raw_component_count": int(component_count),
                "centernet_roi_gt_coverage": roi_coverage,
                "centernet_confidence": float(confidence[0].item()),
                "centernet_peak_margin": float(peak_margin[0].item()),
                "centernet_crop_fraction_model_grid": float(
                    ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) / (128 ** 3)
                ),
                "centernet_crop_bounds_zyx": list(bounds),
                **label_features(gt, reference),
            }
            all_rows.append(row)
            is_selected = raw_metrics["dice"] < args.failure_dice or case_id == QUARANTINED_CASE_ID
            print(
                f"[{index:02d}/91] {case_id} | {status:<23} | "
                f"RAW/LCC={raw_metrics['dice']:.4f}/{lcc_metrics['dice']:.4f} | "
                f"ROIcov={roi_coverage:.4f} | {'REVIEW' if is_selected else 'PASS'}"
            )
            if not is_selected:
                continue
            case_dir = output_dir / case_id
            copy_exact(Path(by_id[case_id]["image"]), case_dir / "image.mha")
            copy_exact(
                Path(by_id[case_id]["label"]), case_dir / "ground_truth" / "ground_truth.seg.nrrd"
            )
            metadata = read_ground_truth_segment_metadata(Path(by_id[case_id]["label"]))
            raw_destination = case_dir / "all90_two_stage_raw_pred" / "all90_two_stage_raw_pred.seg.nrrd"
            lcc_destination = case_dir / "all90_two_stage_lcc_pred" / "all90_two_stage_lcc_pred.seg.nrrd"
            roi_destination = case_dir / "centernet_roi" / "centernet_roi.seg.nrrd"
            write_segmentation(raw_destination, raw, by_id[case_id]["image"], metadata)
            write_segmentation(lcc_destination, lcc, by_id[case_id]["image"], metadata)
            write_segmentation(roi_destination, roi, by_id[case_id]["image"], roi_metadata)
            selected[case_id] = {
                "row": row, "raw_mask": raw, "gt": gt, "segment_metadata": metadata,
                "full_destination": (
                    case_dir / "fullvolume_final91_pred" / "fullvolume_final91_pred.seg.nrrd"
                ),
            }

    del localizer, student, teacher
    torch.cuda.empty_cache()
    if not selected:
        raise RuntimeError("No cases selected for label review")
    selected_ids = sorted(selected)
    print("\nFull-volume Final91 comparison inference for selected cases")
    full_results = infer_full_volume(
        config, selected_ids, by_id, Path(args.full_volume_checkpoint),
        {case_id: selected[case_id]["full_destination"] for case_id in selected_ids},
        {case_id: selected[case_id]["segment_metadata"] for case_id in selected_ids},
    )

    manifest = []
    for order, case_id in enumerate(selected_ids, start=1):
        row = dict(selected[case_id]["row"])
        full = full_results[case_id]
        row.update({
            "review_order": order,
            "selection_reason": (
                "DOCUMENTED_QUARANTINE"
                if case_id == QUARANTINED_CASE_ID and row["two_stage_raw_dice"] >= args.failure_dice
                else "TWO_STAGE_RAW_DICE_LT_0P70"
                + ("+DOCUMENTED_QUARANTINE" if case_id == QUARANTINED_CASE_ID else "")
            ),
            "fullvolume_dice": full["metrics"]["dice"],
            "fullvolume_precision": full["metrics"]["precision"],
            "fullvolume_recall": full["metrics"]["recall"],
            "fullvolume_signed_rve_pct": full["metrics"]["signed_rve_pct"],
            "fullvolume_vs_two_stage_agreement_dice": dice_between(
                full["mask"], selected[case_id]["raw_mask"]
            ),
            "diagnostic_hint_not_ground_truth": "",
            "review_notes": "",
        })
        row["diagnostic_hint_not_ground_truth"] = diagnostic_hint(row)
        manifest.append(row)
        case_dir = output_dir / case_id
        (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        write_csv(case_dir / "metrics.csv", [row])

    all_rows.sort(key=lambda row: str(row["case_id"]))
    write_csv(output_dir / "all91_two_stage_label_qc_metrics.csv", all_rows)
    write_csv(output_dir / "label_qc_review_manifest.csv", manifest)
    training_rows = [row for row in all_rows if row["dataset_status"] == "QC90_TRAINING_CASE"]
    summary = {
        "version": "final91_all91_same_data_label_qc_v1",
        "role": "same-data annotation diagnosis only; not model validation",
        "n_live_labels": len(all_rows), "n_training_cases": len(training_rows),
        "n_quarantined": 1, "quarantined_case_id": QUARANTINED_CASE_ID,
        "failure_threshold": args.failure_dice,
        "n_raw_dice_lt_0p70": sum(float(row["two_stage_raw_dice"]) < args.failure_dice for row in all_rows),
        "n_selected_including_quarantine": len(selected_ids),
        "selected_case_ids": selected_ids,
        "training90_mean_raw_dice_in_sample": float(np.mean([
            float(row["two_stage_raw_dice"]) for row in training_rows
        ])),
        "warning": (
            "QC90 scores are training-set fit diagnostics. Low Dice can identify review candidates, "
            "but high Dice does not prove a label is correct."
        ),
    }
    (output_dir / "label_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "FINAL91 ALL91 SAME-DATA LABEL-QC REVIEW PACK\n\n"
        "This is not model validation: 90 cases were used for model training.\n"
        "Review GT independently before using predictions. Orange is the CenterNet ROI.\n"
        "Both models low vs GT but high model-model agreement is a label-review signal, not proof.\n"
        "A GT with multiple components or the documented quarantine deserves priority review.\n",
        encoding="utf-8",
    )
    archive_path = None
    if args.archive:
        archive_path = Path(shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        ))
    print("\n" + "=" * 120)
    print("FINAL91 ALL91 SAME-DATA LABEL-QC REVIEW PACK — COMPLETE")
    print(f"All labeled cases scored:          {len(all_rows)}")
    print(f"RAW Dice < {args.failure_dice:.2f}:               {summary['n_raw_dice_lt_0p70']}")
    print(f"Selected including quarantine:     {len(selected_ids)}")
    print(f"In-sample QC90 mean RAW Dice:      {summary['training90_mean_raw_dice_in_sample']:.4f}")
    print(f"Pack:                              {output_dir}")
    print(f"Manifest:                          {output_dir / 'label_qc_review_manifest.csv'}")
    if archive_path is not None:
        print(f"ZIP:                               {archive_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
