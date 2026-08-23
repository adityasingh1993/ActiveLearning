#!/usr/bin/env python3
"""Build a native-geometry review pack for failed Final91 Stage-1 ROI detections.

The selected cases come from detector_oof_metrics.csv where the default Stage-1 crop contains
less than 99% of the held-out HUMAN_GOLD bladder.  Every segmentation comparison is leakage
safe: Final91 and Final62 are replayed only with the checkpoint from the frozen original47 fold
where that case was held out.  Their resized-space Dice must reproduce the saved CV result before
the native-grid prediction is exported.

Output layout
-------------
  <output>/<case_id>/
      image.mha
      ground_truth/ground_truth.seg.nrrd
      stage1_probability/stage1_probability.mha
      stage1_prediction/stage1_prediction.seg.nrrd
      proposed_crop/proposed_crop.seg.nrrd
      final91_oof_pred/final91_oof_pred.seg.nrrd
      final62_oof_pred/final62_oof_pred.seg.nrrd
      metrics.json
      metrics.csv
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path


def _consume_gpu(argv):
    gpu = None
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--gpu":
            if index + 1 >= len(argv):
                raise SystemExit("--gpu requires a value")
            gpu = argv[index + 1]
            index += 2
            continue
        if token.startswith("--gpu="):
            gpu = token.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return gpu, cleaned


GPU, CLEAN_ARGV = _consume_gpu(sys.argv)
if GPU is not None:
    if not str(GPU).isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
sys.argv = CLEAN_ARGV

import nrrd  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402
from monai.networks.nets import UNet  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.validate_external_threshold_31 import (  # noqa: E402
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)


SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DETECTOR_DIR = Path("experiments/final91_stage1_roi_detector_e500_cv")
FINAL91_CV = Path("experiments/round5_cv_91_a3")
FINAL62_CV = Path("experiments/round2_cv_62_translation12")
OUTPUT_DIR = DETECTOR_DIR / "failure_review_pack"

SEGMENTATION_THRESHOLD = 0.50
DETECTOR_THRESHOLD = 0.30
DETECTOR_MARGIN = 0.40
MIN_COMPONENT_RELATIVE_SIZE = 0.05
MIN_COMPONENT_VOXELS = 8


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
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
            writer.writerow({key: row.get(key, "") for key in fields})


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")


def read_ground_truth_segment_metadata(path):
    header = nrrd.read_header(str(path))
    prefixes = sorted(
        key.rsplit("_", 1)[0]
        for key in header
        if key.startswith("Segment") and key.endswith("_Name")
    )
    if len(prefixes) > 1:
        raise RuntimeError(f"Expected one foreground segment in {path}, found {prefixes}")
    prefix = prefixes[0] if prefixes else "Segment0"
    try:
        label_value = int(header.get(f"{prefix}_LabelValue", 1))
        layer = int(header.get(f"{prefix}_Layer", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Slicer segment metadata in {path}") from exc
    return {
        "source_prefix": prefix,
        "source_segment_id": str(header.get(f"{prefix}_ID", "Bladder")),
        "source_segment_name": str(header.get(f"{prefix}_Name", "Bladder")),
        "label_value": label_value,
        "layer": layer,
        "tags": str(header.get(f"{prefix}_Tags", "|")),
    }


def derived_segment_metadata(source, suffix, name, color):
    return {
        "segment_id": f"{source['source_segment_id']}_{suffix}",
        "segment_name": name,
        "label_value": int(source["label_value"]),
        "layer": 0,
        "color": color,
        "tags": str(source["tags"]),
    }


def verify_segmentation_metadata(path, expected):
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
    if mismatches or "Segment0_Extent" not in header:
        raise RuntimeError(
            f"Embedded .seg.nrrd metadata verification failed for {path}: "
            f"mismatches={mismatches}, has_extent={'Segment0_Extent' in header}"
        )
    return checks


def write_segmentation(path, mask, image_path, metadata):
    valued = np.asarray(mask, dtype=np.uint8) * int(metadata["label_value"])
    write_mask_with_spatial_geometry(
        str(path),
        valued,
        reference_image_path=str(image_path),
        segment_name=metadata["segment_name"],
        segment_id=metadata["segment_id"],
        label_value=metadata["label_value"],
        segment_color=metadata["color"],
        segment_layer=metadata["layer"],
        segment_tags=metadata["tags"],
    )
    return verify_segmentation_metadata(path, metadata)


def write_probability(path, probability_zyx, reference, case_id, fold, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(probability_zyx, dtype=np.float32))
    image.CopyInformation(reference)
    image.SetMetaData("HASSL_Content", "Final91 Stage1 ROI detector probability")
    image.SetMetaData("HASSL_CaseID", str(case_id))
    image.SetMetaData("HASSL_FrozenOOFFold", str(fold))
    image.SetMetaData("HASSL_Checkpoint", str(checkpoint))
    image.SetMetaData("HASSL_ProbabilityRange", "0.0 1.0")
    sitk.WriteImage(image, str(path), True)


def checkpoint_for(cv_dir, fold):
    path = Path(cv_dir) / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def result_map(cv_dir):
    rows = read_csv(Path(cv_dir) / "cv_results.csv")
    by_id = {str(row["case_id"]): row for row in rows}
    if len(by_id) != 47 or len(rows) != 47:
        raise RuntimeError(f"Expected exact original47 OOF results in {cv_dir}")
    return by_id


def build_fold_map(source_manifest):
    fold_by_id = {}
    for spec in source_manifest.get("folds", []):
        fold = int(spec["fold"])
        for raw_case_id in spec.get("val_ids", []):
            case_id = str(raw_case_id)
            if case_id in fold_by_id:
                raise RuntimeError(f"Frozen source case {case_id} appears in multiple folds")
            fold_by_id[case_id] = fold
    all_ids = set(str(x) for x in source_manifest.get("all_case_ids", []))
    if len(all_ids) != 47 or len(fold_by_id) != 47 or set(fold_by_id) != all_ids:
        raise RuntimeError(
            "Frozen source manifest must contain exactly 47 cases, each held out once"
        )
    return fold_by_id


def bounds_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(axis=0).astype(int), coords.max(axis=0).astype(int)


def detector_crop(probability):
    raw = np.asarray(probability) >= DETECTOR_THRESHOLD
    labeled, component_count = ndimage.label(raw)
    if component_count == 0:
        shape = np.asarray(raw.shape, dtype=int)
        return np.ones_like(raw, dtype=bool), np.zeros(3, dtype=int), shape - 1, 0, 1
    sizes = np.bincount(labeled.ravel())[1:]
    minimum = max(
        MIN_COMPONENT_VOXELS,
        int(math.ceil(float(sizes.max()) * MIN_COMPONENT_RELATIVE_SIZE)),
    )
    kept_labels = np.where(sizes >= minimum)[0] + 1
    kept = np.isin(labeled, kept_labels)
    bounds = bounds_from_mask(kept)
    if bounds is None:
        shape = np.asarray(raw.shape, dtype=int)
        return np.ones_like(raw, dtype=bool), np.zeros(3, dtype=int), shape - 1, int(component_count), 1
    lo, hi = bounds
    extent = hi - lo + 1
    margin = np.ceil(extent.astype(float) * DETECTOR_MARGIN).astype(int)
    shape = np.asarray(raw.shape, dtype=int)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, shape - 1)
    crop = np.zeros_like(raw, dtype=bool)
    crop[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    return crop, lo, hi, int(component_count), 0


def build_detector_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    )


def infer_stage1(
    config_path, detector_dir, selected_ids, fold_by_id, frozen_val_ids_by_fold, cases
):
    config = HASSLConfig.from_yaml(config_path)
    config.preprocessing_mode = "resize"
    config.spatial_size = (96, 96, 96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Failure-pack inference requires CUDA")
    outputs = {}
    for fold in sorted({fold_by_id[x] for x in selected_ids}):
        fold_ids = sorted(x for x in selected_ids if fold_by_id[x] == fold)
        checkpoint_path = checkpoint_for(detector_dir, fold)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(state.get("fold", -1)) != fold:
            raise RuntimeError(f"Detector checkpoint fold mismatch: {checkpoint_path}")
        if sorted(str(x) for x in state.get("val_ids", [])) != frozen_val_ids_by_fold[fold]:
            raise RuntimeError(f"Detector checkpoint held-out IDs changed: {checkpoint_path}")
        recipe = state.get("recipe", {})
        if recipe.get("resize_size") != [96, 96, 96]:
            raise RuntimeError(f"Detector checkpoint is not the locked 96^3 recipe: {checkpoint_path}")

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
            Dataset(
                [{"id": case_id, "image": cases[case_id]["image"]} for case_id in fold_ids],
                transform=transform,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        model = build_detector_model().to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        with torch.no_grad():
            for batch in loader:
                case_value = batch["id"]
                case_id = str(case_value[0] if isinstance(case_value, (list, tuple)) else case_value)
                image_t = batch["image"].to(device)
                with torch.amp.autocast(device.type, enabled=True):
                    probability_t = torch.sigmoid(model(image_t))
                probability_96 = probability_t[0, 0].float().cpu().numpy()
                prediction_96 = probability_96 >= DETECTOR_THRESHOLD
                crop_96, lo, hi, components, fallback = detector_crop(probability_96)

                native_probability = invert_probability_exact(
                    probability_t, batch, inverse_linear, index=0
                )
                reference, probability_zyx = normalize_native_probability(
                    native_probability, cases[case_id]["image"]
                )
                prediction_t = torch.from_numpy(prediction_96[None, None].astype(np.float32))
                crop_t = torch.from_numpy(crop_96[None, None].astype(np.float32))
                native_prediction = invert_probability_exact(
                    prediction_t, batch, inverse_nearest, index=0
                )
                _, prediction_zyx = normalize_native_probability(
                    native_prediction, cases[case_id]["image"]
                )
                native_crop = invert_probability_exact(crop_t, batch, inverse_nearest, index=0)
                _, crop_zyx = normalize_native_probability(native_crop, cases[case_id]["image"])
                outputs[case_id] = {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_epoch": int(state.get("epoch", -1)),
                    "checkpoint_best_val_box_dice": float(state.get("best_val_box_dice", float("nan"))),
                    "reference": reference,
                    "probability_zyx": probability_zyx,
                    "prediction_zyx": prediction_zyx > 0.5,
                    "crop_zyx": crop_zyx > 0.5,
                    "model_probability_max": float(probability_96.max()),
                    "model_probability_mean": float(probability_96.mean()),
                    "model_prediction_fraction": float(prediction_96.mean()),
                    "model_crop_fraction": float(crop_96.mean()),
                    "model_detector_components": components,
                    "model_full_volume_fallback": fallback,
                    "model_crop_bounds_zyx": {
                        "z0": int(lo[0]), "z1": int(hi[0]),
                        "y0": int(lo[1]), "y1": int(hi[1]),
                        "x0": int(lo[2]), "x1": int(hi[2]),
                    },
                }
        del model
        torch.cuda.empty_cache()
    return outputs


def infer_oof_segmentation(config_path, cv_dir, selected_ids, fold_by_id, cases, expected):
    config = HASSLConfig.from_yaml(config_path)
    cv.apply_baseline(config, resize_size=128, epochs=100)
    if config.compute_mode != "prototype" or int(config.num_classes) != 1:
        raise RuntimeError("OOF replay requires binary prototype Student+EMA checkpoints")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("Failure-pack inference requires CUDA")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    outputs = {}
    for fold in sorted({fold_by_id[x] for x in selected_ids}):
        fold_ids = sorted(x for x in selected_ids if fold_by_id[x] == fold)
        checkpoint_path = checkpoint_for(cv_dir, fold)
        transform = get_base_transforms(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )
        inverse = build_invertd(
            keys=["pred"], transform=transform, orig_keys=["image"],
            nearest_interp=False, to_tensor=True,
        )
        loader = DataLoader(
            Dataset([cases[x] for x in fold_ids], transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        student, teacher = load_models(config, checkpoint_path, device)
        if teacher is None:
            raise RuntimeError(f"OOF checkpoint has no EMA teacher: {checkpoint_path}")
        student.eval()
        teacher.eval()
        with torch.no_grad():
            for batch in loader:
                case_value = batch["id"]
                case_id = str(case_value[0] if isinstance(case_value, (list, tuple)) else case_value)
                if int(expected[case_id]["fold"]) != fold:
                    raise RuntimeError(f"{case_id}: saved result fold differs from frozen manifest")
                image_t = batch["image"].to(device)
                target_t = batch["label"].float().to(device)
                with torch.amp.autocast(device.type, enabled=True):
                    student_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                    teacher_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    ensemble = 0.5 * (student_prob + teacher_prob)
                model_prediction = (ensemble > SEGMENTATION_THRESHOLD).float()
                replay = cv.case_metrics(
                    model_prediction, target_t, cv.transformed_spacing(image_t, config)
                )
                recorded_dice = float(expected[case_id]["dice"])
                if abs(float(replay["dice"]) - recorded_dice) > 1e-4:
                    raise RuntimeError(
                        f"{case_id}: OOF replay Dice {replay['dice']:.6f} differs from "
                        f"saved {recorded_dice:.6f} for {checkpoint_path}"
                    )
                native_probability = invert_probability_exact(ensemble, batch, inverse, index=0)
                reference, probability_zyx = normalize_native_probability(
                    native_probability, cases[case_id]["image"]
                )
                outputs[case_id] = {
                    "checkpoint": str(checkpoint_path),
                    "recorded_model_grid_metrics": {
                        "dice": recorded_dice,
                        "precision": float(expected[case_id]["precision"]),
                        "recall": float(expected[case_id]["recall"]),
                        "hd95": float(expected[case_id]["hd95"]),
                    },
                    "replayed_model_grid_metrics": {
                        "dice": float(replay["dice"]),
                        "precision": float(replay["precision"]),
                        "recall": float(replay["recall"]),
                        "hd95": float(replay["hd95"]),
                    },
                    "reference": reference,
                    "prediction_zyx": probability_zyx > SEGMENTATION_THRESHOLD,
                }
        del student, teacher
        torch.cuda.empty_cache()
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Build a native Slicer review pack for failed Final91 Stage-1 ROI detections"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--detector-dir", default=str(DETECTOR_DIR))
    parser.add_argument("--final91-cv-dir", default=str(FINAL91_CV))
    parser.add_argument("--final62-cv-dir", default=str(FINAL62_CV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--failure-coverage", type=float, default=0.99)
    parser.add_argument("--expected-count", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 < args.failure_coverage <= 1:
        parser.error("--failure-coverage must be in (0,1]")
    if args.expected_count < 1:
        parser.error("--expected-count must be >=1")

    source_manifest_path = Path(args.source_manifest)
    source_manifest = read_json(source_manifest_path)
    fold_by_id = build_fold_map(source_manifest)
    frozen_val_ids_by_fold = {
        int(spec["fold"]): sorted(str(x) for x in spec.get("val_ids", []))
        for spec in source_manifest.get("folds", [])
    }
    if set(frozen_val_ids_by_fold) != set(range(5)):
        raise RuntimeError("Frozen source manifest must contain folds 0..4 exactly once")
    detector_dir = Path(args.detector_dir)
    detector_rows = read_csv(detector_dir / "detector_oof_metrics.csv")
    selected_rows = [
        row for row in detector_rows
        if float(row["gt_crop_coverage"]) < args.failure_coverage
    ]
    selected_ids = sorted(str(row["case_id"]) for row in selected_rows)
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Detector failure metrics contain duplicate selected case IDs")
    if len(selected_ids) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} detector failures, found {len(selected_ids)}: {selected_ids}"
        )
    if set(selected_ids) - set(fold_by_id):
        raise RuntimeError("Detector failure selection contains a non-original47 case")
    detector_by_id = {str(row["case_id"]): row for row in selected_rows}

    config = HASSLConfig.from_yaml(args.config)
    cases_list = cv.collect_cases(config)
    cases = {str(case["id"]): case for case in cases_list}
    if set(selected_ids) - set(cases):
        raise RuntimeError(f"Selected cases missing from live data: {sorted(set(selected_ids)-set(cases))}")
    for case_id in selected_ids:
        if not Path(cases[case_id]["image"]).exists() or not Path(cases[case_id]["label"]).exists():
            raise FileNotFoundError(f"Missing image or label for {case_id}")

    final91_dir = Path(args.final91_cv_dir)
    final62_dir = Path(args.final62_cv_dir)
    final91_expected = result_map(final91_dir)
    final62_expected = result_map(final62_dir)
    if any(case_id not in final91_expected or case_id not in final62_expected for case_id in selected_ids):
        raise RuntimeError("A selected detector failure is absent from Final91/Final62 OOF results")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}; use --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_metadata = {
        case_id: read_ground_truth_segment_metadata(Path(cases[case_id]["label"]))
        for case_id in selected_ids
    }
    segment_metadata = {}
    for case_id in selected_ids:
        source = source_metadata[case_id]
        segment_metadata[case_id] = {
            "stage1": derived_segment_metadata(
                source, "STAGE1_DETECTOR", "Stage1 Detector Prediction", "1.0 0.2 0.2"
            ),
            "crop": derived_segment_metadata(
                source, "STAGE1_CROP", "Stage1 Proposed Crop", "1.0 0.8 0.0"
            ),
            "final91": derived_segment_metadata(
                source, "FINAL91_OOF", "Bladder Final91 OOF", "0.0 1.0 0.0"
            ),
            "final62": derived_segment_metadata(
                source, "FINAL62_OOF", "Bladder Final62 OOF", "0.1 0.6 1.0"
            ),
        }

    print("Replaying Stage-1 detector checkpoints")
    stage1 = infer_stage1(
        args.config,
        detector_dir,
        selected_ids,
        fold_by_id,
        frozen_val_ids_by_fold,
        cases,
    )
    print("Replaying Final91 held-out fold checkpoints")
    final91 = infer_oof_segmentation(
        args.config, final91_dir, selected_ids, fold_by_id, cases, final91_expected
    )
    print("Replaying Final62 held-out fold checkpoints")
    final62 = infer_oof_segmentation(
        args.config, final62_dir, selected_ids, fold_by_id, cases, final62_expected
    )

    manifest = []
    for case_id in selected_ids:
        fold = int(fold_by_id[case_id])
        case_dir = output_dir / case_id
        image_path = Path(cases[case_id]["image"])
        label_path = Path(cases[case_id]["label"])
        copy_exact(image_path, case_dir / "image.mha")
        copy_exact(label_path, case_dir / "ground_truth" / "ground_truth.seg.nrrd")

        s1 = stage1[case_id]
        gt = read_gt_binary(label_path, s1["reference"])
        stage1_native_metrics = binary_metrics(s1["prediction_zyx"], gt)
        crop_coverage_native = float(np.logical_and(s1["crop_zyx"], gt).sum() / max(int(gt.sum()), 1))
        crop_fraction_native = float(np.asarray(s1["crop_zyx"], dtype=bool).mean())
        final91_native_metrics = binary_metrics(final91[case_id]["prediction_zyx"], gt)
        final62_native_metrics = binary_metrics(final62[case_id]["prediction_zyx"], gt)

        probability_path = case_dir / "stage1_probability" / "stage1_probability.mha"
        write_probability(
            probability_path, s1["probability_zyx"], s1["reference"], case_id, fold, s1["checkpoint"]
        )
        embedded = {
            "stage1_prediction": write_segmentation(
                case_dir / "stage1_prediction" / "stage1_prediction.seg.nrrd",
                s1["prediction_zyx"], image_path, segment_metadata[case_id]["stage1"],
            ),
            "proposed_crop": write_segmentation(
                case_dir / "proposed_crop" / "proposed_crop.seg.nrrd",
                s1["crop_zyx"], image_path, segment_metadata[case_id]["crop"],
            ),
            "final91_oof_prediction": write_segmentation(
                case_dir / "final91_oof_pred" / "final91_oof_pred.seg.nrrd",
                final91[case_id]["prediction_zyx"], image_path, segment_metadata[case_id]["final91"],
            ),
            "final62_oof_prediction": write_segmentation(
                case_dir / "final62_oof_pred" / "final62_oof_pred.seg.nrrd",
                final62[case_id]["prediction_zyx"], image_path, segment_metadata[case_id]["final62"],
            ),
        }

        metrics = {
            "version": "final91_stage1_detector_failure_review_case_v1",
            "case_id": case_id,
            "frozen_oof_fold": fold,
            "selection": {
                "reason": "STAGE1_GT_CROP_COVERAGE_LT_0P99",
                "recorded_detector_metrics": detector_by_id[case_id],
            },
            "source_paths": {
                "image": str(image_path),
                "ground_truth": str(label_path),
            },
            "stage1_detector": {
                "definition": "E500 expanded-box UNet 96^3; threshold 0.30; substantial components; crop margin 0.40",
                "checkpoint": s1["checkpoint"],
                "checkpoint_epoch": s1["checkpoint_epoch"],
                "checkpoint_best_val_box_dice": s1["checkpoint_best_val_box_dice"],
                "probability_max": s1["model_probability_max"],
                "probability_mean": s1["model_probability_mean"],
                "prediction_fraction_96": s1["model_prediction_fraction"],
                "crop_fraction_96": s1["model_crop_fraction"],
                "component_count_96": s1["model_detector_components"],
                "full_volume_fallback": s1["model_full_volume_fallback"],
                "crop_bounds_96_zyx": s1["model_crop_bounds_zyx"],
                "native_prediction_vs_gt": stage1_native_metrics,
                "native_crop_gt_coverage": crop_coverage_native,
                "native_crop_fraction": crop_fraction_native,
            },
            "final91_oof": {
                "definition": "raw Student+EMA 50/50 ensemble @ 0.50; no LCC",
                "checkpoint": final91[case_id]["checkpoint"],
                "recorded_model_grid_metrics": final91[case_id]["recorded_model_grid_metrics"],
                "replayed_model_grid_metrics": final91[case_id]["replayed_model_grid_metrics"],
                "native_metrics": final91_native_metrics,
            },
            "final62_oof": {
                "definition": "raw Student+EMA 50/50 ensemble @ 0.50; no LCC",
                "checkpoint": final62[case_id]["checkpoint"],
                "recorded_model_grid_metrics": final62[case_id]["recorded_model_grid_metrics"],
                "replayed_model_grid_metrics": final62[case_id]["replayed_model_grid_metrics"],
                "native_metrics": final62_native_metrics,
            },
            "embedded_seg_nrrd_metadata": embedded,
            "review_questions": [
                "Is the HUMAN_GOLD bladder visible and anatomically credible?",
                "Do Stage1, Final91 and Final62 select the same wrong dark structure?",
                "Does Final91 cover the true bladder even though the Stage1 crop excludes it?",
                "Is the failure caused by unclear bladder appearance, a detached false positive, or wrong localization?",
            ],
            "human_review": {
                "bladder_visibility": "",
                "stage1_failure_phenotype": "",
                "final91_failure_phenotype": "",
                "final62_failure_phenotype": "",
                "notes": "",
            },
        }
        (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        flat = {
            "case_id": case_id,
            "fold": fold,
            "stage1_recorded_gt_crop_coverage": float(detector_by_id[case_id]["gt_crop_coverage"]),
            "stage1_native_gt_crop_coverage": crop_coverage_native,
            "stage1_native_crop_fraction": crop_fraction_native,
            "stage1_native_prediction_dice": stage1_native_metrics["dice"],
            "final91_recorded_dice": float(final91_expected[case_id]["dice"]),
            "final91_replayed_dice": final91[case_id]["replayed_model_grid_metrics"]["dice"],
            "final91_native_dice": final91_native_metrics["dice"],
            "final62_recorded_dice": float(final62_expected[case_id]["dice"]),
            "final62_replayed_dice": final62[case_id]["replayed_model_grid_metrics"]["dice"],
            "final62_native_dice": final62_native_metrics["dice"],
            "stage1_components": s1["model_detector_components"],
            "stage1_probability_max": s1["model_probability_max"],
            "review_notes": "",
        }
        write_csv(case_dir / "metrics.csv", [flat])
        manifest.append(flat)
        print(
            f"{case_id} | fold={fold} | crop coverage={crop_coverage_native:.4f} | "
            f"Final91 native Dice={final91_native_metrics['dice']:.4f} | "
            f"Final62 native Dice={final62_native_metrics['dice']:.4f}"
        )

    write_csv(output_dir / "failure_review_manifest.csv", manifest)
    pack_metadata = {
        "version": "final91_stage1_detector_failure_review_pack_v1",
        "selected_case_ids": selected_ids,
        "selection": f"detector_oof_metrics.csv gt_crop_coverage < {args.failure_coverage}",
        "n_selected": len(selected_ids),
        "source_manifest": str(source_manifest_path),
        "detector_dir": str(detector_dir),
        "final91_cv_dir": str(final91_dir),
        "final62_cv_dir": str(final62_dir),
        "leakage_control": "Every model uses only the checkpoint from the case's frozen held-out fold",
        "seg_nrrd_metadata": (
            "Every generated .seg.nrrd contains verified Slicer Binary labelmap representation, "
            "Segment0 ID/Name/LabelValue/Layer/Color/Tags, reference offset and full native extent."
        ),
        "ground_truth": "Copied byte-for-byte; never modified",
        "source_data_modified": False,
        "external31_access": False,
    }
    (output_dir / "pack_metadata.json").write_text(json.dumps(pack_metadata, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "FINAL91 STAGE-1 DETECTOR FAILURE REVIEW PACK\n\n"
        "Load image.mha and the four .seg.nrrd files together in 3D Slicer.\n"
        "Load stage1_probability.mha as a scalar volume and inspect it with a color map.\n"
        "Ground truth is copied unchanged. All other segmentations contain verified Slicer metadata.\n"
        "Final91 and Final62 predictions are leakage-safe OOF predictions from the case's held-out fold.\n"
        "Do not correct or overwrite the source files while reviewing this derived pack.\n",
        encoding="utf-8",
    )

    archive_path = None
    if args.archive:
        archive_path = Path(
            shutil.make_archive(
                str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
            )
        )
    print("=" * 118)
    print("FINAL91 STAGE-1 DETECTOR FAILURE REVIEW PACK — COMPLETE")
    print(f"Cases:       {len(selected_ids)}")
    print(f"Pack:        {output_dir}")
    print(f"Manifest:    {output_dir / 'failure_review_manifest.csv'}")
    if archive_path is not None:
        print(f"Download ZIP: {archive_path}")
    print("External31:  NOT ACCESSED")
    print("=" * 118)


if __name__ == "__main__":
    main()
