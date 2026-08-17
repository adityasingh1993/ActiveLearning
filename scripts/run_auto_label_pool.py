#!/usr/bin/env python3
"""Run the final 47-label segmentation model + QC gate over an unlabeled image pool.

This is the first deployment-shaped stage of the auto-label / active-learning loop.
For every image without a known human label, the script:

1. applies the frozen resize128 preprocessing,
2. runs the final student and EMA teacher from final_checkpoint.pth,
3. forms the same 50/50 probability ensemble used to build the QC development set,
4. extracts the exact same deployment-only 72 QC features,
5. predicts segmentation Dice and P(failure) with qc_model_bundle.pkl,
6. assigns a development policy bucket,
7. inverts the model-space mask back to native source geometry and saves .seg.nrrd.

Development policy buckets
--------------------------
HIGH_CONFIDENCE_PSEUDO_LABEL
    current development gate equivalent to AUTO_ACCEPT. These are candidates only until a
    future locked validation set establishes an acceptable failure bound.

REVIEW
    intermediate QC region; human inspection/correction recommended.

ACTIVE_LEARN_PRIORITY
    high predicted failure probability OR low predicted Dice. These cases are ranked first
    by predicted failure probability (descending), then predicted Dice (ascending).

Important safeguards
--------------------
- The source 47 labeled IDs are excluded using the frozen CV manifest.
- Any image with a human label beside it or under data_dir/labels is also excluded.
- Duplicate case IDs in the selected input pool hard-fail to prevent output collisions.
- QC feature names must exactly match the saved bundle schema.
- A teacher must exist in the final checkpoint; student-only inference would shift the QC
  feature distribution because several features depend on student/teacher disagreement.
- Native inversion must return the exact source voxel grid; model-space fallback is rejected.
- Saved .seg.nrrd geometry is verified against the source image after writing.
- The policy file is treated as DEVELOPMENT calibration, not a production guarantee.

Default inputs:
  experiments/final_supervised_47_translation12/checkpoints/final_checkpoint.pth
  experiments/auto_label_qc_v1/qc_model_bundle.pkl
  experiments/auto_label_qc_v1/qc_policy_development.json
  experiments/cv5_supervised_47_translation12/cv_splits.json

Default output:
  experiments/auto_label_pool_v1/
    auto_label_manifest.csv
    unlabeled_qc_features.csv
    run_metadata.json
    high_confidence_pseudo_label/*.seg.nrrd
    review/*.seg.nrrd
    active_learning_priority/*.seg.nrrd
"""

import argparse
import csv
import glob
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("run_auto_label_pool.py requires SimpleITK for strict native-geometry validation") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.active.query_engine import QueryEngine
from hassl.config import HASSLConfig
from hassl.data.data_engine import _strip_suffix, get_base_transforms
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry
import scripts.train_supervised_cv as cv
from scripts.build_oof_qc_dataset import extract_features, load_models


DEFAULT_FINAL_CHECKPOINT = Path(
    "experiments/final_supervised_47_translation12/checkpoints/final_checkpoint.pth"
)
DEFAULT_QC_DIR = Path("experiments/auto_label_qc_v1")
DEFAULT_QC_BUNDLE = DEFAULT_QC_DIR / "qc_model_bundle.pkl"
DEFAULT_POLICY = DEFAULT_QC_DIR / "qc_policy_development.json"
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/auto_label_pool_v1")

BUCKET_DIR = {
    "HIGH_CONFIDENCE_PSEUDO_LABEL": "high_confidence_pseudo_label",
    "REVIEW": "review",
    "ACTIVE_LEARN_PRIORITY": "active_learning_priority",
}


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def known_human_label_ids(config: HASSLConfig, image_paths):
    """Return IDs with an explicit human label under labels/ or beside the source image."""
    root = Path(config.data_dir)
    out = set()
    for image_path in image_paths:
        image_path = Path(image_path)
        case_id = _strip_suffix(image_path.name, config.image_suffix)
        central = root / "labels" / f"{case_id}{config.label_suffix}"
        beside = image_path.parent / f"{case_id}{config.label_suffix}"
        if central.exists() or beside.exists():
            out.add(case_id)
    return out


def collect_pool_cases(config: HASSLConfig, input_dir: Path, source_labeled_ids, limit=None):
    pattern = str(input_dir / f"**/*{config.image_suffix}")
    image_paths = sorted(glob.glob(pattern, recursive=True))
    if not image_paths:
        raise RuntimeError(f"No {config.image_suffix} images found under {input_dir}")

    explicit_human = known_human_label_ids(config, image_paths)
    exclude = set(source_labeled_ids) | explicit_human

    cases = []
    seen = {}
    for image_path in image_paths:
        case_id = _strip_suffix(os.path.basename(image_path), config.image_suffix)
        if case_id in exclude:
            continue
        if case_id in seen:
            raise RuntimeError(
                "Duplicate unlabeled case ID detected. Output names would collide:\n"
                f"  {case_id}\n  {seen[case_id]}\n  {image_path}\n"
                "Rename the files or restrict --input-dir before continuing."
            )
        seen[case_id] = image_path
        cases.append({"image": image_path, "id": case_id})

    if limit is not None:
        cases = cases[: int(limit)]
    if not cases:
        raise RuntimeError(
            "No unlabeled cases remain after excluding the frozen 47 labeled IDs and explicit human labels."
        )
    return cases, explicit_human


def load_qc_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    required = {"feature_columns", "regression_model", "failure_model"}
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"QC bundle missing required keys: {sorted(missing)}")
    feature_columns = list(bundle["feature_columns"])
    if not feature_columns:
        raise RuntimeError("QC bundle has an empty feature schema")
    if any(str(x).startswith("target_") for x in feature_columns):
        raise RuntimeError("QC bundle feature schema contains target_* leakage")
    return bundle


def load_policy(path: Path):
    payload = read_json(path)
    selected = payload.get("selected_development_policy", {})
    accept = selected.get("auto_accept_if", {})
    active = selected.get("active_learn_priority_if_not_accepted", {})
    try:
        accept_p = float(accept["predicted_failure_probability_lte"])
        accept_dice = float(accept["predicted_dice_gte"])
        active_p = float(active["predicted_failure_probability_gte"])
        active_dice = float(active["OR_predicted_dice_lte"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed QC policy file: {path}") from exc
    for name, value in {
        "accept failure probability": accept_p,
        "accept predicted Dice": accept_dice,
        "active failure probability": active_p,
        "active predicted Dice": active_dice,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Policy {name} must be in [0,1], found {value}")
    return payload, accept_p, accept_dice, active_p, active_dice


def assign_bucket(predicted_dice, failure_probability, accept_p, accept_dice, active_p, active_dice):
    if failure_probability <= accept_p and predicted_dice >= accept_dice:
        return "HIGH_CONFIDENCE_PSEUDO_LABEL"
    if failure_probability >= active_p or predicted_dice <= active_dice:
        return "ACTIVE_LEARN_PRIORITY"
    return "REVIEW"


def qc_predict(bundle, features):
    feature_columns = list(bundle["feature_columns"])
    extracted = list(features.keys())
    if extracted != feature_columns:
        missing = [x for x in feature_columns if x not in features]
        extra = [x for x in extracted if x not in feature_columns]
        raise RuntimeError(
            "QC feature schema mismatch. Refusing distribution-shifted inference.\n"
            f"Missing: {missing}\nExtra: {extra}\n"
            f"Expected ordered schema: {feature_columns}\n"
            f"Extracted ordered schema: {extracted}"
        )
    X = np.asarray([[float(features[name]) for name in feature_columns]], dtype=np.float64)
    predicted_dice = float(np.clip(bundle["regression_model"].predict(X)[0], 0.0, 1.0))
    failure_probability = float(bundle["failure_model"].predict_proba(X)[0, 1])
    return predicted_dice, failure_probability


def verify_native_mask_before_write(native_pred: np.ndarray, reference_image_path: str):
    ref = sitk.ReadImage(str(reference_image_path))
    expected_shape = tuple(reversed(ref.GetSize()))
    actual_shape = tuple(np.squeeze(native_pred).shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "Native inversion did not return the exact source voxel grid. Refusing fallback resampling.\n"
            f"Reference: {reference_image_path}\n"
            f"Expected numpy shape: {expected_shape}\n"
            f"Inverted prediction:  {actual_shape}\n"
            "This usually means MONAI transform-trace metadata was lost during inversion."
        )
    return ref


def verify_saved_geometry(segmentation_path: Path, reference_image):
    saved = sitk.ReadImage(str(segmentation_path))
    checks = {
        "size": (saved.GetSize(), reference_image.GetSize()),
        "spacing": (saved.GetSpacing(), reference_image.GetSpacing()),
        "origin": (saved.GetOrigin(), reference_image.GetOrigin()),
        "direction": (saved.GetDirection(), reference_image.GetDirection()),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        if name == "size":
            equal = tuple(actual) == tuple(expected)
        else:
            equal = bool(np.allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6))
        if not equal:
            failures.append(f"{name}: saved={actual} reference={expected}")
    if failures:
        raise RuntimeError(
            f"Saved segmentation geometry mismatch: {segmentation_path}\n" + "\n".join(failures)
        )


def main_prediction(output):
    return cv.main_prediction(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run final segmentation + leak-safe QC triage over the unlabeled image pool"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None, help="Default: config.data_dir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_FINAL_CHECKPOINT))
    parser.add_argument("--qc-bundle", default=str(DEFAULT_QC_BUNDLE))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument("--seg-threshold", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: process first N unlabeled cases")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.seg_threshold < 1.0:
        parser.error("--seg-threshold must be in (0,1)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    checkpoint = Path(args.checkpoint)
    qc_bundle_path = Path(args.qc_bundle)
    policy_path = Path(args.policy)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite only when intentionally replacing the entire pool run."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError(
            "Final QC development used prototype student + EMA teacher. "
            "Use compute_mode='prototype' to preserve inference distribution."
        )
    cv.apply_baseline(config, args.resize_size, epochs=1)

    # The QC feature dataset and policy were calibrated at segmentation threshold 0.50.
    if abs(float(args.seg_threshold) - 0.50) > 1e-8:
        raise RuntimeError(
            "QC development features were calibrated at segmentation threshold 0.50. "
            "Changing --seg-threshold would shift the feature distribution; rebuild/recalibrate QC first."
        )

    source_manifest = read_json(source_manifest_path)
    source_labeled_ids = set(source_manifest.get("all_case_ids", []))
    if len(source_labeled_ids) != 47:
        raise RuntimeError(
            f"Expected frozen source manifest with 47 labeled IDs, found {len(source_labeled_ids)}: "
            f"{source_manifest_path}"
        )

    bundle = load_qc_bundle(qc_bundle_path)
    policy_payload, accept_p, accept_dice, active_p, active_dice = load_policy(policy_path)

    input_dir = Path(args.input_dir) if args.input_dir else Path(config.data_dir)
    cases, explicit_human = collect_pool_cases(
        config, input_dir, source_labeled_ids, limit=args.limit
    )

    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    loader = DataLoader(
        Dataset(cases, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError(
            "Final checkpoint has no EMA teacher. QC includes student/teacher disagreement features, "
            "so student-only pool inference is intentionally blocked."
        )

    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    inverter = QueryEngine(
        config=config,
        manifest_path=str(output_dir / "_unused_pool_manifest.json"),
    )

    for dirname in BUCKET_DIR.values():
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    feature_rows = []
    case_path_by_id = {case["id"]: case["image"] for case in cases}

    print("=" * 110)
    print("UNLABELED AUTO-LABEL + QC POOL INFERENCE")
    print(f"Input directory:       {input_dir}")
    print(f"Unlabeled cases:       {len(cases)}")
    print(f"Excluded source IDs:   {len(source_labeled_ids)}")
    print(f"Explicit human labels: {len(explicit_human)}")
    print(f"Checkpoint:            {checkpoint}")
    print(f"QC bundle:             {qc_bundle_path}")
    print(f"Policy:                {policy_path}")
    print(f"Device:                {device}")
    print(
        f"High-confidence gate: P(failure)<={accept_p:.4f} AND predicted Dice>={accept_dice:.4f}"
    )
    print(
        f"Active priority gate: P(failure)>={active_p:.4f} OR predicted Dice<={active_dice:.4f}"
    )
    print("NOTE: high-confidence means pseudo-label candidate, not production auto-accept.")
    print("=" * 110)

    student.eval()
    teacher.eval()

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            image_t = batch["image"].to(device)
            case_raw = batch.get("id")
            case_id = case_raw[0] if isinstance(case_raw, (list, tuple)) else str(case_raw)
            source_path = case_path_by_id[case_id]

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_prob_t = torch.sigmoid(main_prediction(inferer(image_t, student)))
                teacher_prob_t = torch.sigmoid(main_prediction(inferer(image_t, teacher)))
                ensemble_prob_t = 0.5 * (student_prob_t + teacher_prob_t)

            image_np = image_t[0, 0].detach().float().cpu().numpy()
            student_prob = student_prob_t[0, 0].detach().float().cpu().numpy()
            teacher_prob = teacher_prob_t[0, 0].detach().float().cpu().numpy()
            ensemble_prob = ensemble_prob_t[0, 0].detach().float().cpu().numpy()

            features, pred_model_np = extract_features(
                image_np,
                student_prob,
                teacher_prob,
                ensemble_prob,
                args.seg_threshold,
            )
            predicted_dice, failure_probability = qc_predict(bundle, features)
            bucket = assign_bucket(
                predicted_dice,
                failure_probability,
                accept_p,
                accept_dice,
                active_p,
                active_dice,
            )

            # Invert the ensemble mask into the native source grid before export. QueryEngine's
            # helper has a compatibility fallback, but this script deliberately rejects that
            # fallback unless the returned numpy grid already matches the native source exactly.
            native_pred = inverter._invert_prediction(
                ensemble_prob_t,
                image_t,
                batch,
                0,
            )
            reference_image = verify_native_mask_before_write(native_pred, source_path)
            bucket_dir = output_dir / BUCKET_DIR[bucket]
            seg_path = bucket_dir / f"{case_id}_pred{config.label_suffix}"
            write_mask_with_spatial_geometry(
                str(seg_path),
                native_pred,
                reference_image_path=source_path,
            )
            verify_saved_geometry(seg_path, reference_image)

            probability_path = ""
            if args.save_probabilities:
                probability_path = str(probability_dir / f"{case_id}.npz")
                np.savez_compressed(
                    probability_path,
                    student_prob=student_prob.astype(np.float32),
                    teacher_prob=teacher_prob.astype(np.float32),
                    ensemble_prob=ensemble_prob.astype(np.float32),
                    prediction_model_space=pred_model_np.astype(np.uint8),
                )

            row = {
                "case_id": case_id,
                "image_path": str(source_path),
                "segmentation_path": str(seg_path),
                "prediction_source": "student_teacher_50_50_ensemble",
                "segmentation_threshold": float(args.seg_threshold),
                "predicted_dice": predicted_dice,
                "predicted_failure_probability": failure_probability,
                "qc_bucket": bucket,
                "pred_vox_model_space": int(features["pred_vox"]),
                "pred_fg_fraction_model_space": float(features["pred_fg_fraction"]),
                "student_teacher_mae": float(features["student_teacher_mae"]),
                "component_count": int(features["component_count"]),
                "largest_component_fraction": float(features["largest_component_fraction"]),
                "probability_npz": probability_path,
            }
            manifest_rows.append(row)

            frow = {"case_id": case_id, "image_path": str(source_path)}
            frow.update(features)
            frow["predicted_dice"] = predicted_dice
            frow["predicted_failure_probability"] = failure_probability
            frow["qc_bucket"] = bucket
            feature_rows.append(frow)

            print(
                f"[{index:4d}/{len(cases)}] {case_id} | pred Dice={predicted_dice:.4f} | "
                f"P(fail)={failure_probability:.4f} | fg={features['pred_fg_fraction']:.5f} | {bucket}"
            )

    # Active-learning ordering: high failure risk first; tie-break by lower predicted Dice.
    active_rows = [row for row in manifest_rows if row["qc_bucket"] == "ACTIVE_LEARN_PRIORITY"]
    active_rows.sort(
        key=lambda row: (
            -float(row["predicted_failure_probability"]),
            float(row["predicted_dice"]),
            str(row["case_id"]),
        )
    )
    active_rank = {row["case_id"]: rank + 1 for rank, row in enumerate(active_rows)}
    for row in manifest_rows:
        row["active_learning_rank"] = active_rank.get(row["case_id"], "")

    # Keep manifest deterministic and easy to scan: bucket, then active rank/risk, then ID.
    bucket_order = {
        "ACTIVE_LEARN_PRIORITY": 0,
        "REVIEW": 1,
        "HIGH_CONFIDENCE_PSEUDO_LABEL": 2,
    }
    manifest_rows.sort(
        key=lambda row: (
            bucket_order[row["qc_bucket"]],
            int(row["active_learning_rank"]) if row["active_learning_rank"] != "" else 10**9,
            -float(row["predicted_failure_probability"]),
            str(row["case_id"]),
        )
    )
    feature_rows.sort(key=lambda row: str(row["case_id"]))

    manifest_path = output_dir / "auto_label_manifest.csv"
    feature_path = output_dir / "unlabeled_qc_features.csv"
    write_csv(manifest_path, manifest_rows)
    feature_fields = ["case_id", "image_path"] + list(bundle["feature_columns"]) + [
        "predicted_dice",
        "predicted_failure_probability",
        "qc_bucket",
    ]
    write_csv(feature_path, feature_rows, feature_fields)

    counts = {
        bucket: int(sum(row["qc_bucket"] == bucket for row in manifest_rows))
        for bucket in BUCKET_DIR
    }
    predicted_dice_values = np.asarray(
        [float(row["predicted_dice"]) for row in manifest_rows], dtype=float
    )
    failure_values = np.asarray(
        [float(row["predicted_failure_probability"]) for row in manifest_rows], dtype=float
    )

    metadata = {
        "version": "auto_label_pool_v1",
        "input_dir": str(input_dir),
        "n_processed": len(manifest_rows),
        "n_source_labeled_excluded": len(source_labeled_ids),
        "n_explicit_human_labeled_excluded_in_input_scan": len(explicit_human),
        "checkpoint": str(checkpoint),
        "qc_bundle": str(qc_bundle_path),
        "policy": str(policy_path),
        "source_manifest": str(source_manifest_path),
        "feature_schema": list(bundle["feature_columns"]),
        "segmentation_threshold": float(args.seg_threshold),
        "prediction_source": "student_teacher_50_50_ensemble",
        "native_geometry_validation": "strict source-grid shape before write + size/spacing/origin/direction after write",
        "policy_thresholds": {
            "high_confidence_failure_probability_lte": accept_p,
            "high_confidence_predicted_dice_gte": accept_dice,
            "active_learning_failure_probability_gte": active_p,
            "active_learning_predicted_dice_lte": active_dice,
        },
        "bucket_counts": counts,
        "pool_predicted_dice_mean": float(np.mean(predicted_dice_values)),
        "pool_predicted_dice_median": float(np.median(predicted_dice_values)),
        "pool_failure_probability_mean": float(np.mean(failure_values)),
        "pool_failure_probability_median": float(np.median(failure_values)),
        "arrays_saved": bool(args.save_probabilities),
        "development_policy_warning": policy_payload.get(
            "warning",
            "Development calibration only; validate on a future locked set before production auto-accept.",
        ),
        "operational_rule": (
            "HIGH_CONFIDENCE_PSEUDO_LABEL is not automatically promoted into training. "
            "Review/locked validation is required before production auto-accept."
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 110)
    print("AUTO-LABEL POOL COMPLETE")
    print(f"Processed:                     {len(manifest_rows)}")
    print(f"HIGH_CONFIDENCE_PSEUDO_LABEL: {counts['HIGH_CONFIDENCE_PSEUDO_LABEL']}")
    print(f"REVIEW:                        {counts['REVIEW']}")
    print(f"ACTIVE_LEARN_PRIORITY:         {counts['ACTIVE_LEARN_PRIORITY']}")
    print(f"Mean predicted Dice:           {np.mean(predicted_dice_values):.4f}")
    print(f"Mean P(failure):               {np.mean(failure_values):.4f}")
    print(f"Manifest:                      {manifest_path}")
    print(f"QC features:                   {feature_path}")
    print(f"Metadata:                      {metadata_path}")
    if active_rows:
        print("\nTop ACTIVE_LEARN_PRIORITY cases:")
        for row in active_rows[:10]:
            print(
                f"  #{active_rank[row['case_id']]:02d} {row['case_id']} | "
                f"pred Dice={float(row['predicted_dice']):.4f} | "
                f"P(fail)={float(row['predicted_failure_probability']):.4f}"
            )
    print("\nNext: inspect a small sample from each bucket, then add diversity-aware selection")
    print("      within ACTIVE_LEARN_PRIORITY for the first human annotation batch.")
    print("=" * 110)


if __name__ == "__main__":
    main()
