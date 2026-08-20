#!/usr/bin/env python3
"""Save leakage-safe hard-v1 OOF predictions as 3D Slicer .seg.nrrd files.

Each hard case is evaluated only with the checkpoint from the frozen original47 fold where
that case was held out. Final62 and Final72 use the same raw Student+EMA 50/50 probability
ensemble at threshold 0.50. Predictions are inverted back to the exact native image grid and
written with 3D Slicer segmentation metadata.

Example:
  python scripts/save_hard_v1_oof_predictions.py \
    --config config_resize128.yaml \
    --image-dir /data/hard_dataset/v1/image \
    --gpu 0

Outputs by default:
  experiments/hard_v1_oof_final62_vs_final72/slicer_predictions/<case_id>/
      final62_oof_pred.seg.nrrd
      final72_oof_pred.seg.nrrd
"""

import csv
import os
import sys
from pathlib import Path


def _consume_option(argv, name):
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
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

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402

try:  # noqa: E402
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("save_hard_v1_oof_predictions.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
from scripts.validate_external_threshold_31 import (  # noqa: E402
    geometry_equal,
    invert_probability_exact,
    normalize_native_probability,
)
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.benchmark_hard_v1_oof_final62_final72 import (  # noqa: E402
    build_fold_map,
    checkpoint_for,
    collect_by_id,
    read_json,
)

DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_FINAL62_CV = Path("experiments/round2_cv_62_translation12")
DEFAULT_FINAL72_CV = Path("experiments/round3_cv_72_translation12")
DEFAULT_OUTPUT_DIR = Path(
    "experiments/hard_v1_oof_final62_vs_final72/slicer_predictions"
)
THRESHOLD = 0.50


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_native_ensemble(config, checkpoint: Path, image_path: Path):
    transform = get_base_transforms(
        config, keys=["image"], is_training=False, apply_strong_aug=False
    )
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=False,
        to_tensor=True,
    )
    loader = DataLoader(
        Dataset([{"image": str(image_path)}], transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    student, teacher = load_models(config, checkpoint, device)
    if teacher is None:
        raise RuntimeError(f"Checkpoint has no EMA teacher: {checkpoint}")
    student.eval()
    teacher.eval()
    inferer = SlidingWindowInferer(
        tuple(config.spatial_size), sw_batch_size=1, overlap=0.25
    )

    with torch.no_grad():
        batch = next(iter(loader))
        image = batch["image"].to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            student_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
            teacher_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
            ensemble_prob = 0.5 * (student_prob + teacher_prob)

        native_prob = invert_probability_exact(
            ensemble_prob, batch, inverse_transform, index=0
        )
        reference_image, prob_zyx = normalize_native_probability(
            native_prob, image_path
        )

    del student, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pred_zyx = (prob_zyx > THRESHOLD).astype(np.uint8)
    return reference_image, pred_zyx


def verify_saved_geometry(path: Path, reference_image):
    saved = sitk.ReadImage(str(path))
    if not geometry_equal(saved, reference_image):
        raise RuntimeError(
            f"Saved prediction geometry does not match source image: {path}"
        )
    arr = np.squeeze(sitk.GetArrayFromImage(saved))
    if arr.ndim != 3:
        raise RuntimeError(f"Saved prediction is not 3D: {path}, shape={arr.shape}")
    return int(np.count_nonzero(arr > 0))


def main():
    p = argparse.ArgumentParser(
        description="Save hard-v1 leakage-safe OOF Final62/Final72 predictions for 3D Slicer"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--image-suffix", default=".mha")
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--final62-cv-dir", default=str(DEFAULT_FINAL62_CV))
    p.add_argument("--final72-cv-dir", default=str(DEFAULT_FINAL72_CV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--model",
        choices=["both", "final62", "final72"],
        default="both",
        help="Which OOF prediction(s) to save; default both.",
    )
    p.add_argument("--expected-count", type=int, default=None)
    args = p.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    images = collect_by_id(image_dir, args.image_suffix)
    hard_ids = sorted(images)
    if not hard_ids:
        raise RuntimeError(f"No *{args.image_suffix} images found under {image_dir}")
    if args.expected_count is not None and len(hard_ids) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} hard cases, found {len(hard_ids)}: {hard_ids}"
        )

    source_manifest = read_json(Path(args.source_manifest))
    fold_by_id = build_fold_map(source_manifest)
    not_oof = sorted(set(hard_ids) - set(fold_by_id))
    if not_oof:
        raise RuntimeError(
            "Only frozen original47 OOF cases are allowed. Not in original47: "
            + ", ".join(not_oof)
        )

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=100)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {config.num_classes}")

    model_specs = []
    if args.model in {"both", "final62"}:
        model_specs.append(("FINAL62", Path(args.final62_cv_dir), "final62_oof_pred.seg.nrrd"))
    if args.model in {"both", "final72"}:
        model_specs.append(("FINAL72", Path(args.final72_cv_dir), "final72_oof_pred.seg.nrrd"))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    print("=" * 118)
    print("HARD-V1 OOF PREDICTION EXPORT — 3D SLICER .seg.nrrd")
    print(f"Cases:       {len(hard_ids)}")
    print(f"Image dir:   {image_dir}")
    print(f"Output dir:  {output_dir}")
    print("Prediction:  Student+EMA 50/50 ensemble @ 0.50")
    print("Geometry:    exact native image grid")
    print("Leakage:     held-out fold checkpoint only")
    print("=" * 118)

    for case_id in hard_ids:
        fold = int(fold_by_id[case_id])
        image_path = images[case_id]
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{case_id} | fold={fold}")

        for model_name, cv_dir, filename in model_specs:
            checkpoint = checkpoint_for(cv_dir, fold)
            reference_image, pred_zyx = infer_native_ensemble(
                config, checkpoint, image_path
            )
            out_path = case_dir / filename
            write_mask_with_spatial_geometry(
                str(out_path),
                pred_zyx,
                reference_image_path=str(image_path),
                segment_name=f"Bladder_{model_name}_OOF",
                segment_id=f"Bladder_{model_name}_OOF",
                label_value=1,
            )
            pred_vox = verify_saved_geometry(out_path, reference_image)
            print(
                f"  {model_name}: {out_path} | foreground={pred_vox} vox | "
                f"checkpoint={checkpoint}"
            )
            manifest_rows.append({
                "case_id": case_id,
                "fold": fold,
                "model": model_name,
                "threshold": THRESHOLD,
                "image_path": str(image_path),
                "checkpoint": str(checkpoint),
                "prediction_path": str(out_path),
                "pred_vox": pred_vox,
                "geometry_verified": 1,
            })

    manifest_path = output_dir / "prediction_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    print("\n" + "=" * 118)
    print("EXPORT COMPLETE")
    print(f"Prediction manifest: {manifest_path}")
    print("In 3D Slicer: load the source .mha, then load either .seg.nrrd as a Segmentation.")
    print("=" * 118)


if __name__ == "__main__":
    main()
