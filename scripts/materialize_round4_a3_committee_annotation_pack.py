#!/usr/bin/env python3
"""Materialize Round-4 selected cases with an A3 committee pre-segmentation for Slicer review.

The Round-4 selector intentionally creates an image-only annotation pack. This script turns the
frozen Round-4 ANNOTATE batch into a separate annotation-ready pack without changing selection,
source images, current HUMAN_GOLD labels, or the selector output.

Prediction definition
---------------------
For each selected case:
  1. run each of the five completed Final72 A3 fold checkpoints,
  2. within each fold use raw 50/50 Student+EMA probability ensemble,
  3. invert each fold probability back to the exact native image grid,
  4. average the five native-grid probabilities,
  5. threshold the five-fold committee mean at 0.50,
  6. write a Slicer-compatible .seg.nrrd using the source image geometry.

The committee pre-segmentation is an ANNOTATION AID ONLY. It is not HUMAN_GOLD until a human
reviewer verifies/corrects and saves the segmentation.

Default layout
--------------
experiments/round4_active_a3_committee_v1/annotation_ready_pack/
  01_<case_id>/
    <image>.mha
    <image_stem>/
      <image_stem>.seg.nrrd
    PROVENANCE.json
  ...
  annotation_pack_manifest.csv

Example
-------
python scripts/materialize_round4_a3_committee_annotation_pack.py \
  --config config_resize128.yaml \
  --gpu 0
"""

import argparse
import csv
import json
import os
import shutil
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

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
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
from scripts.validate_external_threshold_31 import (  # noqa: E402
    geometry_equal,
    invert_probability_exact,
    normalize_native_probability,
)
import scripts.train_supervised_cv as cv  # noqa: E402

ROUND4_DIR = Path("experiments/round4_active_a3_committee_v1")
DEFAULT_BATCH = ROUND4_DIR / "round4_annotation_batch.csv"
DEFAULT_OUTPUT = ROUND4_DIR / "annotation_ready_pack"
A3_DIR = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
FINAL72_AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
EXTERNAL_RESULTS = Path(
    "experiments/external31_round2_qc_gate_v1/external31_locked_gate_case_results.csv"
)
THRESHOLD = 0.50
LABEL_SUFFIX = ".seg.nrrd"


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
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
            writer.writerow({k: row.get(k, "") for k in fields})


def verify_saved_geometry(path: Path, reference_image: sitk.Image):
    saved = sitk.ReadImage(str(path))
    if not geometry_equal(saved, reference_image):
        raise RuntimeError(f"Saved segmentation geometry does not match source image: {path}")
    arr = np.squeeze(sitk.GetArrayFromImage(saved))
    if arr.ndim != 3:
        raise RuntimeError(f"Saved segmentation is not 3D: {path}, shape={arr.shape}")
    return int(np.count_nonzero(arr > 0))


def checkpoint_for(a3_dir: Path, fold: int):
    path = a3_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing A3 Fold{fold} checkpoint: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="Materialize Round4 A3 committee pre-segmentations")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-csv", default=str(DEFAULT_BATCH))
    p.add_argument("--a3-dir", default=str(A3_DIR))
    p.add_argument("--final72-audit", default=str(FINAL72_AUDIT))
    p.add_argument("--external-case-results", default=str(EXTERNAL_RESULTS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--expected-count", type=int, default=10)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.expected_count < 1:
        p.error("--expected-count must be >=1")
    if abs(float(args.threshold) - THRESHOLD) > 1e-8:
        p.error("Round4 annotation preseg is locked to threshold 0.50")

    batch_path = Path(args.batch_csv)
    rows = read_csv(batch_path)
    required = {"case_id", "image_path", "selection_rank", "round4_state"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Round4 batch CSV missing required columns: {sorted(missing)}")
    if len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} Round4 cases, found {len(rows)}")

    ordered = sorted(rows, key=lambda r: int(r["selection_rank"]))
    case_ids = [str(r["case_id"]).strip() for r in ordered]
    if any(not x for x in case_ids) or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Round4 batch contains empty or duplicate case IDs")
    bad_state = [str(r["case_id"]) for r in ordered if str(r["round4_state"]) != "ANNOTATE"]
    if bad_state:
        raise RuntimeError("Only ANNOTATE rows may be materialized: " + ", ".join(bad_state))

    final72 = read_json(Path(args.final72_audit))
    if not final72.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final72 audit is not marked passing")
    human_ids = set(str(x) for x in final72.get("all_current_human_label_ids", []))
    if len(human_ids) != 72:
        raise RuntimeError(f"Expected exactly 72 current HUMAN_GOLD IDs, found {len(human_ids)}")
    overlap_gold = sorted(set(case_ids) & human_ids)
    if overlap_gold:
        raise RuntimeError("Round4 selection overlaps current HUMAN_GOLD: " + ", ".join(overlap_gold))

    external_rows = read_csv(Path(args.external_case_results))
    external_ids = {str(r.get("case_id", "")).strip() for r in external_rows}
    external_ids.discard("")
    if len(external_ids) != 31:
        raise RuntimeError(f"Expected 31 frozen external IDs, found {len(external_ids)}")
    overlap_external = sorted(set(case_ids) & external_ids)
    if overlap_external:
        raise RuntimeError("Round4 selection overlaps frozen external31: " + ", ".join(overlap_external))

    image_by_id = {}
    for row in ordered:
        case_id = str(row["case_id"])
        path = Path(str(row["image_path"]))
        if not path.exists():
            raise FileNotFoundError(f"Missing selected image {case_id}: {path}")
        image_by_id[case_id] = path

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists: {output_dir}. Use --overwrite intentionally.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Round4 A3 committee pack requires prototype DynUNet Student+EMA mode")
    cv.apply_baseline(config, resize_size=128, epochs=100)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {config.num_classes}")

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
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    dataset_items = [{"id": case_id, "image": str(image_by_id[case_id])} for case_id in case_ids]
    probability_sums = {}
    reference_by_id = {}
    per_fold_fg = {case_id: [] for case_id in case_ids}

    print("=" * 124)
    print("ROUND 4 ANNOTATION PRESEG — FINAL72 A3 FIVE-FOLD COMMITTEE")
    print(f"Selected cases: {len(case_ids)}")
    print(f"Device:         {device}")
    print("Per-fold:       Student+EMA 50/50")
    print("Committee:      mean native-grid probability across five folds")
    print("Threshold:      0.50")
    print("Status:         annotation aid only; NOT HUMAN_GOLD")
    print("=" * 124)

    a3_dir = Path(args.a3_dir)
    for fold in range(5):
        checkpoint = checkpoint_for(a3_dir, fold)
        print(f"\n[A3 committee] Fold {fold}/4: {checkpoint}")
        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"A3 Fold{fold} checkpoint has no EMA teacher")
        student.eval()
        teacher.eval()
        loader = DataLoader(Dataset(dataset_items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for idx, batch in enumerate(loader, start=1):
                raw_id = batch.get("id")
                case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
                image_path = image_by_id[case_id]
                image = batch["image"].to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    s_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
                    t_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                    e_prob = 0.5 * (s_prob + t_prob)

                native_prob = invert_probability_exact(e_prob, batch, inverse_transform, index=0)
                reference_image, prob_zyx = normalize_native_probability(native_prob, image_path)
                prob_zyx = np.asarray(prob_zyx, dtype=np.float32)

                if case_id not in probability_sums:
                    probability_sums[case_id] = np.zeros_like(prob_zyx, dtype=np.float64)
                    reference_by_id[case_id] = reference_image
                else:
                    if probability_sums[case_id].shape != prob_zyx.shape:
                        raise RuntimeError(
                            f"Native probability shape changed across folds for {case_id}: "
                            f"{probability_sums[case_id].shape} vs {prob_zyx.shape}"
                        )
                    if not geometry_equal(reference_by_id[case_id], reference_image):
                        raise RuntimeError(f"Native reference geometry changed across folds for {case_id}")

                probability_sums[case_id] += prob_zyx.astype(np.float64)
                per_fold_fg[case_id].append(float(np.mean(prob_zyx > THRESHOLD)))
                print(f"  [{idx:02d}/{len(case_ids):02d}] {case_id}")

        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = []
    for row in ordered:
        rank = int(row["selection_rank"])
        case_id = str(row["case_id"])
        image_path = image_by_id[case_id]
        committee_prob = (probability_sums[case_id] / 5.0).astype(np.float32)
        pred = (committee_prob > THRESHOLD).astype(np.uint8)
        fg_vox = int(pred.sum())
        if fg_vox <= 0:
            print(f"WARNING: {case_id} committee preseg is empty; human must locate bladder from image.")

        case_dir = output_dir / f"{rank:02d}_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        copied_image = case_dir / image_path.name
        shutil.copy2(image_path, copied_image)

        stem = image_path.stem
        seg_dir = case_dir / stem
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_path = seg_dir / f"{stem}{LABEL_SUFFIX}"
        write_mask_with_spatial_geometry(
            str(seg_path),
            pred,
            reference_image_path=str(image_path),
            segment_name="Bladder",
            segment_id="Bladder",
            label_value=1,
            segment_color="0.0 1.0 0.0",
        )
        saved_fg = verify_saved_geometry(seg_path, reference_by_id[case_id])
        if saved_fg != fg_vox:
            raise RuntimeError(f"Foreground count changed while writing {case_id}: {fg_vox} -> {saved_fg}")

        provenance = {
            "case_id": case_id,
            "selection_rank": rank,
            "round": 4,
            "round4_state": "ANNOTATE",
            "selection_profile": row.get("selection_profile", ""),
            "suggested_review_action": row.get("suggested_review_action", ""),
            "human_gold_status": "PENDING_HUMAN_VERIFICATION_OR_CORRECTION",
            "prediction_status": "A3_FIVE_FOLD_COMMITTEE_PRESEG",
            "prediction_definition": "mean of five native-grid A3 fold probabilities; each fold = Student+EMA 50/50; threshold 0.50",
            "source_image": str(image_path),
            "materialized_image": str(copied_image),
            "materialized_segmentation": str(seg_path),
            "committee_pred_vox": fg_vox,
            "committee_pred_fg_fraction": float(np.mean(pred > 0)),
            "fold_pred_fg_fractions": per_fold_fg[case_id],
            "warning": "AI_PRESEG is an annotation aid only and is NOT HUMAN_GOLD until human-reviewed.",
        }
        (case_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        manifest.append({
            "selection_rank": rank,
            "case_id": case_id,
            "selection_profile": row.get("selection_profile", ""),
            "suggested_review_action": row.get("suggested_review_action", ""),
            "source_image": str(image_path),
            "materialized_image": str(copied_image),
            "materialized_segmentation": str(seg_path),
            "committee_pred_vox": fg_vox,
            "committee_pred_fg_fraction": float(np.mean(pred > 0)),
            "fold_pred_fg_fraction_mean": float(np.mean(per_fold_fg[case_id])),
            "fold_pred_fg_fraction_std": float(np.std(per_fold_fg[case_id])),
            "geometry_verified": 1,
            "human_gold_status": "PENDING",
        })
        print(f"[{rank:02d}] {case_id} | preseg={fg_vox} vox | {seg_path}")

    manifest_path = output_dir / "annotation_pack_manifest.csv"
    write_csv(manifest_path, manifest)
    metadata = {
        "version": "round4_a3_committee_annotation_pack_v1",
        "batch_csv": str(batch_path),
        "a3_dir": str(a3_dir),
        "selected_count": len(case_ids),
        "selected_ids": case_ids,
        "current_human_gold_excluded": len(human_ids),
        "external31_excluded": len(external_ids),
        "prediction_definition": "five-fold A3 committee mean of per-fold Student+EMA 50/50 native probabilities",
        "threshold": THRESHOLD,
        "output_dir": str(output_dir),
        "annotation_required": True,
    }
    (output_dir / "annotation_pack_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 124)
    print("ROUND 4 ANNOTATION-READY PACK COMPLETE")
    print(f"Cases:       {len(case_ids)}")
    print(f"Output:      {output_dir}")
    print(f"Manifest:    {manifest_path}")
    print("Next action: open each image + .seg.nrrd in Slicer, verify/correct, then SAVE the segmentation.")
    print("Do not rerun this materializer after human edits; it would rebuild the derived pack.")
    print("=" * 124)


if __name__ == "__main__":
    main()
