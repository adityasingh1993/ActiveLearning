#!/usr/bin/env python3
"""Build a non-destructive Fold-2 annotation review pack for targeted bladder-label QC.

The pack deliberately separates annotation review into two stages:

  01_blinded/<case_id>/
      image.mha
      current_human_gold.seg.nrrd

  02_unblinded/<case_id>/
      image.mha
      current_human_gold.seg.nrrd
      a0_oof_pred.seg.nrrd
      a2_oof_pred.seg.nrrd
      a3_oof_pred.seg.nrrd

Review 01_blinded FIRST. Only after recording KEEP / CORRECT / UNCERTAIN should the reviewer
open 02_unblinded to compare the model predictions. This reduces model-induced annotation bias.

All model predictions are leakage-safe OOF predictions from the exact frozen Fold-2 checkpoint:
  A0 = Final72, translation +/-12 p=.8, no LR flip, DiceCE
  A2 = Final72, translation +/-4 p=.5, no LR flip, DiceCE
  A3 = Final72, translation +/-4 p=.5 + LR flip p=.5, DiceCE
All use the raw Student+EMA 50/50 ensemble at threshold 0.50.

IMPORTANT
---------
- This script never edits source images or HUMAN_GOLD labels.
- Native-grid prediction exports are intended for VISUAL REVIEW in 3D Slicer.
- Official quantitative metrics are copied from each model's cv_results.csv. Do not replace
  those official metrics with Dice recomputed from exported native masks; this project has a
  known direct-vs-native-export discrepancy under investigation.
- By default only the five priority Fold-2 cases are materialized. Use --all-fold2 to include
  every frozen Fold-2 validation case.

Example
-------
python scripts/build_fold2_annotation_review_pack.py \
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.benchmark_hard_v1_oof_final62_final72 import (  # noqa: E402
    build_fold_map,
    checkpoint_for,
    read_json,
)
from scripts.save_hard_v1_oof_predictions import (  # noqa: E402
    infer_native_ensemble,
    verify_saved_geometry,
)

SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
A0_CV = Path("experiments/round3_cv_72_translation12")
A2_CV = Path("experiments/final72_screen_a2_translation4_p05")
A3_CV = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
DEFAULT_OUTPUT = Path("experiments/fold2_annotation_review_pack")
FOLD = 2
THRESHOLD = 0.50

PRIORITY_CASES = [
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c",
    "b4dc115b84f9e59239b3bbc087259e8331ebe131d691114b8d451357f0727519",
    "c728da3b46126de213d7fb4cc20213fcfab444d8455565096802cab8d0496b90",
    "e04536ee28c370ae6ee41a935464a4e651a4f3a415056c3df73398f266060813",
    "fc9cec27ee37f36d45c323a5001dc76e9cd588a98fc8d7d1c96d7bb00e9ca1ed",
]

MODEL_SPECS = {
    "A0": {
        "cv_dir": A0_CV,
        "filename": "a0_oof_pred.seg.nrrd",
        "segment_name": "Bladder_A0_OOF",
        "description": "Final72 +/-12 vox p=.8, no LR flip, DiceCE",
    },
    "A2": {
        "cv_dir": A2_CV,
        "filename": "a2_oof_pred.seg.nrrd",
        "segment_name": "Bladder_A2_OOF",
        "description": "Final72 +/-4 vox p=.5, no LR flip, DiceCE",
    },
    "A3": {
        "cv_dir": A3_CV,
        "filename": "a3_oof_pred.seg.nrrd",
        "segment_name": "Bladder_A3_OOF",
        "description": "Final72 +/-4 vox p=.5 + LR flip p=.5, DiceCE",
    },
}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def copy_exact(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        raise RuntimeError(f"Copy verification failed: {src} -> {dst}")


def geometry_equal(a, b, atol=1e-6):
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=1e-6, atol=atol)
    )


def verify_gt_overlaps_image_grid(image_path: Path, label_path: Path):
    """Lightweight geometry sanity check without modifying either source file.

    Slicer .seg.nrrd may encode segmentation geometry differently from a plain ITK image, so
    this check is intentionally limited to successful 3D readability and non-empty foreground.
    Prediction files are separately required to match the source image grid exactly.
    """
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetDimension() != 3 or label.GetDimension() != 3:
        raise RuntimeError(
            f"Expected 3D image/GT: image={image.GetDimension()}D label={label.GetDimension()}D"
        )
    arr = np.squeeze(sitk.GetArrayFromImage(label))
    if arr.ndim != 3 or int(np.count_nonzero(arr > 0)) == 0:
        raise RuntimeError(f"GT is empty or not 3D: {label_path} shape={arr.shape}")
    return image, int(np.count_nonzero(arr > 0))


def official_metrics_by_model(selected_ids):
    out = {}
    for model, spec in MODEL_SPECS.items():
        rows = read_csv(Path(spec["cv_dir"]) / "cv_results.csv")
        by_id = {str(r["case_id"]): r for r in rows}
        missing = sorted(set(selected_ids) - set(by_id))
        if missing:
            raise RuntimeError(f"{model} cv_results.csv missing selected IDs: {missing}")
        out[model] = by_id
    return out


def write_review_instructions(output_dir: Path, selected_ids):
    text = f"""FOLD-2 BLADDER ANNOTATION REVIEW PACK
======================================

PURPOSE
-------
Targeted annotation QC for frozen Fold 2 before any additional model tuning.

REVIEW ORDER
------------
1. Open ONLY 01_blinded first.
2. For each case, load image.mha + current_human_gold.seg.nrrd in 3D Slicer.
3. Review axial/coronal/sagittal views and record one decision in blinded_review_sheet.csv:
       KEEP       = annotation is consistent with the dataset convention
       CORRECT    = annotation has a clear labeling error/incomplete boundary/wrong structure
       UNCERTAIN  = image quality is insufficient to decide confidently
4. Record notes BEFORE opening model predictions.
5. Only then open 02_unblinded and inspect A0/A2/A3 OOF predictions.
6. Model disagreement must NOT by itself justify changing HUMAN_GOLD.
7. If a label is corrected, save it to a NEW dataset/version. Do not overwrite Final72 in place.

WHAT TO CHECK
-------------
- Is the annotated structure definitely bladder?
- Is the complete visible bladder/lumen captured using the same convention as the rest of the set?
- Are weak/ambiguous boundaries treated consistently across slices?
- Any missing islands/slices or accidental adjacent hypoechoic tissue?
- Any orientation/geometry issue?
- Is the bladder unusually small or poorly visualized?

SELECTED CASES ({len(selected_ids)})
--------------------
""" + "\n".join(f"- {x}" for x in selected_ids) + "\n"
    (output_dir / "README_REVIEW_ORDER.txt").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Build blinded/unblinded frozen Fold-2 annotation review pack")
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    p.add_argument("--a0-cv-dir", default=str(A0_CV))
    p.add_argument("--a2-cv-dir", default=str(A2_CV))
    p.add_argument("--a3-cv-dir", default=str(A3_CV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--all-fold2", action="store_true", help="Include every frozen Fold-2 held-out case")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing review-pack directory")
    args = p.parse_args()

    # Respect CLI model-directory overrides without changing the locked model definitions.
    MODEL_SPECS["A0"]["cv_dir"] = Path(args.a0_cv_dir)
    MODEL_SPECS["A2"]["cv_dir"] = Path(args.a2_cv_dir)
    MODEL_SPECS["A3"]["cv_dir"] = Path(args.a3_cv_dir)

    source_manifest = read_json(Path(args.source_manifest))
    fold_by_id = build_fold_map(source_manifest)
    fold2_ids = sorted([case_id for case_id, fold in fold_by_id.items() if int(fold) == FOLD])
    if not fold2_ids:
        raise RuntimeError("Frozen manifest contains no Fold-2 validation cases")

    selected_ids = fold2_ids if args.all_fold2 else list(PRIORITY_CASES)
    not_fold2 = sorted([x for x in selected_ids if fold_by_id.get(x) != FOLD])
    if not_fold2:
        raise RuntimeError(f"Refusing pack: requested cases are not frozen Fold 2: {not_fold2}")

    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, resize_size=128, epochs=100)
    cases = {str(c["id"]): c for c in cv.collect_cases(config)}
    missing_cases = sorted(set(selected_ids) - set(cases))
    if missing_cases:
        raise RuntimeError(f"Current Final72 discovery is missing review cases: {missing_cases}")

    official = official_metrics_by_model(selected_ids)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Review pack already exists: {output_dir}. Use --overwrite only if you intend to rebuild it."
            )
        shutil.rmtree(output_dir)
    blinded_root = output_dir / "01_blinded"
    unblinded_root = output_dir / "02_unblinded"
    blinded_root.mkdir(parents=True, exist_ok=True)
    unblinded_root.mkdir(parents=True, exist_ok=True)

    write_review_instructions(output_dir, selected_ids)

    blinded_sheet = []
    official_rows = []
    manifest_rows = []

    print("=" * 118)
    print("FOLD-2 ANNOTATION REVIEW PACK")
    print(f"Cases:          {len(selected_ids)}")
    print(f"Frozen fold:    {FOLD}")
    print(f"Output:         {output_dir}")
    print("Review order:   01_blinded FIRST -> record decision -> 02_unblinded")
    print("Predictions:    OOF Student+EMA 50/50 @ 0.50, visual-review only")
    print("Source labels:  NEVER MODIFIED")
    print("=" * 118)

    for review_index, case_id in enumerate(selected_ids, start=1):
        case = cases[case_id]
        image_path = Path(case["image"])
        label_path = Path(case["label"])
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing source for {case_id}: image={image_path}, label={label_path}")

        reference_image, gt_vox = verify_gt_overlaps_image_grid(image_path, label_path)

        bdir = blinded_root / case_id
        udir = unblinded_root / case_id
        copy_exact(image_path, bdir / "image.mha")
        copy_exact(label_path, bdir / "current_human_gold.seg.nrrd")
        copy_exact(image_path, udir / "image.mha")
        copy_exact(label_path, udir / "current_human_gold.seg.nrrd")

        blinded_sheet.append({
            "review_order": review_index,
            "case_id": case_id,
            "frozen_fold": FOLD,
            "initial_blinded_decision": "",
            "annotation_issue_type": "",
            "confidence": "",
            "blinded_notes": "",
            "post_unblinding_decision": "",
            "post_unblinding_notes": "",
        })

        print(f"\n[{review_index}/{len(selected_ids)}] {case_id} | GT foreground={gt_vox} vox")

        for model, spec in MODEL_SPECS.items():
            checkpoint = checkpoint_for(Path(spec["cv_dir"]), FOLD)
            pred_reference, pred_zyx = infer_native_ensemble(config, checkpoint, image_path)
            if not geometry_equal(pred_reference, reference_image):
                raise RuntimeError(f"{case_id} {model}: inference reference geometry differs from source image")

            out_path = udir / spec["filename"]
            write_mask_with_spatial_geometry(
                str(out_path),
                pred_zyx,
                reference_image_path=str(image_path),
                segment_name=spec["segment_name"],
                segment_id=spec["segment_name"],
                label_value=1,
            )
            pred_vox = verify_saved_geometry(out_path, reference_image)
            m = official[model][case_id]
            print(
                f"  {model}: Dice={float(m['dice']):.4f} | Prec={float(m['precision']):.4f} | "
                f"Rec={float(m['recall']):.4f} | exported_fg={pred_vox} vox"
            )

            official_rows.append({
                "case_id": case_id,
                "fold": FOLD,
                "model": model,
                "model_description": spec["description"],
                "official_dice": float(m["dice"]),
                "official_precision": float(m["precision"]),
                "official_recall": float(m["recall"]),
                "official_hd95_mm": float(m["hd95"]),
                "official_rve_abs_pct": float(m["rve"]),
                "official_metric_source": str(Path(spec["cv_dir"]) / "cv_results.csv"),
                "checkpoint": str(checkpoint),
                "threshold": THRESHOLD,
                "exported_prediction": str(out_path),
                "exported_pred_vox": pred_vox,
                "warning": "Export is for visual QC; official metrics come from cv_results.csv",
            })

        manifest_rows.append({
            "review_order": review_index,
            "case_id": case_id,
            "fold": FOLD,
            "source_image": str(image_path),
            "source_human_gold": str(label_path),
            "blinded_dir": str(bdir),
            "unblinded_dir": str(udir),
            "source_gt_vox": gt_vox,
        })

    write_csv(output_dir / "blinded_review_sheet.csv", blinded_sheet)
    write_csv(output_dir / "official_oof_metrics.csv", official_rows)
    write_csv(output_dir / "pack_manifest.csv", manifest_rows)

    plan = {
        "version": "fold2_annotation_review_pack_v1",
        "fold": FOLD,
        "selected_case_ids": selected_ids,
        "all_fold2": bool(args.all_fold2),
        "review_protocol": "blinded image+GT first; model predictions only after initial decision",
        "prediction_definition": "raw Student+EMA 50/50 ensemble at threshold 0.50",
        "models": {
            model: {
                "cv_dir": str(spec["cv_dir"]),
                "description": spec["description"],
            }
            for model, spec in MODEL_SPECS.items()
        },
        "source_manifest": str(args.source_manifest),
        "non_destructive": True,
        "external31_access": False,
        "native_export_metrics_warning": (
            "Use official_oof_metrics.csv for quantitative values. Native exported masks are for visual review only."
        ),
    }
    (output_dir / "pack_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("PACK COMPLETE")
    print(f"1. Read:       {output_dir / 'README_REVIEW_ORDER.txt'}")
    print(f"2. Review:     {blinded_root}")
    print(f"3. Record:     {output_dir / 'blinded_review_sheet.csv'}")
    print(f"4. Then open:  {unblinded_root}")
    print(f"OOF metrics:   {output_dir / 'official_oof_metrics.csv'}")
    print("No source image or HUMAN_GOLD label was modified.")
    print("=" * 118)


if __name__ == "__main__":
    main()
