#!/usr/bin/env python3
"""Rebuild the selected Round-3 annotation pack without rerunning inference or selection.

For each selected case this script:
1. copies the original image into the case folder,
2. reads the existing AI pre-segmentation mask,
3. rewrites it through the Slicer-compatible NRRD writer using the image as native-geometry reference,
4. stores the segmentation under a folder named after the image stem.

Example layout for FusedVolume.mha:

annotation_pack/
└── 01_<case_id>/
    ├── FusedVolume.mha
    ├── FusedVolume/
    │   └── FusedVolume.seg.nrrd
    └── PROVENANCE.json

No segmentation inference, QC prediction, or active-learning selection is rerun.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.data.nrrd_utils import write_mask_with_spatial_geometry

DEFAULT_ROUND3_DIR = Path("experiments/round3_failure_aware_v1")
DEFAULT_BATCH_CSV = DEFAULT_ROUND3_DIR / "active_learning_batch_round3_failure_aware.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ROUND3_DIR / "annotation_pack"


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Round-3 batch CSV is empty: {path}")
    return rows


def image_stem(path: Path) -> str:
    """Return image basename without its final extension (FusedVolume.mha -> FusedVolume)."""
    return path.stem


def read_binary_mask(seg_path: Path) -> np.ndarray:
    """Read an existing segmentation as a 3D Z,Y,X uint8 mask."""
    seg_img = sitk.ReadImage(str(seg_path))
    if seg_img.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D segmentation, got dimension={seg_img.GetDimension()}: {seg_path}"
        )
    arr = sitk.GetArrayFromImage(seg_img)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation array, got shape={arr.shape}: {seg_path}")
    return (arr > 0).astype(np.uint8)


def verify_rewritten_seg(seg_path: Path, image_path: Path):
    """Verify geometry and required Slicer segment metadata after writing."""
    seg_img = sitk.ReadImage(str(seg_path))
    ref_img = sitk.ReadImage(str(image_path))

    if seg_img.GetSize() != ref_img.GetSize():
        raise RuntimeError(
            f"Size mismatch after rewrite: seg={seg_img.GetSize()} ref={ref_img.GetSize()}"
        )
    if not np.allclose(seg_img.GetSpacing(), ref_img.GetSpacing(), rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"Spacing mismatch after rewrite: seg={seg_img.GetSpacing()} ref={ref_img.GetSpacing()}"
        )
    if not np.allclose(seg_img.GetOrigin(), ref_img.GetOrigin(), rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"Origin mismatch after rewrite: seg={seg_img.GetOrigin()} ref={ref_img.GetOrigin()}"
        )
    if not np.allclose(seg_img.GetDirection(), ref_img.GetDirection(), rtol=0.0, atol=1e-8):
        raise RuntimeError("Direction mismatch after rewrite")

    expected_metadata = {
        "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ReferenceImageExtentOffset": "0 0 0",
        "Segment0_ID": "Bladder",
        "Segment0_LabelValue": "1",
        "Segment0_Layer": "0",
        "Segment0_Color": "0.0 1.0 0.0",
        "Segment0_Name": "Bladder",
        "Segment0_Tags": "|",
    }
    missing = []
    mismatched = []
    for key, expected in expected_metadata.items():
        if not seg_img.HasMetaDataKey(key):
            missing.append(key)
            continue
        actual = seg_img.GetMetaData(key)
        if str(actual) != str(expected):
            mismatched.append((key, actual, expected))

    if missing:
        raise RuntimeError(f"Missing Slicer metadata in {seg_path}: {missing}")
    if mismatched:
        raise RuntimeError(f"Unexpected Slicer metadata in {seg_path}: {mismatched}")

    sx, sy, sz = ref_img.GetSize()
    expected_extent = f"0 {sx - 1} 0 {sy - 1} 0 {sz - 1}"
    if not seg_img.HasMetaDataKey("Segment0_Extent"):
        raise RuntimeError(f"Missing Segment0_Extent in {seg_path}")
    if seg_img.GetMetaData("Segment0_Extent") != expected_extent:
        raise RuntimeError(
            f"Segment0_Extent mismatch in {seg_path}: "
            f"{seg_img.GetMetaData('Segment0_Extent')} != {expected_extent}"
        )


def main():
    p = argparse.ArgumentParser(
        description="Rewrite and rematerialize the already-selected Round-3 annotation pack"
    )
    p.add_argument("--batch-csv", default=str(DEFAULT_BATCH_CSV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--segment-name", default="Bladder")
    p.add_argument("--segment-color", default="0.0 1.0 0.0")
    args = p.parse_args()

    batch_csv = Path(args.batch_csv)
    output_dir = Path(args.output_dir)
    rows = read_csv(batch_csv)

    required = {"case_id", "image_path", "segmentation_path", "selection_rank"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise RuntimeError(
            f"Round-3 batch CSV is missing required columns: {sorted(missing_columns)}"
        )

    # This utility is intentionally destructive only to the derived annotation pack.
    # It never changes the source image, source AI prediction, selection CSV, or QC outputs.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print("=" * 110)
    print("ROUND-3 ANNOTATION PACK — REWRITE + REMATERIALIZE")
    print(f"Batch CSV:   {batch_csv}")
    print(f"Output pack: {output_dir}")
    print(f"Cases:       {len(rows)}")
    print("No inference or active-learning selection is rerun.")
    print("=" * 110)

    ordered = sorted(rows, key=lambda r: int(r["selection_rank"]))
    for row in ordered:
        rank = int(row["selection_rank"])
        case_id = str(row["case_id"])
        image_path = Path(str(row["image_path"]))
        source_seg_path = Path(str(row["segmentation_path"]))

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {case_id}: {image_path}")
        if not source_seg_path.exists():
            raise FileNotFoundError(f"Missing source AI preseg for {case_id}: {source_seg_path}")

        case_dir = output_dir / f"{rank:02d}_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        copied_image = case_dir / image_path.name
        shutil.copy2(image_path, copied_image)

        stem = image_stem(image_path)
        seg_dir = case_dir / stem
        seg_dir.mkdir(parents=True, exist_ok=True)
        rewritten_seg = seg_dir / f"{stem}.seg.nrrd"

        mask = read_binary_mask(source_seg_path)
        write_mask_with_spatial_geometry(
            str(rewritten_seg),
            mask,
            reference_image_path=str(image_path),
            segment_name=args.segment_name,
            segment_id=args.segment_name,
            label_value=1,
            segment_color=args.segment_color,
        )
        verify_rewritten_seg(rewritten_seg, image_path)

        provenance = {
            "case_id": case_id,
            "selection_rank": rank,
            "status": "ACTIVE_LEARNING_SELECTED",
            "prediction_status": "AI_PRESEG",
            "human_gold_status": "PENDING",
            "source_image": str(image_path),
            "source_ai_preseg": str(source_seg_path),
            "materialized_image": str(copied_image),
            "materialized_segmentation": str(rewritten_seg),
            "segment_name": args.segment_name,
            "segment_label_value": 1,
            "segment_color": args.segment_color,
            "failure_proxy": row.get("failure_proxy", ""),
            "suggested_review_action": row.get("suggested_review_action", ""),
            "source_prediction": "Final62 locked Student+EMA 50/50 ensemble @ threshold 0.50",
            "rewrite": "Slicer-compatible .seg.nrrd with native geometry from source image",
            "warning": "AI_PRESEG is not HUMAN_GOLD until human verification/correction.",
        }
        provenance_path = case_dir / "PROVENANCE.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        manifest.append({
            "selection_rank": rank,
            "case_id": case_id,
            "image": str(copied_image),
            "segmentation_folder": str(seg_dir),
            "segmentation": str(rewritten_seg),
            "source_segmentation": str(source_seg_path),
        })

        print(
            f"[{rank:02d}] {case_id} | {copied_image.name} | "
            f"{stem}/{rewritten_seg.name} | verified"
        )

    manifest_path = output_dir / "annotation_pack_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print("=" * 110)
    print(f"Rebuilt annotation pack: {output_dir}")
    print(f"Manifest:                {manifest_path}")
    print("All rewritten segmentations passed native-geometry and Slicer-metadata verification.")
    print("=" * 110)


if __name__ == "__main__":
    main()
