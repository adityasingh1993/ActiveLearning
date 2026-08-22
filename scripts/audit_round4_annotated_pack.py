#!/usr/bin/env python3
"""Audit the 10 human-reviewed Round-4 A3-committee annotation-pack segmentations.

This script is non-destructive by default. It verifies that:
- the Round-4 batch contains exactly the selected ANNOTATE case IDs,
- the annotation-ready pack contains exactly those ranked case directories,
- every case has exactly one copied .mha image and one edited .seg.nrrd,
- image/segmentation size, spacing, origin, and direction match,
- every human-reviewed segmentation is readable, 3D, and non-empty,
- no Round-4 selected case overlaps the previously audited Final72 HUMAN_GOLD set.

An unchanged committee pre-segmentation is allowed: human verification without voxel edits is a
valid annotation outcome. This audit therefore checks integrity/geometry, not whether editing
occurred.

By default nothing is copied into a training label folder. After the audit passes, optionally use
--promote-to <label_dir> to copy each passing mask as <case_id>.seg.nrrd. Promotion preflights the
whole batch and refuses to overwrite existing labels unless --overwrite-existing is explicitly set.

Default inputs
--------------
experiments/round4_active_a3_committee_v1/round4_annotation_batch.csv
experiments/round4_active_a3_committee_v1/annotation_ready_pack/
experiments/round3_supervised_72_translation12/round3_label_audit.json

Example (audit only; recommended first)
---------------------------------------
python scripts/audit_round4_annotated_pack.py

Example (only after audit PASS)
-------------------------------
python scripts/audit_round4_annotated_pack.py --promote-to /path/to/final82/label
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROUND4_DIR = Path("experiments/round4_active_a3_committee_v1")
DEFAULT_BATCH = ROUND4_DIR / "round4_annotation_batch.csv"
DEFAULT_PACK = ROUND4_DIR / "annotation_ready_pack"
DEFAULT_PREVIOUS_AUDIT = Path(
    "experiments/round3_supervised_72_translation12/round3_label_audit.json"
)
DEFAULT_OUTPUT_DIR = ROUND4_DIR / "human_annotation_audit"
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
            writer.writerow({key: row.get(key, "") for key in fields})


def geometry_flags(image: sitk.Image, seg: sitk.Image):
    return {
        "size_match": tuple(image.GetSize()) == tuple(seg.GetSize()),
        "spacing_match": bool(
            np.allclose(image.GetSpacing(), seg.GetSpacing(), rtol=0.0, atol=1e-8)
        ),
        "origin_match": bool(
            np.allclose(image.GetOrigin(), seg.GetOrigin(), rtol=0.0, atol=1e-8)
        ),
        "direction_match": bool(
            np.allclose(image.GetDirection(), seg.GetDirection(), rtol=0.0, atol=1e-8)
        ),
    }


def binary_array(image: sitk.Image, path: Path):
    arr = np.asarray(sitk.GetArrayFromImage(image))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D segmentation array, got {arr.shape}: {path}")
    return arr > 0


def unique_one(paths, description, case_id):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} for {case_id}, found {len(paths)}: "
            + ", ".join(str(x) for x in paths[:10])
        )
    return paths[0]


def main():
    p = argparse.ArgumentParser(description="Audit human-reviewed Round-4 annotation pack")
    p.add_argument("--batch-csv", default=str(DEFAULT_BATCH))
    p.add_argument("--annotation-pack", default=str(DEFAULT_PACK))
    p.add_argument("--previous-audit", default=str(DEFAULT_PREVIOUS_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=10)
    p.add_argument(
        "--promote-to",
        default=None,
        help="Optional HUMAN_GOLD label directory. Audit must pass before any copy occurs.",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing existing <case_id>.seg.nrrd in --promote-to. Use intentionally.",
    )
    args = p.parse_args()

    batch_path = Path(args.batch_csv)
    pack = Path(args.annotation_pack)
    previous_audit_path = Path(args.previous_audit)
    output_dir = Path(args.output_dir)

    if args.expected_count < 1:
        p.error("--expected-count must be >=1")
    if not pack.exists():
        raise FileNotFoundError(pack)

    batch = read_csv(batch_path)
    required = {"case_id", "selection_rank", "round4_state"}
    missing = required - set(batch[0])
    if missing:
        raise RuntimeError(f"Round-4 batch CSV missing columns: {sorted(missing)}")
    if len(batch) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} Round-4 selected cases, batch has {len(batch)}"
        )

    bad_states = [
        str(r.get("case_id", ""))
        for r in batch
        if str(r.get("round4_state", "")).strip().upper() != "ANNOTATE"
    ]
    if bad_states:
        raise RuntimeError(
            "Round-4 batch contains non-ANNOTATE cases: " + ", ".join(bad_states)
        )

    case_ids = [str(r["case_id"]).strip() for r in batch]
    if any(not x for x in case_ids) or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Round-4 batch contains empty or duplicate case IDs")

    previous = read_json(previous_audit_path)
    if not previous.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Previous Final72 HUMAN_GOLD audit is not marked passing")
    prior_gold = set(str(x) for x in previous.get("all_current_human_label_ids", []))
    if len(prior_gold) != 72:
        raise RuntimeError(f"Expected 72 prior HUMAN_GOLD IDs, found {len(prior_gold)}")
    overlap = sorted(set(case_ids) & prior_gold)
    if overlap:
        raise RuntimeError(
            "Round-4 selected cases overlap prior Final72 HUMAN_GOLD: " + ", ".join(overlap)
        )

    expected_dirs = {
        f"{int(r['selection_rank']):02d}_{str(r['case_id']).strip()}" for r in batch
    }
    actual_dirs = {x.name for x in pack.iterdir() if x.is_dir()}
    missing_dirs = sorted(expected_dirs - actual_dirs)
    unexpected_dirs = sorted(actual_dirs - expected_dirs)
    if missing_dirs or unexpected_dirs:
        raise RuntimeError(
            "Annotation-ready pack does not match frozen Round-4 batch.\n"
            f"Missing: {missing_dirs}\nUnexpected: {unexpected_dirs}"
        )

    rows = []
    failures = []
    by_case = {}

    print("=" * 118)
    print("ROUND-4 HUMAN ANNOTATION AUDIT — NON-DESTRUCTIVE")
    print(f"Batch:             {batch_path}")
    print(f"Annotation pack:   {pack}")
    print(f"Selected cases:    {len(batch)}")
    print(f"Prior HUMAN_GOLD:  {len(prior_gold)}")
    print("No source image or annotation file will be modified.")
    print("=" * 118)

    for r in sorted(batch, key=lambda x: int(x["selection_rank"])):
        rank = int(r["selection_rank"])
        case_id = str(r["case_id"]).strip()
        case_dir = pack / f"{rank:02d}_{case_id}"

        try:
            image_path = unique_one(case_dir.glob("*.mha"), "top-level .mha image", case_id)
            seg_path = unique_one(
                case_dir.rglob(f"*{LABEL_SUFFIX}"), "human-reviewed .seg.nrrd", case_id
            )

            image = sitk.ReadImage(str(image_path))
            seg = sitk.ReadImage(str(seg_path))
            if image.GetDimension() != 3 or seg.GetDimension() != 3:
                raise RuntimeError(
                    f"Expected 3D image/seg, got image={image.GetDimension()}D "
                    f"seg={seg.GetDimension()}D"
                )

            geom = geometry_flags(image, seg)
            bad_geom = [k for k, v in geom.items() if not v]
            if bad_geom:
                raise RuntimeError("Image/segmentation geometry mismatch: " + ", ".join(bad_geom))

            mask = binary_array(seg, seg_path)
            fg = int(mask.sum())
            if fg <= 0:
                raise RuntimeError("Human-reviewed segmentation contains zero foreground voxels")

            voxel_volume_mm3 = float(np.prod(np.asarray(seg.GetSpacing(), dtype=float)))
            volume_mm3 = float(fg * voxel_volume_mm3)
            volume_ml = float(volume_mm3 / 1000.0)

            row = {
                "selection_rank": rank,
                "case_id": case_id,
                "selection_profile": str(r.get("selection_profile", "")),
                "suggested_review_action": str(r.get("suggested_review_action", "")),
                "status": "HUMAN_REVIEWED_AUDIT_PASS",
                "image_path": str(image_path),
                "human_segmentation_path": str(seg_path),
                "foreground_voxels": fg,
                "foreground_volume_ml": volume_ml,
                "size_match": int(geom["size_match"]),
                "spacing_match": int(geom["spacing_match"]),
                "origin_match": int(geom["origin_match"]),
                "direction_match": int(geom["direction_match"]),
                "geometry_match": 1,
                "audit_ok": 1,
                "audit_error": "",
            }
            by_case[case_id] = row
            print(
                f"[{rank:02d}] {case_id} | FG={fg} vox | "
                f"volume={volume_ml:.5f} mL | geometry PASS"
            )
        except Exception as exc:
            row = {
                "selection_rank": rank,
                "case_id": case_id,
                "selection_profile": str(r.get("selection_profile", "")),
                "suggested_review_action": str(r.get("suggested_review_action", "")),
                "status": "HUMAN_REVIEWED_AUDIT_FAIL",
                "image_path": "",
                "human_segmentation_path": "",
                "foreground_voxels": "",
                "foreground_volume_ml": "",
                "size_match": 0,
                "spacing_match": 0,
                "origin_match": 0,
                "direction_match": 0,
                "geometry_match": 0,
                "audit_ok": 0,
                "audit_error": str(exc),
            }
            failures.append(f"{case_id}: {exc}")
            print(f"[{rank:02d}] {case_id} | FAIL | {exc}")
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round4_human_annotation_audit.csv"
    json_path = output_dir / "round4_human_annotation_audit.json"
    write_csv(csv_path, rows)

    metadata = {
        "version": "round4_human_annotation_pack_audit_v1",
        "batch_csv": str(batch_path),
        "annotation_pack": str(pack),
        "previous_audit": str(previous_audit_path),
        "expected_round4_labels": int(args.expected_count),
        "selected_ids": case_ids,
        "prior_human_gold_count": len(prior_gold),
        "prior_human_gold_overlap": overlap,
        "n_audit_pass": int(sum(int(r.get("audit_ok", 0)) for r in rows)),
        "n_audit_fail": int(len(failures)),
        "all_round4_annotations_passed": not failures and len(rows) == args.expected_count,
        "promotion_requested": args.promote_to is not None,
        "promotion_destination": args.promote_to,
        "training_status_if_pass": "FINAL82_HUMAN_GOLD_READY_FOR_PROMOTION",
        "note": (
            "Audit validates label integrity and native image geometry. An unchanged A3 committee "
            "pre-segmentation may be valid if it was human-verified."
        ),
    }

    if failures:
        metadata["promoted_count"] = 0
        metadata["promoted_ids"] = []
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Round-4 human annotation audit FAILED. Nothing was promoted.\n"
            + "\n".join(failures)
        )

    promoted = []
    if args.promote_to is not None:
        destination = Path(args.promote_to)
        destination.mkdir(parents=True, exist_ok=True)

        collisions = []
        for case_id in case_ids:
            target = destination / f"{case_id}{LABEL_SUFFIX}"
            if target.exists() and not args.overwrite_existing:
                collisions.append(str(target))
        if collisions:
            metadata["promoted_count"] = 0
            metadata["promoted_ids"] = []
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise RuntimeError(
                "Promotion aborted before copying: destination labels already exist. "
                "Use --overwrite-existing only if replacement is intentional.\n"
                + "\n".join(collisions)
            )

        for case_id in case_ids:
            src = Path(by_case[case_id]["human_segmentation_path"])
            dst = destination / f"{case_id}{LABEL_SUFFIX}"
            shutil.copy2(src, dst)
            promoted.append({
                "case_id": case_id,
                "source": str(src),
                "destination": str(dst),
            })
        write_csv(output_dir / "round4_human_gold_promotion.csv", promoted)
        metadata["promoted_count"] = len(promoted)
        metadata["promoted_ids"] = [x["case_id"] for x in promoted]
    else:
        metadata["promoted_count"] = 0
        metadata["promoted_ids"] = []

    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 118)
    print("ROUND-4 HUMAN ANNOTATION AUDIT: PASS")
    print(f"Passing annotations: {len(rows)}/{args.expected_count}")
    print("Geometry/non-empty:  PASS for all")
    print("Prior-label overlap: 0")
    print(f"Audit CSV:            {csv_path}")
    print(f"Audit metadata:       {json_path}")
    if args.promote_to is None:
        print("Promotion:             NOT REQUESTED (safe audit-only mode)")
        print("Next: inspect this PASS, then promote into an explicit Final82 label directory.")
    else:
        print(f"Promotion:             {len(promoted)} HUMAN_GOLD labels -> {args.promote_to}")
    print("=" * 118)


if __name__ == "__main__":
    main()
