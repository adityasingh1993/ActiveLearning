#!/usr/bin/env python3
"""Audit the 10 human-corrected Round-3 annotation-pack segmentations.

This script is intentionally non-destructive by default. It verifies that:
- the annotation pack contains exactly the frozen Round-3 selected case IDs,
- each expected case has one materialized image and one edited .seg.nrrd,
- image/segmentation size, spacing, origin, and direction match exactly,
- each human segmentation is readable, 3D, and contains foreground,
- no selected case overlaps the prior Final62 HUMAN_GOLD set.

It also compares the human-corrected mask with the original AI pre-segmentation when that
source file is available. This comparison is descriptive only: an unchanged mask is allowed
because a human may legitimately verify the AI mask without editing it.

By default nothing is copied into the central training label folder. After the audit passes,
use --promote-to <label_dir> to copy each passing human mask as <case_id>.seg.nrrd. Promotion
refuses to overwrite an existing label unless --overwrite-existing is explicitly supplied.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_ROUND3_DIR = Path("experiments/round3_failure_aware_v1")
DEFAULT_BATCH = DEFAULT_ROUND3_DIR / "active_learning_batch_round3_failure_aware.csv"
DEFAULT_PACK = DEFAULT_ROUND3_DIR / "annotation_pack"
DEFAULT_PREVIOUS_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = DEFAULT_ROUND3_DIR / "human_annotation_audit"
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
        "spacing_match": bool(np.allclose(image.GetSpacing(), seg.GetSpacing(), rtol=0.0, atol=1e-8)),
        "origin_match": bool(np.allclose(image.GetOrigin(), seg.GetOrigin(), rtol=0.0, atol=1e-8)),
        "direction_match": bool(np.allclose(image.GetDirection(), seg.GetDirection(), rtol=0.0, atol=1e-8)),
    }


def binary_array(image: sitk.Image, path: Path):
    arr = np.asarray(sitk.GetArrayFromImage(image))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D segmentation array, got {arr.shape}: {path}")
    return arr > 0


def compare_masks(human, ai):
    if human.shape != ai.shape:
        return {
            "ai_comparison_available": 0,
            "human_vs_ai_dice": "",
            "human_vs_ai_changed_voxels": "",
            "human_vs_ai_signed_rve_pct": "",
        }
    h = np.asarray(human, dtype=bool)
    a = np.asarray(ai, dtype=bool)
    inter = int(np.logical_and(h, a).sum())
    hsum = int(h.sum())
    asum = int(a.sum())
    eps = 1e-8
    return {
        "ai_comparison_available": 1,
        "human_vs_ai_dice": float((2.0 * inter + eps) / (hsum + asum + eps)),
        "human_vs_ai_changed_voxels": int(np.logical_xor(h, a).sum()),
        "human_vs_ai_signed_rve_pct": float(100.0 * (hsum - asum) / (asum + eps)),
    }


def unique_one(paths, description, case_id):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} for {case_id}, found {len(paths)}: "
            + ", ".join(str(x) for x in paths[:10])
        )
    return paths[0]


def main():
    p = argparse.ArgumentParser(description="Audit human-corrected Round-3 annotation pack")
    p.add_argument("--batch-csv", default=str(DEFAULT_BATCH))
    p.add_argument("--annotation-pack", default=str(DEFAULT_PACK))
    p.add_argument("--previous-audit", default=str(DEFAULT_PREVIOUS_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--expected-count", type=int, default=10)
    p.add_argument(
        "--promote-to",
        default=None,
        help="Optional central HUMAN_GOLD label directory. Audit must pass before any copy occurs.",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow promotion to replace an existing <case_id>.seg.nrrd. Use only intentionally.",
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
    required = {"case_id", "selection_rank", "image_path", "segmentation_path"}
    missing = required - set(batch[0])
    if missing:
        raise RuntimeError(f"Round-3 batch CSV missing columns: {sorted(missing)}")
    if len(batch) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} selected Round-3 cases, batch has {len(batch)}")

    case_ids = [str(r["case_id"]).strip() for r in batch]
    if any(not x for x in case_ids) or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Round-3 batch contains empty or duplicate case IDs")

    previous = read_json(previous_audit_path)
    if not previous.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Previous Final62/Round-2 HUMAN_GOLD audit is not marked passing")
    prior_gold = set(str(x) for x in previous.get("all_current_human_label_ids", []))
    overlap = sorted(set(case_ids) & prior_gold)
    if overlap:
        raise RuntimeError(
            "Round-3 selected cases overlap prior Final62 HUMAN_GOLD: " + ", ".join(overlap)
        )

    # Protect against silently auditing a different pack.
    expected_dirs = {f"{int(r['selection_rank']):02d}_{str(r['case_id'])}" for r in batch}
    actual_dirs = {p.name for p in pack.iterdir() if p.is_dir()}
    missing_dirs = sorted(expected_dirs - actual_dirs)
    unexpected_dirs = sorted(actual_dirs - expected_dirs)
    if missing_dirs or unexpected_dirs:
        raise RuntimeError(
            "Annotation-pack directory set does not match the frozen Round-3 batch.\n"
            f"Missing: {missing_dirs}\nUnexpected: {unexpected_dirs}"
        )

    rows = []
    failures = []
    by_case = {}
    print("=" * 116)
    print("ROUND-3 HUMAN ANNOTATION AUDIT — NON-DESTRUCTIVE")
    print(f"Batch:            {batch_path}")
    print(f"Annotation pack:  {pack}")
    print(f"Selected cases:   {len(batch)}")
    print(f"Prior HUMAN_GOLD: {len(prior_gold)}")
    print("=" * 116)

    for r in sorted(batch, key=lambda x: int(x["selection_rank"])):
        rank = int(r["selection_rank"])
        case_id = str(r["case_id"])
        case_dir = pack / f"{rank:02d}_{case_id}"
        image_path = unique_one(case_dir.glob("*.mha"), "top-level .mha image", case_id)
        seg_path = unique_one(case_dir.rglob(f"*{LABEL_SUFFIX}"), "human .seg.nrrd", case_id)

        try:
            image = sitk.ReadImage(str(image_path))
            seg = sitk.ReadImage(str(seg_path))
            if image.GetDimension() != 3 or seg.GetDimension() != 3:
                raise RuntimeError(
                    f"Expected 3D image/seg, got image={image.GetDimension()}D seg={seg.GetDimension()}D"
                )
            geom = geometry_flags(image, seg)
            geometry_match = all(geom.values())
            human = binary_array(seg, seg_path)
            fg = int(human.sum())
            if fg <= 0:
                raise RuntimeError("Human segmentation contains zero foreground voxels")
            if not geometry_match:
                bad = [k for k, v in geom.items() if not v]
                raise RuntimeError("Image/segmentation geometry mismatch: " + ", ".join(bad))

            ai_path = Path(str(r.get("segmentation_path", "")))
            ai_cmp = {
                "ai_comparison_available": 0,
                "human_vs_ai_dice": "",
                "human_vs_ai_changed_voxels": "",
                "human_vs_ai_signed_rve_pct": "",
            }
            if ai_path.exists():
                ai_img = sitk.ReadImage(str(ai_path))
                ai = binary_array(ai_img, ai_path)
                ai_cmp = compare_masks(human, ai)

            row = {
                "selection_rank": rank,
                "case_id": case_id,
                "status": "HUMAN_CORRECTED_AUDIT_PASS",
                "image_path": str(image_path),
                "human_segmentation_path": str(seg_path),
                "source_ai_segmentation_path": str(ai_path),
                "foreground_voxels": fg,
                "size_match": int(geom["size_match"]),
                "spacing_match": int(geom["spacing_match"]),
                "origin_match": int(geom["origin_match"]),
                "direction_match": int(geom["direction_match"]),
                "geometry_match": int(geometry_match),
                "audit_ok": 1,
                "audit_error": "",
                **ai_cmp,
            }
            by_case[case_id] = row
            change_text = (
                f"AI->human Dice={float(ai_cmp['human_vs_ai_dice']):.4f}, "
                f"changed={int(ai_cmp['human_vs_ai_changed_voxels'])} vox"
                if int(ai_cmp["ai_comparison_available"])
                else "AI comparison unavailable"
            )
            print(f"[{rank:02d}] {case_id} | FG={fg} | geometry PASS | {change_text}")
        except Exception as exc:
            row = {
                "selection_rank": rank,
                "case_id": case_id,
                "status": "HUMAN_CORRECTED_AUDIT_FAIL",
                "image_path": str(image_path),
                "human_segmentation_path": str(seg_path),
                "source_ai_segmentation_path": str(r.get("segmentation_path", "")),
                "foreground_voxels": "",
                "geometry_match": 0,
                "audit_ok": 0,
                "audit_error": str(exc),
            }
            failures.append(f"{case_id}: {exc}")
            print(f"[{rank:02d}] {case_id} | FAIL | {exc}")
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round3_human_annotation_audit.csv"
    json_path = output_dir / "round3_human_annotation_audit.json"
    write_csv(csv_path, rows)

    metadata = {
        "version": "round3_human_annotation_pack_audit_v1",
        "batch_csv": str(batch_path),
        "annotation_pack": str(pack),
        "previous_audit": str(previous_audit_path),
        "expected_round3_labels": int(args.expected_count),
        "selected_ids": case_ids,
        "prior_human_gold_count": len(prior_gold),
        "prior_human_gold_overlap": overlap,
        "n_audit_pass": int(sum(int(r.get("audit_ok", 0)) for r in rows)),
        "n_audit_fail": int(len(failures)),
        "all_round3_annotations_passed": not failures and len(rows) == args.expected_count,
        "promotion_requested": args.promote_to is not None,
        "promotion_destination": args.promote_to,
        "training_status_if_pass": "HUMAN_GOLD_READY",
        "note": (
            "Human-vs-AI change metrics are descriptive only. An unchanged segmentation may still be valid "
            "if it was human-verified."
        ),
    }

    if failures:
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Round-3 human annotation audit FAILED. Nothing was promoted.\n" + "\n".join(failures)
        )

    promoted = []
    if args.promote_to is not None:
        destination = Path(args.promote_to)
        destination.mkdir(parents=True, exist_ok=True)
        # Preflight the whole batch before copying any file.
        collisions = []
        for case_id in case_ids:
            target = destination / f"{case_id}{LABEL_SUFFIX}"
            if target.exists() and not args.overwrite_existing:
                collisions.append(str(target))
        if collisions:
            raise RuntimeError(
                "Promotion aborted before copying: destination labels already exist. "
                "Use --overwrite-existing only if replacement is intentional.\n" + "\n".join(collisions)
            )
        for case_id in case_ids:
            src = Path(by_case[case_id]["human_segmentation_path"])
            dst = destination / f"{case_id}{LABEL_SUFFIX}"
            shutil.copy2(src, dst)
            promoted.append({"case_id": case_id, "source": str(src), "destination": str(dst)})
        write_csv(output_dir / "round3_human_gold_promotion.csv", promoted)
        metadata["promoted_count"] = len(promoted)
        metadata["promoted_ids"] = [x["case_id"] for x in promoted]
    else:
        metadata["promoted_count"] = 0
        metadata["promoted_ids"] = []

    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 116)
    print("ROUND-3 HUMAN ANNOTATION AUDIT: PASS")
    print(f"Passing annotations: {len(rows)}/{args.expected_count}")
    print("Geometry/non-empty:  PASS for all")
    print("Prior-label overlap: 0")
    print(f"Audit CSV:            {csv_path}")
    print(f"Audit metadata:       {json_path}")
    if args.promote_to is None:
        print("Promotion:             NOT REQUESTED (safe audit-only mode)")
    else:
        print(f"Promotion:             {len(promoted)} HUMAN_GOLD labels -> {args.promote_to}")
    print("=" * 116)


if __name__ == "__main__":
    main()
