#!/usr/bin/env python3
"""Materialize GT-derived ROI crops for the audited 62 human-label experiment set.

This is an oracle feasibility dataset. Ground truth defines each crop, so this dataset is for
controlled research only and is never a deployable inference input pipeline.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("build_oracle_roi_dataset.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import discover_round1_cases
from scripts.oracle_roi_utils import make_oracle_roi

DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_OUTPUT_DIR = Path("experiments/oracle_roi_dataset_62_v1")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Build GT-derived oracle ROI dataset from audited human labels")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--margin-fraction", type=float, default=0.40)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.margin_fraction < 0:
        p.error("--margin-fraction must be >=0")

    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    expected_n = int(audit.get("n_current_valid_human_labels", len(audited_ids)))
    if not audited_ids or len(audited_ids) != expected_n:
        raise RuntimeError("Round-2 audit ID/count mismatch")

    config = HASSLConfig.from_yaml(args.config)
    _, _, by_id, _ = discover_round1_cases(config, Path(args.source_manifest))
    current_ids = sorted(str(x) for x in by_id)
    if current_ids != audited_ids:
        raise RuntimeError(
            "Current labeled dataset no longer matches the passing Round-2 audit. "
            "Do not build the oracle dataset until provenance is reconciled."
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, case_id in enumerate(audited_ids, start=1):
        case = by_id[case_id]
        crop_image, crop_label, meta = make_oracle_roi(
            case["image"], case["label"], margin_fraction=float(args.margin_fraction)
        )
        image_out = output_dir / f"{case_id}{config.image_suffix}"
        label_out = labels_dir / f"{case_id}{config.label_suffix}"
        sitk.WriteImage(crop_image, str(image_out), useCompression=True)
        sitk.WriteImage(crop_label, str(label_out), useCompression=True)
        rows.append({
            "case_id": case_id,
            "source_image": str(case["image"]),
            "source_label": str(case["label"]),
            "roi_image": str(image_out),
            "roi_label": str(label_out),
            **meta,
        })
        print(
            f"[{index:2d}/{len(audited_ids)}] {case_id} | "
            f"ROI={meta['roi_size_x']}x{meta['roi_size_y']}x{meta['roi_size_z']} | "
            f"GT fraction {meta['full_gt_fraction']:.5f}->{meta['roi_gt_fraction']:.5f}"
        )

    csv_path = output_dir / "oracle_roi_manifest.csv"
    write_csv(csv_path, rows)
    payload = {
        "version": "oracle_roi_dataset_62_v1",
        "purpose": "GT-derived ROI upper-bound feasibility experiment only",
        "source_round2_audit": str(audit_path),
        "n_cases": len(rows),
        "case_ids": audited_ids,
        "margin_fraction_each_side": float(args.margin_fraction),
        "image_suffix": config.image_suffix,
        "label_suffix": config.label_suffix,
        "warning": (
            "Ground truth defines every crop. This dataset must not be interpreted as deployable localization "
            "or used to claim production inference performance."
        ),
    }
    json_path = output_dir / "oracle_roi_dataset_metadata.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print("ORACLE ROI DATASET COMPLETE")
    print(f"Cases:             {len(rows)}")
    print(f"Margin each side:  {args.margin_fraction:.2f} x target bbox extent")
    print(f"Dataset:           {output_dir}")
    print(f"Manifest:          {csv_path}")
    print("Use only for oracle ROI feasibility experiments.")
    print("=" * 104)


if __name__ == "__main__":
    main()
