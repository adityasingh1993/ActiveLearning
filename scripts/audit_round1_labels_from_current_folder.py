#!/usr/bin/env python3
"""Audit the current Round-1 labels without assuming why selected cases are unlabeled.

The central label folder is treated as the source of truth. New human labels are discovered as
current valid labels minus the frozen 47 Round-0 IDs. If a prior active-learning selection
manifest is available, selected cases that still have no label are recorded as
SELECTED_UNLABELED only; no NOT_ANNOTATABLE reason is invented.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import (
    DEFAULT_SELECTION_MANIFEST,
    DEFAULT_SOURCE_MANIFEST,
    audit_case,
    discover_round1_cases,
    read_csv,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit frozen + newly added Round-1 labels from the current central label folder"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    parser.add_argument("--expected-new-labels", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = HASSLConfig.from_yaml(args.config)
    source_manifest_path = Path(args.source_manifest)
    selection_manifest_path = Path(args.selection_manifest)

    _, source_ids, by_id, new_ids = discover_round1_cases(config, source_manifest_path)
    if args.expected_new_labels is not None and len(new_ids) != int(args.expected_new_labels):
        raise RuntimeError(
            f"Expected {args.expected_new_labels} newly added human labels, found {len(new_ids)}: {new_ids}"
        )

    total = len(by_id)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"experiments/round1_supervised_{total}_translation12")
    )

    selected_rows = read_csv(selection_manifest_path)
    selected_ids = []
    if selected_rows:
        if "case_id" not in selected_rows[0]:
            raise RuntimeError(f"Selection manifest has no case_id column: {selection_manifest_path}")
        selected_ids = [str(row["case_id"]) for row in selected_rows]
        if len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError(f"Duplicate case IDs in selection manifest: {selection_manifest_path}")

    audit_rows = []
    failures = []
    for case_id in sorted(by_id):
        row = audit_case(by_id[case_id])
        status = "FROZEN_SOURCE_LABEL" if case_id in source_ids else "ROUND1_NEW_HUMAN_LABEL"
        row = {"status": status, **row, "reason": ""}
        audit_rows.append(row)
        if not int(row["audit_ok"]):
            failures.append(f"{case_id}: {row['audit_error']}")

    if failures:
        raise RuntimeError(
            "Round-1 label audit failed. Do not train until every visible human label passes.\n"
            + "\n".join(failures[:10])
        )

    selected_unlabeled = sorted(set(selected_ids) - set(by_id)) if selected_ids else []
    selected_new_labeled = sorted(set(selected_ids) & set(new_ids)) if selected_ids else []
    for case_id in selected_unlabeled:
        audit_rows.append({
            "status": "SELECTED_UNLABELED",
            "case_id": case_id,
            "image_path": "",
            "label_path": "",
            "image_exists": "",
            "label_exists": 0,
            "foreground_voxels": "",
            "label_min": "",
            "label_max": "",
            "size_match": "",
            "spacing_match": "",
            "origin_match": "",
            "direction_match": "",
            "geometry_match": "",
            "audit_ok": "",
            "audit_error": "",
            "reason": "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round1_label_audit.csv"
    json_path = output_dir / "round1_label_audit.json"
    write_csv(csv_path, audit_rows)

    metadata = {
        "version": "round1_label_audit_from_current_folder_v1",
        "source_manifest": str(source_manifest_path),
        "selection_manifest": str(selection_manifest_path) if selection_manifest_path.exists() else None,
        "n_frozen_source_labels": len(source_ids),
        "n_current_valid_human_labels": len(by_id),
        "n_new_human_labels": len(new_ids),
        "new_human_label_ids": new_ids,
        "expected_new_human_labels": (
            int(args.expected_new_labels) if args.expected_new_labels is not None else len(new_ids)
        ),
        "selected_ids": selected_ids,
        "selected_new_labeled_ids": selected_new_labeled,
        "selected_unlabeled_ids": selected_unlabeled,
        "all_visible_labels_passed_audit": True,
        "training_rule": (
            "Round-1 training may use frozen source labels plus new_human_label_ids only; "
            "selected_unlabeled cases are excluded."
        ),
        "note": (
            "SELECTED_UNLABELED does not imply NOT_ANNOTATABLE. Record target-not-visible or other "
            "annotation reasons separately when the specific case IDs are known."
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 100)
    print("ROUND-1 HUMAN LABEL AUDIT — CURRENT FOLDER")
    print(f"Frozen source labels:    {len(source_ids)}")
    print(f"New human labels:        {len(new_ids)}")
    print(f"Total usable labels:     {len(by_id)}")
    print(f"New IDs:                 {', '.join(new_ids)}")
    if selected_ids:
        print(f"Previous selected batch: {len(selected_ids)}")
        print(f"Selected + labeled:      {len(selected_new_labeled)}")
        print(f"Selected + unlabeled:    {len(selected_unlabeled)}")
        for case_id in selected_unlabeled:
            print(f"  SELECTED_UNLABELED: {case_id}")
    print("Geometry/non-empty audit: PASS")
    print(f"Audit CSV:               {csv_path}")
    print(f"Audit metadata:          {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
