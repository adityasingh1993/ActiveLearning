#!/usr/bin/env python3
"""Audit Round-2 human labels against the frozen Round-1 training provenance.

Round-2 is intentionally derived from provenance instead of a hard-coded total label count.
The central label folder remains the human-label source of truth, but newly appearing labels
are accepted as Round-2 training labels only when they belong to the frozen Round-2 active-
learning selection manifest (unless --allow-unselected-new-labels is explicitly supplied).

This protects external validation labels from being silently promoted into training.

Definitions
-----------
frozen source labels : exact original 47 IDs from the Round-0 CV manifest
Round-1 human labels : new_human_label_ids recorded by the passing Round-1 audit
Round-2 new labels   : current valid human labels - frozen source - Round-1 human labels

Every visible label is checked for readability, non-empty foreground, and exact image/label
SimpleITK geometry using the existing audit_case implementation.
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
    DEFAULT_SOURCE_MANIFEST,
    audit_case,
    discover_round1_cases,
    read_csv,
    write_csv,
)

DEFAULT_PREVIOUS_AUDIT = Path(
    "experiments/round1_supervised_55_translation12/round1_label_audit.json"
)
DEFAULT_ROUND2_SELECTION = Path(
    "experiments/auto_label_pool_round1_raw_v1/active_learning_batch_nonaccepted.csv"
)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(
        description="Audit newly added Round-2 labels with strict selection provenance"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--previous-audit", default=str(DEFAULT_PREVIOUS_AUDIT))
    p.add_argument("--selection-manifest", default=str(DEFAULT_ROUND2_SELECTION))
    p.add_argument("--expected-round2-labels", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--allow-unselected-new-labels",
        action="store_true",
        help="Allow newly discovered labels outside the frozen Round-2 selection. Not recommended.",
    )
    args = p.parse_args()

    if args.expected_round2_labels is not None and args.expected_round2_labels < 0:
        p.error("--expected-round2-labels must be >=0")

    config = HASSLConfig.from_yaml(args.config)
    source_manifest_path = Path(args.source_manifest)
    previous_audit_path = Path(args.previous_audit)
    selection_manifest_path = Path(args.selection_manifest)

    previous = read_json(previous_audit_path)
    if not previous.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Previous Round-1 audit does not record a passing label audit")

    previous_round1_ids = sorted(str(x) for x in previous.get("new_human_label_ids", []))
    previous_total = int(previous.get("n_current_valid_human_labels", 0))
    if not previous_round1_ids:
        raise RuntimeError("Previous audit contains no Round-1 new_human_label_ids")

    _, source_ids, by_id, all_non_source_ids = discover_round1_cases(config, source_manifest_path)
    source_ids = set(str(x) for x in source_ids)
    current_ids = set(by_id)
    previous_round1_set = set(previous_round1_ids)

    missing_previous = sorted(previous_round1_set - current_ids)
    if missing_previous:
        raise RuntimeError(
            "Previously audited Round-1 human labels disappeared from the dataset: "
            + ", ".join(missing_previous)
        )

    expected_previous_total = len(source_ids) + len(previous_round1_set)
    if previous_total and previous_total != expected_previous_total:
        raise RuntimeError(
            "Previous audit is internally inconsistent: "
            f"n_current_valid_human_labels={previous_total}, source+Round1={expected_previous_total}"
        )

    round2_new_ids = sorted(set(all_non_source_ids) - previous_round1_set)

    selected_rows = read_csv(selection_manifest_path)
    if not selected_rows:
        raise RuntimeError(
            f"Round-2 selection manifest is missing or empty: {selection_manifest_path}. "
            "Do not infer Round-2 provenance from the label folder alone."
        )
    if "case_id" not in selected_rows[0]:
        raise RuntimeError(f"Selection manifest has no case_id column: {selection_manifest_path}")
    selected_ids = [str(row["case_id"]).strip() for row in selected_rows]
    if any(not x for x in selected_ids):
        raise RuntimeError("Selection manifest contains an empty case_id")
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError(f"Duplicate case IDs in selection manifest: {selection_manifest_path}")
    selected_set = set(selected_ids)

    unselected_new = sorted(set(round2_new_ids) - selected_set)
    if unselected_new and not args.allow_unselected_new_labels:
        raise RuntimeError(
            "New human labels were discovered outside the frozen Round-2 acquisition batch. "
            "Refusing to contaminate the controlled Round-2 experiment.\n"
            f"Unselected new IDs ({len(unselected_new)}): {unselected_new}\n"
            "If these are external validation labels, keep them outside the central training label folder."
        )

    if args.expected_round2_labels is not None and len(round2_new_ids) != args.expected_round2_labels:
        raise RuntimeError(
            f"Expected {args.expected_round2_labels} Round-2 new labels, found {len(round2_new_ids)}: "
            f"{round2_new_ids}"
        )

    selected_new_labeled = sorted(selected_set & set(round2_new_ids))
    selected_unlabeled = sorted(selected_set - current_ids)
    selected_previously_labeled = sorted(selected_set & (source_ids | previous_round1_set))

    audit_rows = []
    failures = []
    for case_id in sorted(by_id):
        row = audit_case(by_id[case_id])
        if case_id in source_ids:
            status = "FROZEN_SOURCE_LABEL"
        elif case_id in previous_round1_set:
            status = "ROUND1_HUMAN_LABEL"
        elif case_id in set(round2_new_ids):
            status = "ROUND2_NEW_HUMAN_LABEL"
        else:
            status = "UNEXPECTED_HUMAN_LABEL"
        row = {"status": status, **row, "reason": ""}
        audit_rows.append(row)
        if not int(row["audit_ok"]):
            failures.append(f"{case_id}: {row['audit_error']}")

    for case_id in selected_unlabeled:
        audit_rows.append({
            "status": "ROUND2_SELECTED_UNLABELED",
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

    if failures:
        raise RuntimeError(
            "Round-2 label audit failed. Do not train until every visible human label passes.\n"
            + "\n".join(failures[:10])
        )

    total = len(by_id)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"experiments/round2_supervised_{total}_translation12")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round2_label_audit.csv"
    json_path = output_dir / "round2_label_audit.json"
    write_csv(csv_path, audit_rows)

    metadata = {
        "version": "round2_label_audit_from_current_folder_v1",
        "source_manifest": str(source_manifest_path),
        "previous_audit": str(previous_audit_path),
        "selection_manifest": str(selection_manifest_path),
        "n_frozen_source_labels": len(source_ids),
        "n_round1_human_labels": len(previous_round1_set),
        "n_round2_new_human_labels": len(round2_new_ids),
        "n_current_valid_human_labels": len(by_id),
        "round1_human_label_ids": previous_round1_ids,
        "round2_new_human_label_ids": round2_new_ids,
        "all_non_source_human_label_ids": sorted(set(all_non_source_ids)),
        "all_current_human_label_ids": sorted(current_ids),
        "selected_ids": selected_ids,
        "selected_new_labeled_ids": selected_new_labeled,
        "selected_unlabeled_ids": selected_unlabeled,
        "selected_previously_labeled_ids": selected_previously_labeled,
        "unselected_new_label_ids": unselected_new,
        "expected_round2_labels": args.expected_round2_labels,
        "all_visible_labels_passed_audit": True,
        "selection_provenance_enforced": not args.allow_unselected_new_labels,
        "training_rule": (
            "Controlled Round-2 training uses frozen source + Round-1 human + audited Round-2 new human labels. "
            "External validation labels and pseudo-labels are excluded."
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 108)
    print("ROUND-2 HUMAN LABEL AUDIT — PROVENANCE SAFE")
    print(f"Frozen source labels:       {len(source_ids)}")
    print(f"Round-1 human labels:       {len(previous_round1_set)}")
    print(f"Round-2 new human labels:   {len(round2_new_ids)}")
    print(f"Total usable human labels:  {len(by_id)}")
    print(f"Frozen Round-2 selection:   {len(selected_ids)}")
    print(f"Selected + newly labeled:   {len(selected_new_labeled)}")
    print(f"Selected + unlabeled:       {len(selected_unlabeled)}")
    if round2_new_ids:
        print("Round-2 new IDs:")
        for case_id in round2_new_ids:
            print(f"  {case_id}")
    if selected_unlabeled:
        print("Still selected/unlabeled:")
        for case_id in selected_unlabeled:
            print(f"  {case_id}")
    print("Geometry/non-empty audit:   PASS")
    print("Unselected-label guard:     " + ("BYPASSED" if args.allow_unselected_new_labels else "ENFORCED"))
    print(f"Audit CSV:                  {csv_path}")
    print(f"Audit metadata:             {json_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
