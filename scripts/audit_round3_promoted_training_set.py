#!/usr/bin/env python3
"""Audit the promoted Round-3 HUMAN_GOLD training set before Final72 training.

Expected controlled state:
- exact prior Final62 HUMAN_GOLD from the passing Round-2 audit,
- exact 10 Round-3 selected/annotated cases from the passing annotation-pack audit,
- no other new labels,
- every visible image/label pair readable, non-empty, and exact-geometry matched.

This script is read-only with respect to images/labels. It writes audit reports only.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import audit_case, discover_round1_cases, write_csv

DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_PREVIOUS_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_ROUND3_ANNOTATION_AUDIT = Path(
    "experiments/round3_failure_aware_v1/human_annotation_audit/round3_human_annotation_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/round3_supervised_72_translation12")
EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 62
EXPECTED_ROUND3 = 10
EXPECTED_TOTAL = 72


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Audit promoted Round-3 HUMAN_GOLD for controlled Final72 training")
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    p.add_argument("--previous-audit", default=str(DEFAULT_PREVIOUS_AUDIT))
    p.add_argument("--round3-annotation-audit", default=str(DEFAULT_ROUND3_ANNOTATION_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = p.parse_args()

    source_manifest = Path(args.source_manifest)
    previous_path = Path(args.previous_audit)
    annotation_path = Path(args.round3_annotation_audit)
    output_dir = Path(args.output_dir)

    previous = read_json(previous_path)
    if not previous.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Prior Final62 audit is not marked passing")
    prior_ids = sorted(str(x) for x in previous.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR:
        raise RuntimeError(f"Expected {EXPECTED_PRIOR} prior HUMAN_GOLD IDs, found {len(prior_ids)}")

    ann = read_json(annotation_path)
    if not ann.get("all_round3_annotations_passed", False):
        raise RuntimeError("Round-3 annotation audit is not marked passing")
    round3_ids = sorted(str(x) for x in ann.get("selected_ids", []))
    if len(round3_ids) != EXPECTED_ROUND3 or len(set(round3_ids)) != EXPECTED_ROUND3:
        raise RuntimeError(f"Expected exactly {EXPECTED_ROUND3} unique Round-3 selected IDs")
    overlap = sorted(set(prior_ids) & set(round3_ids))
    if overlap:
        raise RuntimeError("Round-3 IDs overlap prior Final62 HUMAN_GOLD: " + ", ".join(overlap))

    config = HASSLConfig.from_yaml(args.config)
    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest)
    source_ids = sorted(str(x) for x in source_ids)
    if len(source_ids) != EXPECTED_SOURCE:
        raise RuntimeError(f"Expected frozen {EXPECTED_SOURCE} source IDs, found {len(source_ids)}")

    current_ids = sorted(str(x) for x in by_id)
    expected_ids = sorted(set(prior_ids) | set(round3_ids))
    unexpected = sorted(set(current_ids) - set(expected_ids))
    missing = sorted(set(expected_ids) - set(current_ids))
    if unexpected or missing:
        raise RuntimeError(
            "Current central HUMAN_GOLD set does not equal Final62 + selected Round3.\n"
            f"Unexpected labels ({len(unexpected)}): {unexpected}\n"
            f"Missing expected labels ({len(missing)}): {missing}"
        )
    if len(current_ids) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} current labels, found {len(current_ids)}")

    rows = []
    failures = []
    prior_set = set(prior_ids)
    round3_set = set(round3_ids)
    for case_id in current_ids:
        row = audit_case(by_id[case_id])
        if case_id in round3_set:
            status = "ROUND3_NEW_HUMAN_GOLD"
        elif case_id in prior_set:
            status = "PRIOR_FINAL62_HUMAN_GOLD"
        else:
            status = "UNEXPECTED"
        row = {"status": status, **row}
        rows.append(row)
        if not int(row["audit_ok"]):
            failures.append(f"{case_id}: {row['audit_error']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round3_label_audit.csv"
    json_path = output_dir / "round3_label_audit.json"
    write_csv(csv_path, rows)

    metadata = {
        "version": "round3_promoted_training_set_audit_v1",
        "source_manifest": str(source_manifest),
        "previous_round2_audit": str(previous_path),
        "round3_annotation_audit": str(annotation_path),
        "n_frozen_source_labels": len(source_ids),
        "n_prior_final62_human_labels": len(prior_ids),
        "n_round3_new_human_labels": len(round3_ids),
        "n_current_valid_human_labels": len(current_ids),
        "prior_final62_human_label_ids": prior_ids,
        "round3_new_human_label_ids": round3_ids,
        "all_current_human_label_ids": current_ids,
        "unexpected_new_label_ids": unexpected,
        "missing_expected_label_ids": missing,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "selection_provenance_enforced": True,
        "training_rule": (
            "Controlled Round-3/Final72 training uses exact prior Final62 HUMAN_GOLD plus exact 10 audited Round-3 human annotations. "
            "Pseudo-labels and external31 labels are excluded."
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError("Round-3 full training-set audit FAILED. Do not train.\n" + "\n".join(failures[:20]))

    print("=" * 110)
    print("ROUND-3 / FINAL72 HUMAN_GOLD AUDIT — PASS")
    print(f"Frozen original source:   {len(source_ids)}")
    print(f"Prior Final62 labels:     {len(prior_ids)}")
    print(f"Round-3 new labels:       {len(round3_ids)}")
    print(f"Total HUMAN_GOLD:         {len(current_ids)}")
    print("Geometry/non-empty:       PASS for all")
    print("Unexpected labels:        0")
    print("Missing expected labels:  0")
    print(f"Audit CSV:                {csv_path}")
    print(f"Audit metadata:           {json_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
