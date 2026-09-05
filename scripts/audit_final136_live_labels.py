#!/usr/bin/env python3
"""Read-only geometry/non-empty audit for the explicit all-136 training experiment.

This audit intentionally permits the historically quarantined 9435... case because the all-136
experiment requested by the user includes every currently labeled case. It never edits labels and
never accesses External31.
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


SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
OUTPUT_DIR = Path("experiments/final136_two_stage_all136/audit")
EXPECTED_TOTAL = 136
PRIOR_QUARANTINE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)


def main():
    parser = argparse.ArgumentParser(description="Audit all 136 live labels for Final136 training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TOTAL)
    args = parser.parse_args()

    if args.expected_count != EXPECTED_TOTAL:
        parser.error("Final136 audit is locked to --expected-count 136")

    config = HASSLConfig.from_yaml(args.config)
    _, source_ids, by_id, new_ids = discover_round1_cases(config, args.source_manifest)
    current_ids = sorted(str(case_id) for case_id in by_id)
    if len(current_ids) != EXPECTED_TOTAL or len(set(current_ids)) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TOTAL} unique live labels, found {len(set(current_ids))}"
        )
    if PRIOR_QUARANTINE_ID not in current_ids:
        raise RuntimeError("Historically quarantined 9435... case is absent from the all-136 dataset")

    rows = []
    failures = []
    for case_id in current_ids:
        row = audit_case(by_id[case_id])
        row = {
            "training_status": (
                "INTENTIONALLY_INCLUDED_PRIOR_QUARANTINE"
                if case_id == PRIOR_QUARANTINE_ID
                else "TRAIN_ALL136"
            ),
            **row,
        }
        rows.append(row)
        if not int(row.get("audit_ok", 0)):
            failures.append(f"{case_id}: {row.get('audit_error', 'audit failed')}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final136_live_label_audit.csv"
    json_path = output_dir / "final136_live_label_audit.json"
    write_csv(csv_path, rows)
    metadata = {
        "version": "final136_live_label_audit_all136_v1",
        "source_manifest": str(args.source_manifest),
        "n_frozen_source": len(source_ids),
        "n_new_since_original47": len(new_ids),
        "n_total_human_gold": len(current_ids),
        "all_current_human_label_ids": current_ids,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "selection_provenance_enforced": False,
        "training_scope_provenance_enforced": True,
        "training_scope": "all_136_live_labels",
        "quarantined_case_ids": [],
        "intentionally_included_prior_quarantine_ids": [PRIOR_QUARANTINE_ID],
        "label_source": "live central label directory; read-only audit",
        "external31_access": False,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError("FINAL136 AUDIT FAILED. Do not train.\n" + "\n".join(failures[:30]))

    print("=" * 112)
    print("FINAL136 LIVE HUMAN_GOLD AUDIT — PASS")
    print(f"Frozen original47:       {len(source_ids)}")
    print(f"New since original47:    {len(new_ids)}")
    print(f"Total training labels:   {len(current_ids)}")
    print("Prior 9435 quarantine:   INTENTIONALLY INCLUDED")
    print("Geometry/non-empty:      PASS for all")
    print("External31:              NOT ACCESSED")
    print(f"Audit metadata:          {json_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
