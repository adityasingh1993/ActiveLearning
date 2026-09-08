#!/usr/bin/env python3
"""Read-only geometry/non-empty audit for the explicit all-136 training experiment.

The current dataset must contain exactly 136 valid image-label pairs. Four historical cases from
the frozen original-47 manifest may be absent by explicit user decision; no other missing frozen
case is accepted. The audit never edits labels and never accesses External31.
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
from scripts.stage2_cohort_contract import (
    FINAL136_ALLOWED_MISSING_FROZEN_IDS as ALLOWED_MISSING_FROZEN_IDS,
    PRIOR_QUARANTINE_ID,
)


SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
OUTPUT_DIR = Path("experiments/final136_two_stage_all136/audit")
EXPECTED_TOTAL = 136


def main():
    parser = argparse.ArgumentParser(description="Audit all 136 live labels for Final136 training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TOTAL)
    parser.add_argument(
        "--allow-missing-frozen",
        action="store_true",
        help=(
            "Allow exactly the four explicitly approved original47 IDs to be absent from the "
            "current paired dataset."
        ),
    )
    args = parser.parse_args()

    if args.expected_count != EXPECTED_TOTAL:
        parser.error("Final136 audit is locked to --expected-count 136")
    if not args.allow_missing_frozen:
        parser.error(
            "Final136 requires --allow-missing-frozen for the four explicitly approved "
            "historical IDs"
        )

    config = HASSLConfig.from_yaml(args.config)
    _, source_ids, by_id, new_ids = discover_round1_cases(
        config,
        args.source_manifest,
        require_all_frozen=False,
    )
    current_ids = sorted(str(case_id) for case_id in by_id)
    missing_frozen_ids = sorted(set(source_ids) - set(current_ids))
    if set(missing_frozen_ids) != ALLOWED_MISSING_FROZEN_IDS:
        unexpected_missing = sorted(set(missing_frozen_ids) - ALLOWED_MISSING_FROZEN_IDS)
        approved_still_present = sorted(ALLOWED_MISSING_FROZEN_IDS - set(missing_frozen_ids))
        details = []
        if unexpected_missing:
            details.append("unexpected missing: " + ", ".join(unexpected_missing))
        if approved_still_present:
            details.append("approved IDs still present: " + ", ".join(approved_still_present))
        raise RuntimeError(
            "Frozen-manifest mismatch differs from the exact approved four-case allowlist"
            + (" (" + "; ".join(details) + ")" if details else "")
        )
    if len(current_ids) != EXPECTED_TOTAL or len(set(current_ids)) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TOTAL} unique current image-label pairs, "
            f"found {len(set(current_ids))}"
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
        "version": "final136_live_label_audit_all136_missing4_v1",
        "source_manifest": str(args.source_manifest),
        "n_frozen_source": len(source_ids),
        "n_frozen_source_present": len(set(source_ids) & set(current_ids)),
        "n_frozen_source_missing": len(missing_frozen_ids),
        "missing_frozen_case_ids": missing_frozen_ids,
        "allowed_missing_frozen_case_ids": sorted(ALLOWED_MISSING_FROZEN_IDS),
        "missing_frozen_allowed": True,
        "n_new_since_original47": len(new_ids),
        "n_total_human_gold": len(current_ids),
        "all_current_human_label_ids": current_ids,
        "n_training_cases": len(current_ids),
        "training_case_ids": current_ids,
        "excluded_training_case_ids": [],
        "n_excluded_training_cases": 0,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "all_training_labels_passed_audit": len(failures) == 0,
        "excluded_case_audit_failures": [],
        "selection_provenance_enforced": False,
        "training_scope_provenance_enforced": True,
        "training_scope": "all_136_current_live_labels_with_exact_four_missing_frozen_allowed",
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
    print(f"Frozen present/missing:  {len(set(source_ids) & set(current_ids))} / "
          f"{len(missing_frozen_ids)}")
    print("Approved missing IDs:    " + ", ".join(missing_frozen_ids))
    print(f"New since original47:    {len(new_ids)}")
    print(f"Total training labels:   {len(current_ids)}")
    print("Prior 9435 quarantine:   INTENTIONALLY INCLUDED")
    print("Geometry/non-empty:      PASS for all 136 current pairs")
    print("External31:              NOT ACCESSED")
    print(f"Audit metadata:          {json_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
