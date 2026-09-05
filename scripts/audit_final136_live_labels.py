#!/usr/bin/env python3
"""Read-only geometry/non-empty audit for the explicit 136-case dataset.

The audit checks all 136 visible labels, records four user-selected exclusions, and produces the
exact 132-case training scope. It never edits labels and never accesses External31.
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
OUTPUT_DIR = Path("experiments/final136_two_stage_train132/audit")
EXPECTED_TOTAL = 136
PRIOR_QUARANTINE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)
EXCLUDED_CASE_IDS = {
    "81a0f3f3fa1e2ad8bd01d3915898298ffff947a83b5749846b228a612356fb2f",
    "96165b4ca29e10f85866ca53ac68e4e44a127fe676e5745e0a94d50602819f0f",
    "a31909b0e87f789c68489e8ebe6a5adfb72c5a19dbe5546cd762ea2f82c7037a",
    "d8269b9a976314fb41fa63187c4b2dc05ee94e2973125df67bb5e4fd75bc7563",
}
EXPECTED_TRAINING = EXPECTED_TOTAL - len(EXCLUDED_CASE_IDS)


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
            "Allow original47 cases absent from the current paired dataset. Missing IDs are "
            "recorded; exactly 136 current valid pairs are still required."
        ),
    )
    args = parser.parse_args()

    if args.expected_count != EXPECTED_TOTAL:
        parser.error("Final136 audit is locked to --expected-count 136")

    config = HASSLConfig.from_yaml(args.config)
    _, source_ids, by_id, new_ids = discover_round1_cases(
        config,
        args.source_manifest,
        require_all_frozen=not args.allow_missing_frozen,
    )
    current_ids = sorted(str(case_id) for case_id in by_id)
    missing_frozen_ids = sorted(set(source_ids) - set(current_ids))
    if len(current_ids) != EXPECTED_TOTAL or len(set(current_ids)) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TOTAL} unique live labels, found {len(set(current_ids))}"
        )
    if PRIOR_QUARANTINE_ID not in current_ids:
        raise RuntimeError("Historically quarantined 9435... case is absent from the 136-case dataset")
    missing_exclusions = sorted(EXCLUDED_CASE_IDS - set(current_ids))
    if missing_exclusions:
        raise RuntimeError(
            "Requested training exclusions are absent from the 136-case dataset: "
            + ", ".join(missing_exclusions)
        )
    training_ids = sorted(set(current_ids) - EXCLUDED_CASE_IDS)
    if len(training_ids) != EXPECTED_TRAINING:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TRAINING} training cases, found {len(training_ids)}"
        )

    rows = []
    training_failures = []
    excluded_failures = []
    for case_id in current_ids:
        row = audit_case(by_id[case_id])
        if case_id in EXCLUDED_CASE_IDS:
            training_status = "EXCLUDED_BY_USER"
        elif case_id == PRIOR_QUARANTINE_ID:
            training_status = "INTENTIONALLY_INCLUDED_PRIOR_QUARANTINE"
        else:
            training_status = "TRAIN_FINAL132"
        row = {
            "training_status": training_status,
            **row,
        }
        rows.append(row)
        if not int(row.get("audit_ok", 0)):
            failure = f"{case_id}: {row.get('audit_error', 'audit failed')}"
            if case_id in EXCLUDED_CASE_IDS:
                excluded_failures.append(failure)
            else:
                training_failures.append(failure)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final136_live_label_audit.csv"
    json_path = output_dir / "final136_live_label_audit.json"
    write_csv(csv_path, rows)
    metadata = {
        "version": "final136_live_label_audit_train132_v1",
        "source_manifest": str(args.source_manifest),
        "n_frozen_source": len(source_ids),
        "n_frozen_source_present": len(set(source_ids) & set(current_ids)),
        "n_frozen_source_missing": len(missing_frozen_ids),
        "missing_frozen_case_ids": missing_frozen_ids,
        "missing_frozen_allowed": bool(args.allow_missing_frozen),
        "n_new_since_original47": len(new_ids),
        "n_total_human_gold": len(current_ids),
        "all_current_human_label_ids": current_ids,
        "n_training_cases": len(training_ids),
        "training_case_ids": training_ids,
        "excluded_training_case_ids": sorted(EXCLUDED_CASE_IDS),
        "n_excluded_training_cases": len(EXCLUDED_CASE_IDS),
        "all_visible_labels_passed_audit": not training_failures and not excluded_failures,
        "all_training_labels_passed_audit": len(training_failures) == 0,
        "excluded_case_audit_failures": excluded_failures,
        "selection_provenance_enforced": False,
        "training_scope_provenance_enforced": True,
        "training_scope": "132_of_136_live_labels_after_four_explicit_exclusions",
        "quarantined_case_ids": sorted(EXCLUDED_CASE_IDS),
        "intentionally_included_prior_quarantine_ids": [PRIOR_QUARANTINE_ID],
        "label_source": "live central label directory; read-only audit",
        "external31_access": False,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if training_failures:
        raise RuntimeError(
            "FINAL136 TRAINING-SCOPE AUDIT FAILED. Do not train.\n"
            + "\n".join(training_failures[:30])
        )

    print("=" * 112)
    print("FINAL136 LIVE HUMAN_GOLD AUDIT — PASS")
    print(f"Frozen original47:       {len(source_ids)}")
    print(f"Frozen present/missing:  {len(set(source_ids) & set(current_ids))} / "
          f"{len(missing_frozen_ids)}")
    if missing_frozen_ids:
        print("Missing frozen IDs:      " + ", ".join(missing_frozen_ids))
    print(f"New since original47:    {len(new_ids)}")
    print(f"Total audited labels:    {len(current_ids)}")
    print(f"Total training labels:   {len(training_ids)}")
    print("Excluded by user:        " + ", ".join(sorted(EXCLUDED_CASE_IDS)))
    print(f"Excluded audit failures: {len(excluded_failures)} (recorded; non-blocking)")
    print("Prior 9435 quarantine:   INTENTIONALLY INCLUDED")
    print("Geometry/non-empty:      PASS for all")
    print("External31:              NOT ACCESSED")
    print(f"Audit metadata:          {json_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
