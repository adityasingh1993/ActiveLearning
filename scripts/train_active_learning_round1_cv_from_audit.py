#!/usr/bin/env python3
"""Run controlled Round-1 CV with label counts derived from the audit metadata.

This thin profile avoids hard-coding the number of newly annotated active-learning cases.
The underlying controlled CV implementation remains scripts/train_active_learning_round1_cv.py.

Example for the current Round-1 set (47 frozen + 8 new = 55 usable labels):

  python scripts/train_active_learning_round1_cv_from_audit.py \
    --config config_resize128.yaml \
    --audit-metadata experiments/round1_supervised_55_translation12/round1_label_audit.json \
    --output-dir experiments/round1_cv_55_translation12 \
    --fold all

The audit metadata is the source of truth for the new-label count. The original 47 held-out
folds are still reused exactly by the underlying trainer.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_active_learning_round1_cv as base


def _peek_audit_path(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit-metadata", required=True)
    known, _ = parser.parse_known_args(argv)
    return Path(known.audit_metadata)


def main():
    audit_path = _peek_audit_path(sys.argv[1:])
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Audit metadata does not record a passing label audit")

    new_ids = sorted(str(x) for x in audit.get("new_human_label_ids", []))
    expected = int(audit.get("expected_new_human_labels", len(new_ids)))
    if expected != len(new_ids):
        raise RuntimeError(
            "Audit metadata is internally inconsistent: "
            f"expected_new_human_labels={expected}, discovered={len(new_ids)}"
        )
    if expected < 1:
        raise RuntimeError("Round-1 audit contains no new human labels")

    total = base.EXPECTED_SOURCE_CASES + expected
    reported_total = int(audit.get("n_current_valid_human_labels", total))
    if reported_total != total:
        raise RuntimeError(
            "Audit total does not equal frozen source + new labels: "
            f"reported={reported_total}, expected={total}"
        )

    # Configure the existing controlled-CV implementation from audited provenance.
    base.EXPECTED_NEW_LABELS = expected
    base.EXPECTED_TOTAL_LABELS = total

    print("=" * 104)
    print("ROUND-1 CV PROFILE FROM AUDIT")
    print(f"Audit:              {audit_path}")
    print(f"Frozen labels:      {base.EXPECTED_SOURCE_CASES}")
    print(f"New human labels:   {expected}")
    print(f"Total usable labels:{total}")
    print("The original 47 held-out fold assignments remain unchanged.")
    print("=" * 104)

    base.main()


if __name__ == "__main__":
    main()
