#!/usr/bin/env python3
"""Read-only audit for the live Final91 HUMAN_GOLD dataset after Round-5 annotation.

Expected controlled state:
  prior Final82 HUMAN_GOLD (82) + exact Round-5 ANNOTATE IDs (9) = 91 labels.

The user has already copied corrected Round-5 masks directly into the central labels directory.
This script never copies, promotes, or edits labels. It validates the live training state before CV.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import audit_case, discover_round1_cases, write_csv

SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
FINAL82_AUDIT = Path("experiments/round4_cv_82_a3/final82_live_label_audit.json")
ROUND5_BATCH = Path("experiments/round5_active_final82_a3_committee_v1/round5_annotation_batch.csv")
OUTPUT_DIR = Path("experiments/round5_supervised_91_a3")
EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 82
EXPECTED_ROUND5 = 9
EXPECTED_TOTAL = 91


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Audit live Final91 HUMAN_GOLD labels")
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    p.add_argument("--final82-audit", default=str(FINAL82_AUDIT))
    p.add_argument("--round5-batch", default=str(ROUND5_BATCH))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    source_manifest = Path(args.source_manifest)
    final82_audit = Path(args.final82_audit)
    round5_batch = Path(args.round5_batch)
    output_dir = Path(args.output_dir)

    prior = read_json(final82_audit)
    if not prior.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final82 live-label audit is not marked passing")
    prior_ids = sorted(str(x) for x in prior.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR or len(set(prior_ids)) != EXPECTED_PRIOR:
        raise RuntimeError(f"Expected exactly {EXPECTED_PRIOR} unique Final82 IDs, found {len(set(prior_ids))}")

    batch = read_csv(round5_batch)
    if len(batch) != EXPECTED_ROUND5:
        raise RuntimeError(f"Expected exactly {EXPECTED_ROUND5} Round5 ANNOTATE rows, found {len(batch)}")
    required = {"case_id", "round5_state"}
    missing_cols = required - set(batch[0])
    if missing_cols:
        raise RuntimeError(f"Round5 batch missing columns: {sorted(missing_cols)}")
    bad_states = [str(r.get("case_id", "")) for r in batch if str(r.get("round5_state", "")).strip().upper() != "ANNOTATE"]
    if bad_states:
        raise RuntimeError("Round5 batch contains non-ANNOTATE rows: " + ", ".join(bad_states))
    round5_ids = sorted(str(r["case_id"]).strip() for r in batch)
    if any(not x for x in round5_ids) or len(set(round5_ids)) != EXPECTED_ROUND5:
        raise RuntimeError("Round5 batch contains empty or duplicate case IDs")
    overlap = sorted(set(prior_ids) & set(round5_ids))
    if overlap:
        raise RuntimeError("Round5 IDs overlap Final82 HUMAN_GOLD: " + ", ".join(overlap))

    config = HASSLConfig.from_yaml(args.config)
    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest)
    source_ids = sorted(str(x) for x in source_ids)
    if len(source_ids) != EXPECTED_SOURCE or len(set(source_ids)) != EXPECTED_SOURCE:
        raise RuntimeError("Frozen source manifest is not exact original47")

    current_ids = sorted(str(x) for x in by_id)
    expected_ids = sorted(set(prior_ids) | set(round5_ids))
    unexpected = sorted(set(current_ids) - set(expected_ids))
    missing = sorted(set(expected_ids) - set(current_ids))
    if unexpected or missing:
        raise RuntimeError(
            "Live label directory is not exact Final82 + Round5 selected9.\n"
            f"Unexpected labels ({len(unexpected)}): {unexpected}\n"
            f"Missing labels ({len(missing)}): {missing}"
        )
    if len(current_ids) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected exactly {EXPECTED_TOTAL} live labels, found {len(current_ids)}")

    rows = []
    failures = []
    prior_set, r5_set = set(prior_ids), set(round5_ids)
    for case_id in current_ids:
        row = audit_case(by_id[case_id])
        status = "ROUND5_NEW_HUMAN_GOLD" if case_id in r5_set else "PRIOR_FINAL82_HUMAN_GOLD"
        row = {"status": status, **row}
        rows.append(row)
        if not int(row.get("audit_ok", 0)):
            failures.append(f"{case_id}: {row.get('audit_error', 'audit failed')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final91_live_label_audit.csv"
    json_path = output_dir / "final91_live_label_audit.json"
    write_csv(csv_path, rows)

    metadata = {
        "version": "round5_final91_live_label_audit_v1",
        "source_manifest": str(source_manifest),
        "prior_final82_audit": str(final82_audit),
        "round5_annotation_batch": str(round5_batch),
        "n_frozen_source": len(source_ids),
        "n_prior_final82": len(prior_ids),
        "n_round5_new": len(round5_ids),
        "n_total_human_gold": len(current_ids),
        "prior_final82_human_label_ids": prior_ids,
        "round5_new_human_label_ids": round5_ids,
        "all_current_human_label_ids": current_ids,
        "unexpected_label_ids": unexpected,
        "missing_expected_label_ids": missing,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "selection_provenance_enforced": True,
        "label_source": "live central label directory; Round5 masks copied directly by user",
        "external31_access": False,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError("FINAL91 AUDIT FAILED. Do not train.\n" + "\n".join(failures[:30]))

    print("=" * 112)
    print("ROUND-5 / FINAL91 LIVE HUMAN_GOLD AUDIT — PASS")
    print(f"Frozen original47:       {len(source_ids)}")
    print(f"Prior Final82 labels:    {len(prior_ids)}")
    print(f"Round5 new labels:       {len(round5_ids)}")
    print(f"Total HUMAN_GOLD:        {len(current_ids)}")
    print("Geometry/non-empty:      PASS for all")
    print("Unexpected labels:       0")
    print("Missing expected labels: 0")
    print(f"Audit CSV:               {csv_path}")
    print(f"Audit metadata:          {json_path}")
    print("External31:              NOT ACCESSED")
    print("=" * 112)


if __name__ == "__main__":
    main()
