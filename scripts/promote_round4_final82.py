#!/usr/bin/env python3
"""Safely promote audited Round-4 labels and verify the controlled Final82 HUMAN_GOLD state.

This is the controlled transition from Final72 -> Final82.

Requirements
------------
- previous Final72 promoted-training audit is passing (exactly 72 HUMAN_GOLD),
- Round-4 annotation-pack audit is passing (exactly 10 selected annotations),
- Round-4 IDs do not overlap the prior 72,
- every selected source image is discoverable under config.data_dir,
- every reviewed mask is readable, non-empty, and geometry-matched to its image,
- no destination label already exists.

The central label directory is resolved exactly as the supervised CV collector does:
    Path(config.data_dir) / "labels"

After promotion the script rediscoveries the whole labeled dataset and requires exactly:
    72 prior HUMAN_GOLD + 10 Round-4 HUMAN_GOLD = 82 total.
Every one of the 82 image/label pairs is audited again.

No external31 label or metric is read.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from scripts.audit_round1_labels import audit_case, discover_round1_cases, write_csv

SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
FINAL72_AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
ROUND4_AUDIT_DIR = Path("experiments/round4_active_a3_committee_v1/human_annotation_audit")
ROUND4_AUDIT_JSON = ROUND4_AUDIT_DIR / "round4_human_annotation_audit.json"
ROUND4_AUDIT_CSV = ROUND4_AUDIT_DIR / "round4_human_annotation_audit.csv"
OUTPUT_DIR = Path("experiments/round4_supervised_82_a3")

EXPECTED_SOURCE = 47
EXPECTED_PRIOR = 72
EXPECTED_ROUND4 = 10
EXPECTED_TOTAL = 82


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
    p = argparse.ArgumentParser(description="Promote audited Round4 HUMAN_GOLD and verify Final82")
    p.add_argument("--config", required=True)
    p.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    p.add_argument("--final72-audit", default=str(FINAL72_AUDIT))
    p.add_argument("--round4-audit-json", default=str(ROUND4_AUDIT_JSON))
    p.add_argument("--round4-audit-csv", default=str(ROUND4_AUDIT_CSV))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    source_manifest_path = Path(args.source_manifest)
    final72_path = Path(args.final72_audit)
    r4_json_path = Path(args.round4_audit_json)
    r4_csv_path = Path(args.round4_audit_csv)
    output_dir = Path(args.output_dir)

    config = HASSLConfig.from_yaml(args.config)
    root = Path(config.data_dir)
    label_dir = root / "labels"

    prior = read_json(final72_path)
    if not prior.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Final72 promoted-training audit is not passing")
    prior_ids = sorted(str(x) for x in prior.get("all_current_human_label_ids", []))
    if len(prior_ids) != EXPECTED_PRIOR or len(set(prior_ids)) != EXPECTED_PRIOR:
        raise RuntimeError(f"Expected exactly {EXPECTED_PRIOR} unique prior HUMAN_GOLD IDs")

    r4 = read_json(r4_json_path)
    if not r4.get("all_round4_annotations_passed", False):
        raise RuntimeError("Round4 annotation audit is not passing")
    round4_ids = sorted(str(x) for x in r4.get("selected_ids", []))
    if len(round4_ids) != EXPECTED_ROUND4 or len(set(round4_ids)) != EXPECTED_ROUND4:
        raise RuntimeError(f"Expected exactly {EXPECTED_ROUND4} unique Round4 selected IDs")
    overlap = sorted(set(prior_ids) & set(round4_ids))
    if overlap:
        raise RuntimeError("Round4 IDs overlap prior Final72 HUMAN_GOLD: " + ", ".join(overlap))

    audit_rows = read_csv(r4_csv_path)
    audited_by_id = {str(r["case_id"]): r for r in audit_rows if int(r.get("audit_ok", 0)) == 1}
    if set(audited_by_id) != set(round4_ids):
        raise RuntimeError("Passing Round4 audit CSV IDs do not match Round4 selected IDs")

    # Confirm current state is still exactly Final72 before any write.
    _, source_ids, current_by_id, _ = discover_round1_cases(config, source_manifest_path)
    current_ids = sorted(str(x) for x in current_by_id)
    if current_ids != prior_ids:
        unexpected = sorted(set(current_ids) - set(prior_ids))
        missing = sorted(set(prior_ids) - set(current_ids))
        raise RuntimeError(
            "Current training labels changed since Final72 audit; refusing promotion.\n"
            f"Unexpected: {unexpected}\nMissing: {missing}"
        )
    if len(source_ids) != EXPECTED_SOURCE:
        raise RuntimeError(f"Expected frozen original{EXPECTED_SOURCE}, found {len(source_ids)}")

    # Full preflight: source masks, source images in config.data_dir, and destination collisions.
    promotions = []
    errors = []
    for case_id in round4_ids:
        row = audited_by_id[case_id]
        src = Path(str(row.get("human_segmentation_path", "")))
        image_from_audit = Path(str(row.get("image_path", "")))
        dst = label_dir / f"{case_id}{config.label_suffix}"

        candidates = sorted(root.rglob(f"{case_id}{config.image_suffix}")) if root.exists() else []
        if not src.exists():
            errors.append(f"{case_id}: audited human segmentation missing: {src}")
            continue
        if not image_from_audit.exists():
            errors.append(f"{case_id}: audited image missing: {image_from_audit}")
            continue
        if len(candidates) != 1:
            errors.append(
                f"{case_id}: expected exactly one discoverable image under {root}, found {len(candidates)}: {candidates[:5]}"
            )
            continue
        if dst.exists():
            errors.append(f"{case_id}: destination label already exists: {dst}")
            continue

        # Re-audit the exact source annotation against the image that will be discovered for training.
        check = audit_case({"id": case_id, "image": str(candidates[0]), "label": str(src)})
        if not int(check.get("audit_ok", 0)):
            errors.append(f"{case_id}: pre-promotion image/label audit failed: {check.get('audit_error', '')}")
            continue
        promotions.append((case_id, src, dst, candidates[0]))

    if errors or len(promotions) != EXPECTED_ROUND4:
        raise RuntimeError("Final82 promotion preflight FAILED. Nothing copied.\n" + "\n".join(errors))

    label_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    try:
        for case_id, src, dst, image_path in promotions:
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"PROMOTED {case_id}: {src} -> {dst}")
    except Exception:
        # Roll back only files created by this invocation.
        for path in reversed(copied):
            try:
                path.unlink()
            except Exception:
                pass
        raise

    # Verify exact Final82 state after promotion.
    _, source_ids2, by_id82, _ = discover_round1_cases(config, source_manifest_path)
    current82 = sorted(str(x) for x in by_id82)
    expected82 = sorted(set(prior_ids) | set(round4_ids))
    unexpected = sorted(set(current82) - set(expected82))
    missing = sorted(set(expected82) - set(current82))
    if current82 != expected82 or len(current82) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Promotion completed but Final82 discovery is not the expected controlled state.\n"
            f"Current={len(current82)} Expected={EXPECTED_TOTAL}\nUnexpected={unexpected}\nMissing={missing}"
        )

    rows = []
    failures = []
    prior_set = set(prior_ids)
    r4_set = set(round4_ids)
    for case_id in current82:
        row = audit_case(by_id82[case_id])
        status = "ROUND4_NEW_HUMAN_GOLD" if case_id in r4_set else "PRIOR_FINAL72_HUMAN_GOLD"
        row = {"status": status, **row}
        rows.append(row)
        if not int(row.get("audit_ok", 0)):
            failures.append(f"{case_id}: {row.get('audit_error', '')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "round4_label_audit.csv"
    json_out = output_dir / "round4_label_audit.json"
    write_csv(csv_out, rows)

    metadata = {
        "version": "round4_final82_promoted_training_set_audit_v1",
        "source_manifest": str(source_manifest_path),
        "previous_final72_audit": str(final72_path),
        "round4_annotation_audit": str(r4_json_path),
        "central_label_dir": str(label_dir),
        "n_frozen_source_labels": len(source_ids2),
        "n_prior_final72_human_labels": len(prior_ids),
        "n_round4_new_human_labels": len(round4_ids),
        "n_current_valid_human_labels": len(current82),
        "prior_final72_human_label_ids": prior_ids,
        "round4_new_human_label_ids": round4_ids,
        "all_current_human_label_ids": current82,
        "unexpected_new_label_ids": unexpected,
        "missing_expected_label_ids": missing,
        "all_visible_labels_passed_audit": len(failures) == 0,
        "selection_provenance_enforced": True,
        "training_rule": (
            "Controlled Final82 uses exact prior Final72 HUMAN_GOLD plus exact 10 audited Round4 human annotations. "
            "All post-original47 labels are train-only in CV; external31 is excluded."
        ),
    }
    json_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError("Final82 full training-set audit FAILED. Do not train.\n" + "\n".join(failures[:20]))

    print("=" * 112)
    print("ROUND-4 / FINAL82 HUMAN_GOLD PROMOTION + AUDIT — PASS")
    print(f"Frozen original source:  {len(source_ids2)}")
    print(f"Prior Final72 labels:    {len(prior_ids)}")
    print(f"Round4 new labels:       {len(round4_ids)}")
    print(f"Total HUMAN_GOLD:        {len(current82)}")
    print("Geometry/non-empty:      PASS for all")
    print("Unexpected labels:       0")
    print("Missing expected labels: 0")
    print(f"Central label dir:       {label_dir}")
    print(f"Audit CSV:               {csv_out}")
    print(f"Audit metadata:          {json_out}")
    print("=" * 112)


if __name__ == "__main__":
    main()
