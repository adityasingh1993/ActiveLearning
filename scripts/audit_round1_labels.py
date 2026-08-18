#!/usr/bin/env python3
"""Audit newly added Round-1 human labels against the frozen 47-case source set.

The user-facing workflow keeps new corrected labels in the same central label folder used
by the original dataset. This script discovers those labels automatically:

    frozen source labels = IDs in experiments/cv5_supervised_47_translation12/cv_splits.json
    current human labels = cases visible through train_supervised_cv.collect_cases()
    new Round-1 labels   = current human labels - frozen source labels

For every visible human label it checks image/label readability, non-empty foreground, and
SimpleITK grid geometry (size/spacing/origin/direction). The frozen 47 must all still exist.
By default exactly 9 new usable labels are expected for the first active-learning round.

If the previous active-learning batch manifest exists, the audit also records selected cases
that are still unlabeled. With the current 10-case batch this should identify the one case
reported as not annotatable because the target was not visible.

Outputs (default experiments/round1_supervised_56_translation12/):
  round1_label_audit.csv
  round1_label_audit.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("audit_round1_labels.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv


DEFAULT_SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
DEFAULT_SELECTION_MANIFEST = Path("experiments/auto_label_pool_v1/active_learning_batch.csv")
DEFAULT_OUTPUT_DIR = Path("experiments/round1_supervised_56_translation12")


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_equal(actual, expected, atol=1e-5):
    return bool(
        np.allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=1e-6,
            atol=atol,
        )
    )


def audit_case(case):
    image_path = Path(case["image"])
    label_path = Path(case["label"])
    row = {
        "case_id": str(case["id"]),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_exists": int(image_path.exists()),
        "label_exists": int(label_path.exists()),
        "foreground_voxels": "",
        "label_min": "",
        "label_max": "",
        "size_match": 0,
        "spacing_match": 0,
        "origin_match": 0,
        "direction_match": 0,
        "geometry_match": 0,
        "audit_ok": 0,
        "audit_error": "",
    }
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"missing label: {label_path}")

        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label)
        finite = np.asarray(label_array)[np.isfinite(label_array)]
        if finite.size == 0:
            raise RuntimeError("label contains no finite voxels")

        foreground = int(np.count_nonzero(finite > 0))
        row["foreground_voxels"] = foreground
        row["label_min"] = float(np.min(finite))
        row["label_max"] = float(np.max(finite))
        if foreground <= 0:
            raise RuntimeError("label has no positive foreground voxels")

        row["size_match"] = int(tuple(label.GetSize()) == tuple(image.GetSize()))
        row["spacing_match"] = int(_geometry_equal(label.GetSpacing(), image.GetSpacing()))
        row["origin_match"] = int(_geometry_equal(label.GetOrigin(), image.GetOrigin()))
        row["direction_match"] = int(_geometry_equal(label.GetDirection(), image.GetDirection()))
        row["geometry_match"] = int(
            row["size_match"]
            and row["spacing_match"]
            and row["origin_match"]
            and row["direction_match"]
        )
        if not row["geometry_match"]:
            raise RuntimeError(
                "image/label geometry mismatch: "
                f"image size={image.GetSize()} label size={label.GetSize()}"
            )
        row["audit_ok"] = 1
    except Exception as exc:
        row["audit_error"] = str(exc)
    return row


def load_source_ids(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_ids = [str(x) for x in payload.get("all_case_ids", [])]
    if len(source_ids) != 47 or len(set(source_ids)) != 47:
        raise RuntimeError(
            f"Expected frozen manifest with exactly 47 unique all_case_ids, found {len(set(source_ids))}: {path}"
        )
    return payload, set(source_ids)


def discover_round1_cases(config, source_manifest_path=DEFAULT_SOURCE_MANIFEST):
    """Return current labeled cases, frozen IDs, and newly added human-labeled cases."""
    source_manifest_path = Path(source_manifest_path)
    source_manifest, source_ids = load_source_ids(source_manifest_path)
    cases = cv.collect_cases(config)
    by_id = {str(case["id"]): case for case in cases}
    if len(by_id) != len(cases):
        raise RuntimeError("Duplicate labeled case IDs detected in current dataset")

    current_ids = set(by_id)
    missing_frozen = sorted(source_ids - current_ids)
    if missing_frozen:
        raise RuntimeError(
            "Frozen source labels disappeared from the current dataset: " + ", ".join(missing_frozen)
        )

    new_ids = sorted(current_ids - source_ids)
    return source_manifest, source_ids, by_id, new_ids


def main():
    parser = argparse.ArgumentParser(
        description="Audit the 47 frozen + newly added Round-1 human labels"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-new-labels", type=int, default=9)
    parser.add_argument(
        "--allow-new-count-mismatch",
        action="store_true",
        help="Report rather than fail if the number of newly discovered labels is not --expected-new-labels.",
    )
    parser.add_argument(
        "--missing-selected-reason",
        default="target_not_visible",
        help="Reason recorded when exactly one previously selected case remains unlabeled.",
    )
    args = parser.parse_args()

    if args.expected_new_labels < 1:
        parser.error("--expected-new-labels must be >=1")

    config = HASSLConfig.from_yaml(args.config)
    source_manifest_path = Path(args.source_manifest)
    selection_manifest_path = Path(args.selection_manifest)
    output_dir = Path(args.output_dir)

    _, source_ids, by_id, new_ids = discover_round1_cases(config, source_manifest_path)
    if len(new_ids) != args.expected_new_labels and not args.allow_new_count_mismatch:
        raise RuntimeError(
            f"Expected {args.expected_new_labels} newly added human labels, found {len(new_ids)}: {new_ids}. "
            "Resolve the label folder or use --allow-new-count-mismatch only if intentional."
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
        if case_id in source_ids:
            status = "FROZEN_SOURCE_LABEL"
        else:
            status = "ROUND1_NEW_HUMAN_LABEL"
        row = {"status": status, **row, "reason": ""}
        audit_rows.append(row)
        if not int(row["audit_ok"]):
            failures.append(f"{case_id}: {row['audit_error']}")

    selected_unlabeled = sorted(set(selected_ids) - set(by_id)) if selected_ids else []
    selected_new_labeled = sorted(set(selected_ids) & set(new_ids)) if selected_ids else []
    for case_id in selected_unlabeled:
        audit_rows.append({
            "status": "NOT_ANNOTATABLE",
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
            "reason": args.missing_selected_reason if len(selected_unlabeled) == 1 else "selected_but_unlabeled",
        })

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            "Round-1 label audit failed. Do not train until every visible human label passes geometry/non-empty checks.\n"
            + preview
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "round1_label_audit.csv"
    json_path = output_dir / "round1_label_audit.json"
    write_csv(csv_path, audit_rows)

    metadata = {
        "version": "round1_label_audit_v1",
        "source_manifest": str(source_manifest_path),
        "selection_manifest": str(selection_manifest_path) if selection_manifest_path.exists() else None,
        "n_frozen_source_labels": len(source_ids),
        "n_current_valid_human_labels": len(by_id),
        "n_new_human_labels": len(new_ids),
        "new_human_label_ids": new_ids,
        "expected_new_human_labels": int(args.expected_new_labels),
        "selected_ids": selected_ids,
        "selected_new_labeled_ids": selected_new_labeled,
        "selected_unlabeled_ids": selected_unlabeled,
        "selected_unlabeled_reason": (
            args.missing_selected_reason if len(selected_unlabeled) == 1 else None
        ),
        "all_visible_labels_passed_audit": True,
        "training_rule": "Round-1 training may use frozen source labels plus new_human_label_ids only; NOT_ANNOTATABLE cases are excluded.",
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 100)
    print("ROUND-1 HUMAN LABEL AUDIT")
    print(f"Frozen source labels:    {len(source_ids)}")
    print(f"New human labels:        {len(new_ids)}")
    print(f"Total usable labels:     {len(by_id)}")
    print(f"New IDs:                 {', '.join(new_ids)}")
    if selected_ids:
        print(f"Previous selected batch: {len(selected_ids)}")
        print(f"Selected + labeled:      {len(selected_new_labeled)}")
        print(f"Selected + unlabeled:    {len(selected_unlabeled)}")
        for case_id in selected_unlabeled:
            reason = args.missing_selected_reason if len(selected_unlabeled) == 1 else "selected_but_unlabeled"
            print(f"  NOT_ANNOTATABLE: {case_id} | reason={reason}")
    print("Geometry/non-empty audit: PASS")
    print(f"Audit CSV:               {csv_path}")
    print(f"Audit metadata:          {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
