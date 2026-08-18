#!/usr/bin/env python3
"""Prepare a fresh 5-fold OOF split over all 62 audited HUMAN_GOLD cases.

This split is intentionally separate from the controlled Round-2 CV manifest. The prior
Round-2 comparison held out only the original 47 cases while the 15 later labels were
train-only. A guided refiner needs a coarse probability map for EVERY HUMAN_GOLD case from
a model that did not train on that case, so we create a new all-62 OOF manifest here.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv

DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/guided_refiner_oof62_coarse_v1")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Prepare leakage-safe all-62 OOF split for guided refiner")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite-manifest", action="store_true")
    args = p.parse_args()

    if args.folds < 2:
        p.error("--folds must be >=2")

    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "cv_splits.json"
    metadata_path = output_dir / "oof62_split_metadata.json"

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    reported_n = int(audit.get("n_current_valid_human_labels", len(audited_ids)))
    if len(audited_ids) != reported_n or reported_n != 62:
        raise RuntimeError(f"Expected exactly 62 audited HUMAN_GOLD cases, found {len(audited_ids)}")

    config = HASSLConfig.from_yaml(args.config)
    cases = cv.collect_cases(config)
    current_ids = sorted(str(c["id"]) for c in cases)
    if current_ids != audited_ids:
        added = sorted(set(current_ids) - set(audited_ids))
        missing = sorted(set(audited_ids) - set(current_ids))
        raise RuntimeError(
            "Current labeled dataset no longer matches the frozen Final62 audit.\n"
            f"Unexpected current IDs: {added}\nMissing audited IDs: {missing}"
        )

    patient_regex = getattr(config, "patient_id_regex", None)
    proposed = cv.create_manifest(cases, int(args.folds), int(args.seed), patient_regex)

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not args.overwrite_manifest:
        existing = read_json(manifest_path)
        cv.validate_manifest(existing, cases, int(args.folds), manifest_path)
        if existing != proposed:
            raise RuntimeError(
                f"Existing {manifest_path} is valid but differs from the deterministic proposed split. "
                "Refusing to silently replace it; use --overwrite-manifest only intentionally."
            )
        manifest = existing
        state = "REUSED"
    else:
        manifest = proposed
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        state = "CREATED"

    held_out = [str(x) for fold in manifest["folds"] for x in fold["val_ids"]]
    if len(held_out) != 62 or sorted(held_out) != audited_ids or len(set(held_out)) != 62:
        raise RuntimeError("OOF manifest does not hold out every audited case exactly once")

    metadata = {
        "version": "guided_refiner_oof62_split_v1",
        "purpose": "Generate leakage-safe coarse EMA probability guidance for all 62 HUMAN_GOLD cases",
        "source_audit": str(audit_path),
        "n_cases": len(audited_ids),
        "n_folds": int(args.folds),
        "seed": int(args.seed),
        "patient_id_regex": patient_regex,
        "case_ids": audited_ids,
        "manifest": str(manifest_path),
        "warning": (
            "This manifest is separate from the controlled Round-2 original47 evaluation split. "
            "It exists only to create OOF guidance channels for offline refiner training."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 108)
    print("GUIDED REFINER OOF62 SPLIT READY")
    print(f"State:             {state}")
    print(f"Audited cases:     {len(audited_ids)}")
    print(f"Folds:             {args.folds}")
    for fold in manifest["folds"]:
        print(f"  fold {fold['fold']}: train={len(fold['train_ids'])} held-out={len(fold['val_ids'])}")
    print("Held out exactly once: PASS")
    print(f"Manifest:          {manifest_path}")
    print(f"Metadata:          {metadata_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
