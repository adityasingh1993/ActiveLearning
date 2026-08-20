#!/usr/bin/env python3
"""Run the locked SwinUNETR EMA threshold check on a standalone external31 dataset.

This wrapper fixes image resolution for the frozen external31 layout when those cases are NOT
members of the historical auto-label pool manifest. It matches image and GT files directly by
case ID, then delegates all inference, native-grid inversion, metrics, threshold-lock validation,
and Final62 training-overlap protection to
validate_final62_swinunetr_locked_threshold_external31.py.

Typical layout:
    /data/v1/compressed/image/<case_id>.mha
    /data/v1/compressed/label/<case_id>.seg.nrrd

No threshold is selected here. The candidate value is still read verbatim from the internal-OOF
locked_threshold.json created before this external evaluation.
"""

import sys
from pathlib import Path

# When this file is executed as `python scripts/<name>.py`, Python puts the scripts directory
# (not the repository root) on sys.path. Add the repo root before importing another scripts.*
# module so the wrapper works from a normal repository checkout/container invocation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def consume_required(argv, name):
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if value is None:
        raise SystemExit(f"{name} is required")
    return value, cleaned


def consume_optional(argv, name, default):
    args = list(argv)
    value = default
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


IMAGE_DIR_RAW, CLEAN_ARGV = consume_required(sys.argv, "--image-dir")
IMAGE_SUFFIX, CLEAN_ARGV = consume_optional(CLEAN_ARGV, "--image-suffix", ".mha")
IMAGE_DIR = Path(IMAGE_DIR_RAW)
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"External image directory does not exist: {IMAGE_DIR}")
if not IMAGE_SUFFIX:
    raise SystemExit("--image-suffix must be non-empty")

# Leave --gpu in argv. The delegated module consumes it before importing torch/MONAI, preserving
# the existing physical-GPU selection semantics.
sys.argv = CLEAN_ARGV

import scripts.validate_final62_swinunetr_locked_threshold_external31 as base  # noqa: E402


def collect_external_images(root: Path, suffix: str):
    by_id = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        name = path.name
        if not name.endswith(suffix):
            continue
        case_id = name[: -len(suffix)]
        if not case_id:
            continue
        if case_id in by_id:
            raise RuntimeError(
                "Duplicate external image case ID found:\n"
                f"  {case_id}\n  {by_id[case_id]}\n  {path}"
            )
        by_id[case_id] = path
    if not by_id:
        raise RuntimeError(f"No external images ending with {suffix!r} found under {root}")
    return by_id


IMAGE_BY_ID = collect_external_images(IMAGE_DIR, IMAGE_SUFFIX)


def resolve_direct_external_cases(_pool_rows, gt_by_id, expected_count):
    """Match the frozen external dataset directly by ID instead of via an AL pool manifest."""
    matched_ids = sorted(set(IMAGE_BY_ID) & set(gt_by_id))
    if expected_count > 0 and len(matched_ids) != expected_count:
        gt_without_image = sorted(set(gt_by_id) - set(IMAGE_BY_ID))
        image_without_gt = sorted(set(IMAGE_BY_ID) - set(gt_by_id))
        raise RuntimeError(
            "Frozen direct external-validation count mismatch.\n"
            f"Expected matched cases: {expected_count}\n"
            f"Matched image+GT IDs:    {len(matched_ids)}\n"
            f"GT IDs without image ({len(gt_without_image)}): {gt_without_image[:20]}\n"
            f"Image IDs without GT ({len(image_without_gt)}): {image_without_gt[:20]}"
        )

    return [
        {
            "id": case_id,
            "image": str(IMAGE_BY_ID[case_id]),
            "gt_path": str(gt_by_id[case_id]),
        }
        for case_id in matched_ids
    ]


# The delegated main() still performs:
# - exact n=31 enforcement,
# - threshold-lock integrity checks,
# - Final62 HUMAN_GOLD overlap/leakage protection,
# - exact native-grid probability inversion,
# - EMA-only 0.50 vs internally locked threshold evaluation.
base.resolve_validation_cases = resolve_direct_external_cases


if __name__ == "__main__":
    print("=" * 124)
    print("FINAL62 SWIN LOCKED-THRESHOLD EXTERNAL31 — DIRECT DATASET RESOLUTION")
    print(f"Image root:   {IMAGE_DIR}")
    print(f"Image suffix: {IMAGE_SUFFIX}")
    print(f"Images found: {len(IMAGE_BY_ID)}")
    print("Case matching: direct image/GT ID intersection; auto-label pool manifest is NOT used for case resolution")
    print("=" * 124)
    base.main()
