#!/usr/bin/env python3
"""Reproduce the established Final62 128^3 no-SSL supervised fit on the current code path.

Purpose
-------
Sanity-check the historical Final62 128^3 baseline after the 160^3 experiment. This wrapper
uses the same provenance-safe final trainer used by the 160^3 experiment, but locks resolution
back to 128^3 and writes to a fresh namespace. No SSL checkpoint is loaded.

The intended comparison is:
    historical Final62 128^3 no SSL
    vs
    current-code Final62 128^3 no SSL reproduction

All labels, fixed epoch rule, seed, optimizer, loss, EMA behavior, augmentation, and threshold
remain unchanged.
"""

import os
import sys


def _consume_gpu_argument(argv):
    args = list(argv)
    gpu = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == "--gpu":
            if i + 1 >= len(args):
                raise SystemExit("--gpu requires 0 or 1")
            gpu = args[i + 1]
            i += 2
            continue
        if token.startswith("--gpu="):
            gpu = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if gpu is not None:
        if gpu not in {"0", "1"}:
            raise SystemExit(f"--gpu must be 0 or 1, got {gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu, cleaned


def _has_option(argv, name):
    return any(token == name or token.startswith(name + "=") for token in argv[1:])


def _option_value(argv, name):
    for i, token in enumerate(argv[1:], start=1):
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


SELECTED_GPU, CLEAN_ARGV = _consume_gpu_argument(sys.argv)

explicit_resize = _option_value(CLEAN_ARGV, "--resize-size")
if explicit_resize is not None and int(explicit_resize) != 128:
    raise SystemExit(
        f"This reproducibility experiment is locked to --resize-size 128, got {explicit_resize}"
    )
if not _has_option(CLEAN_ARGV, "--resize-size"):
    CLEAN_ARGV.extend(["--resize-size", "128"])

if not _has_option(CLEAN_ARGV, "--output-dir"):
    CLEAN_ARGV.extend(
        ["--output-dir", "experiments/final_supervised_round2_62_translation12_resize128_repro"]
    )

sys.argv = CLEAN_ARGV

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_final_supervised_round2_from_audit import main


if __name__ == "__main__":
    print("=" * 112)
    print("FINAL62 BASELINE REPRODUCTION — 128^3 — NO SSL")
    print("Purpose: reproduce the historical 128^3 baseline on the same current code path as 160^3")
    print("Frozen: 62 HUMAN_GOLD | seed 42 | DiceCE | AdamW 1e-4 | EMA | translation +/-12 | threshold .50")
    print("SSL initialization: OFF")
    print(
        "Physical GPU: "
        + (SELECTED_GPU if SELECTED_GPU is not None else "<environment/config>")
    )
    print("=" * 112)
    main()
