#!/usr/bin/env python3
"""Controlled Final62 160^3 supervised training experiment.

This entrypoint intentionally changes exactly one modeling variable relative to the established
Final62 baseline: spatial resize 128^3 -> 160^3. It reuses the provenance-safe Final62 trainer,
so labels, fixed epoch rule, DiceCE, AdamW, EMA, translation-only augmentation, threshold, and
lambda_unsup=0 remain unchanged. No SSL checkpoint is loaded.

Examples
--------
python scripts/train_final62_resolution160.py --config config_resize128.yaml --gpu 1
python scripts/train_final62_resolution160.py --config config_resize128.yaml --gpu 0 --overwrite
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

# Lock the experiment to 160^3. Refuse a conflicting explicit value rather than silently
# changing the requested experiment.
explicit_resize = _option_value(CLEAN_ARGV, "--resize-size")
if explicit_resize is not None and int(explicit_resize) != 160:
    raise SystemExit(
        f"This controlled experiment is locked to --resize-size 160, got {explicit_resize}"
    )
if not _has_option(CLEAN_ARGV, "--resize-size"):
    CLEAN_ARGV.extend(["--resize-size", "160"])

if not _has_option(CLEAN_ARGV, "--output-dir"):
    CLEAN_ARGV.extend(
        ["--output-dir", "experiments/final_supervised_round2_62_translation12_resize160"]
    )

sys.argv = CLEAN_ARGV

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_final_supervised_round2_from_audit import main


if __name__ == "__main__":
    print("=" * 112)
    print("CONTROLLED FINAL62 RESOLUTION EXPERIMENT — 160^3 — NO SSL")
    print("Only intended model change: resize128^3 -> resize160^3")
    print("Frozen: 62 HUMAN_GOLD | DiceCE | AdamW 1e-4 | EMA | translation +/-12 | threshold .50")
    print("SSL initialization: OFF")
    print(
        "Physical GPU: "
        + (SELECTED_GPU if SELECTED_GPU is not None else "<environment/config>")
    )
    print("=" * 112)
    main()
