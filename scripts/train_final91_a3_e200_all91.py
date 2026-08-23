#!/usr/bin/env python3
"""Dedicated Final91 A3 all-91 long-training entry point (200 epochs by default)."""

import argparse
from pathlib import Path

CV_DIR = Path("experiments/round5_cv_91_a3")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
OUTPUT = Path("experiments/final91_a3_all91_e200")


def main():
    parser = argparse.ArgumentParser(
        description="Train Final91 A3 on all 91 HUMAN_GOLD (200 epochs by default)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--cv-dir", default=str(CV_DIR))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-every-n-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 101 or args.epochs > 300:
        parser.error("--epochs must be between 101 and 300")
    if args.validation_every_n_epochs < 1:
        parser.error("--validation-every-n-epochs must be >=1")

    from train_final91_a3_all91 import main as train_final91

    forwarded = [
        "--config", args.config,
        "--audit-metadata", args.audit_metadata,
        "--cv-dir", args.cv_dir,
        "--source-manifest", args.source_manifest,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--validation-every-n-epochs", str(args.validation_every_n_epochs),
        "--seed", str(args.seed),
    ]
    if args.overwrite:
        forwarded.append("--overwrite")
    train_final91(forwarded)


if __name__ == "__main__":
    main()
