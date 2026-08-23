#!/usr/bin/env python3
"""Dedicated Final91 all-91 entry point for mild ultrasound appearance augmentation."""

import argparse
from pathlib import Path

CV_DIR = Path("experiments/round5_cv_91_a3")
AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_MANIFEST = Path("experiments/cv5_supervised_47_translation12/cv_splits.json")
OUTPUT = Path("experiments/final91_a4_appearance_all91")


def main():
    parser = argparse.ArgumentParser(
        description="Train Final91 A4 appearance variant on all 91 HUMAN_GOLD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--cv-dir", default=str(CV_DIR))
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from train_final91_a3_all91 import main as train_final91

    forwarded = [
        "--config", args.config,
        "--audit-metadata", args.audit_metadata,
        "--cv-dir", args.cv_dir,
        "--source-manifest", args.source_manifest,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--appearance-augmentation",
    ]
    if args.epochs is not None:
        forwarded.extend(["--epochs", str(args.epochs)])
    if args.overwrite:
        forwarded.append("--overwrite")
    train_final91(forwarded)


if __name__ == "__main__":
    main()
