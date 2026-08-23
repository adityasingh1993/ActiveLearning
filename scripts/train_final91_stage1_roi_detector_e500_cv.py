#!/usr/bin/env python3
"""Dedicated 500-epoch Final91 Stage-1 ROI detector experiment.

This wrapper keeps the long detector run in a separate output directory so it cannot overwrite
or silently mix with the original 60-epoch experiment. Validation runs every 10 epochs plus the
first and final epoch; the checkpoint with the best held-out expanded-box Dice is retained.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_final91_stage1_roi_detector_cv import main as detector_main


DEFAULT_OUTPUT = "experiments/final91_stage1_roi_detector_e500_cv"


def main():
    parser = argparse.ArgumentParser(description="Final91 Stage-1 ROI detector — 500 epochs")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--fold",
        default="2",
        help="all, one fold 0..4, or comma-separated folds such as 1,2; default 2",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-metadata", default=None)
    parser.add_argument("--source-cv-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--postprocess-sweep",
        action="store_true",
        help="Reuse an existing E500 checkpoint and run only the crop-rule sweep",
    )
    args = parser.parse_args()

    forwarded = [
        "train_final91_stage1_roi_detector_cv.py",
        "--config", args.config,
        "--fold", args.fold,
        "--gpu", str(args.gpu),
        "--output-dir", args.output_dir,
        "--epochs", "500",
        "--validation-every-n-epochs", "10",
    ]
    if args.audit_metadata:
        forwarded.extend(["--audit-metadata", args.audit_metadata])
    if args.source_cv_dir:
        forwarded.extend(["--source-cv-dir", args.source_cv_dir])
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.postprocess_sweep:
        forwarded.append("--postprocess-sweep")

    sys.argv = forwarded
    detector_main()


if __name__ == "__main__":
    main()
