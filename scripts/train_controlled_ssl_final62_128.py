#!/usr/bin/env python3
"""Backward-compatible entrypoint for the controlled Final62 SSL experiment.

The canonical experiment is now weighted v2 after the equal-raw-weight run showed rotation
contributing ~98% of the objective by epoch 5-7. This wrapper intentionally routes the old
command to the weighted implementation so stale commands cannot accidentally reproduce the
superseded equal-weight recipe.

This file supports both:
    python scripts/train_controlled_ssl_final62_128.py ...
and:
    python -m scripts.train_controlled_ssl_final62_128 ...
"""

import sys
from pathlib import Path

# When this file is executed directly, Python adds <repo>/scripts to sys.path rather than
# <repo>. Add the repository root explicitly before importing through the scripts namespace.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_controlled_ssl_final62_128_weighted import main


if __name__ == "__main__":
    main()
