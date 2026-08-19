#!/usr/bin/env python3
"""Backward-compatible entrypoint for the controlled Final62 SSL experiment.

The canonical experiment is now cross-volume queue SSL v3. Earlier within-volume spatial-token
InfoNCE variants became nearly trivial by epoch 2-6, even after loss reweighting. This wrapper
routes the established command to the queue-based implementation so stale commands cannot
accidentally reproduce the superseded objectives.

Supports both:
    python scripts/train_controlled_ssl_final62_128.py ...
and:
    python -m scripts.train_controlled_ssl_final62_128 ...
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_controlled_ssl_final62_128_queue import main


if __name__ == "__main__":
    main()
