#!/usr/bin/env python3
"""Backward-compatible entrypoint for the controlled Final62 SSL experiment.

The canonical experiment is now weighted v2 after the equal-raw-weight run showed rotation
contributing ~98% of the objective by epoch 5-7. This wrapper intentionally routes the old
command to the weighted implementation so stale commands cannot accidentally reproduce the
superseded equal-weight recipe.
"""

from scripts.train_controlled_ssl_final62_128_weighted import main


if __name__ == "__main__":
    main()
