"""
HASSL Human Review Queue.

Tracks segmentation predictions that failed the QC Gate and need to be
manually reviewed and corrected by a clinician or annotation expert before
they can be included in the training pool.

The queue is persisted as a JSON file (`human_review_queue.json`) alongside
the pool manifest. Each entry contains:
    - volume_id       : Unique volume identifier
    - qc_flags        : Which QC checkers failed
    - qc_score        : Composite QC quality score
    - sha256_hash     : Tamper-evident audit hash (FDA SaMD compliant)
    - status          : "pending" | "in_review" | "resolved"
    - round_added     : Active learning round number
    - preseg_path     : Optional path to pre-segmentation mask for Slicer

When a volume is resolved (corrected by a clinician), it is moved to the
labeled pool via the QueryEngine.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReviewQueue:
    """Persistent human review queue for QC Gate failures.

    Args:
        queue_path: Path to the JSON queue file. Created if missing.
    """

    STATUS_PENDING = "pending"
    STATUS_IN_REVIEW = "in_review"
    STATUS_RESOLVED = "resolved"

    def __init__(self, queue_path: str = "./experiments/logs/human_review_queue.json"):
        self.queue_path = Path(queue_path)
        self.entries: Dict[str, dict] = {}
        self._load()

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load queue from disk if it exists."""
        if self.queue_path.exists():
            try:
                with open(self.queue_path, "r") as f:
                    data = json.load(f)
                self.entries = data.get("entries", {})
                logger.debug("ReviewQueue: loaded %d entries from %s", len(self.entries), self.queue_path)
            except Exception as e:
                logger.warning("ReviewQueue: could not load queue file (%s). Starting fresh.", e)
                self.entries = {}

    def _save(self) -> None:
        """Persist queue to disk with SHA-256 ledger hash."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.entries, sort_keys=True).encode("utf-8")
        ledger_hash = hashlib.sha256(payload).hexdigest()
        data = {"entries": self.entries, "ledger_hash": ledger_hash}
        with open(self.queue_path, "w") as f:
            json.dump(data, f, indent=4)

    # ──────────────────────────────────────────────────────────────────────
    # Queue Operations
    # ──────────────────────────────────────────────────────────────────────

    def add_failed(
        self,
        volume_id: str,
        qc_flags: List[str],
        qc_score: float,
        sha256_hash: str,
        round_num: int = 0,
        preseg_path: Optional[str] = None,
    ) -> None:
        """Add a QC-failed volume to the review queue.

        If the volume is already in the queue, the entry is updated unless
        it has already been resolved.

        Args:
            volume_id: Unique volume identifier.
            qc_flags: List of QC checker names that failed.
            qc_score: Composite QC quality score [0, 1].
            sha256_hash: Audit hash from QCReport.
            round_num: Active learning round number.
            preseg_path: Optional path to pre-segmentation NIfTI for Slicer.
        """
        if volume_id in self.entries and self.entries[volume_id]["status"] == self.STATUS_RESOLVED:
            logger.debug(
                "ReviewQueue.add_failed: %s already resolved — skipping.", volume_id
            )
            return

        self.entries[volume_id] = {
            "volume_id": volume_id,
            "qc_flags": qc_flags,
            "qc_score": round(qc_score, 6),
            "sha256_hash": sha256_hash,
            "status": self.STATUS_PENDING,
            "round_added": round_num,
            "preseg_path": preseg_path,
        }
        self._save()
        logger.warning(
            "ReviewQueue: volume '%s' queued for human review (flags=%s, score=%.3f).",
            volume_id,
            qc_flags,
            qc_score,
        )

    def get_pending(self) -> List[dict]:
        """Return all entries with status 'pending' or 'in_review'."""
        return [
            e for e in self.entries.values()
            if e["status"] in (self.STATUS_PENDING, self.STATUS_IN_REVIEW)
        ]

    def mark_in_review(self, volume_id: str) -> None:
        """Mark a volume as currently being reviewed by a clinician."""
        if volume_id in self.entries:
            self.entries[volume_id]["status"] = self.STATUS_IN_REVIEW
            self._save()

    def mark_resolved(self, volume_id: str, corrected_label_path: Optional[str] = None) -> None:
        """Mark a volume as resolved after human correction.

        Args:
            volume_id: Volume that has been corrected.
            corrected_label_path: Optional path to the corrected label file.
        """
        if volume_id not in self.entries:
            logger.warning("ReviewQueue.mark_resolved: '%s' not found in queue.", volume_id)
            return

        self.entries[volume_id]["status"] = self.STATUS_RESOLVED
        if corrected_label_path:
            self.entries[volume_id]["corrected_label_path"] = corrected_label_path
        self._save()
        logger.info("ReviewQueue: '%s' marked as resolved.", volume_id)

    # ──────────────────────────────────────────────────────────────────────
    # Statistics & Reporting
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics of the review queue."""
        statuses = [e["status"] for e in self.entries.values()]
        all_flags: List[str] = []
        for e in self.entries.values():
            all_flags.extend(e.get("qc_flags", []))

        from collections import Counter
        return {
            "total": len(self.entries),
            "pending": statuses.count(self.STATUS_PENDING),
            "in_review": statuses.count(self.STATUS_IN_REVIEW),
            "resolved": statuses.count(self.STATUS_RESOLVED),
            "top_failure_reasons": dict(Counter(all_flags).most_common(5)),
        }

    def print_summary(self) -> None:
        """Print a human-readable summary of the review queue."""
        s = self.stats()
        print(f"\n{'─' * 50}")
        print(f"  Human Review Queue Summary")
        print(f"{'─' * 50}")
        print(f"  Total entries  : {s['total']}")
        print(f"  Pending        : {s['pending']}")
        print(f"  In review      : {s['in_review']}")
        print(f"  Resolved       : {s['resolved']}")
        if s["top_failure_reasons"]:
            print(f"  Top failure reasons:")
            for reason, count in s["top_failure_reasons"].items():
                print(f"    - {reason}: {count}")
        print(f"{'─' * 50}\n")
