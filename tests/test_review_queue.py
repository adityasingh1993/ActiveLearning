"""Tests for hassl/active/review_queue.py"""
import json
import pytest
from pathlib import Path

from hassl.active.review_queue import ReviewQueue


@pytest.fixture
def queue_path(tmp_path):
    return str(tmp_path / "logs" / "human_review_queue.json")


def test_review_queue_initialises_empty(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    assert len(q.entries) == 0
    assert q.get_pending() == []


def test_add_failed_creates_entry(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed(
        volume_id="vol_001",
        qc_flags=["uncertainty", "topology"],
        qc_score=0.23,
        sha256_hash="abc123",
        round_num=1,
    )
    assert "vol_001" in q.entries
    assert q.entries["vol_001"]["status"] == ReviewQueue.STATUS_PENDING
    assert q.entries["vol_001"]["qc_flags"] == ["uncertainty", "topology"]
    assert q.entries["vol_001"]["round_added"] == 1


def test_get_pending_returns_pending_entries(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["uncertainty"], 0.2, "hash1", 0)
    q.add_failed("vol_002", ["volume"], 0.3, "hash2", 0)
    pending = q.get_pending()
    assert len(pending) == 2
    vol_ids = [e["volume_id"] for e in pending]
    assert "vol_001" in vol_ids
    assert "vol_002" in vol_ids


def test_mark_in_review(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["topology"], 0.4, "hash1", 1)
    q.mark_in_review("vol_001")
    assert q.entries["vol_001"]["status"] == ReviewQueue.STATUS_IN_REVIEW
    # Should still appear in get_pending (in_review is included)
    assert len(q.get_pending()) == 1


def test_mark_resolved_removes_from_pending(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["volume"], 0.25, "hash1", 0)
    q.mark_resolved("vol_001", corrected_label_path="/data/labels/vol_001.nii.gz")
    assert q.entries["vol_001"]["status"] == ReviewQueue.STATUS_RESOLVED
    assert q.entries["vol_001"]["corrected_label_path"] == "/data/labels/vol_001.nii.gz"
    assert len(q.get_pending()) == 0


def test_resolved_entry_not_overwritten(queue_path):
    """A resolved volume should not be re-added to the queue."""
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["volume"], 0.25, "hash1", 0)
    q.mark_resolved("vol_001")
    q.add_failed("vol_001", ["uncertainty"], 0.10, "hash2", 1)  # Should be skipped
    assert q.entries["vol_001"]["status"] == ReviewQueue.STATUS_RESOLVED
    assert q.entries["vol_001"]["qc_flags"] == ["volume"]  # Original flags preserved


def test_queue_persists_to_disk(tmp_path):
    queue_path = str(tmp_path / "logs" / "queue.json")
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["uncertainty"], 0.18, "hashXYZ", 2)

    # Reload from disk
    q2 = ReviewQueue(queue_path=queue_path)
    assert "vol_001" in q2.entries
    assert q2.entries["vol_001"]["qc_score"] == pytest.approx(0.18, abs=1e-5)


def test_queue_ledger_hash_in_file(tmp_path):
    queue_path = str(tmp_path / "logs" / "queue.json")
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["convexity"], 0.55, "hashABC", 0)

    with open(queue_path) as f:
        data = json.load(f)
    assert "ledger_hash" in data
    assert len(data["ledger_hash"]) == 64


def test_stats_summary(queue_path):
    q = ReviewQueue(queue_path=queue_path)
    q.add_failed("vol_001", ["uncertainty", "volume"], 0.1, "h1", 0)
    q.add_failed("vol_002", ["topology"], 0.3, "h2", 0)
    q.mark_resolved("vol_002")

    s = q.stats()
    assert s["total"] == 2
    assert s["pending"] == 1
    assert s["resolved"] == 1
    assert "uncertainty" in s["top_failure_reasons"]
