"""Tests for hassl/utils/qc_gate.py"""
import numpy as np
import pytest
import torch

from hassl.utils.qc_gate import (
    QCGate, QCReport,
    UncertaintyChecker, VolumeChecker, TopologyChecker, ConvexityChecker,
    _sha256_report,
)


# ─── UncertaintyChecker ──────────────────────────────────────────────────────

def test_uncertainty_checker_low_entropy_passes():
    checker = UncertaintyChecker(threshold=0.35)
    # Nearly perfect prediction (prob ~1.0) → very low entropy → passes
    prob = np.ones((16, 16, 16), dtype=np.float32) * 0.99
    score = checker.check(prob)
    assert score > 0.0
    assert checker.passes(prob)


def test_uncertainty_checker_high_entropy_fails():
    checker = UncertaintyChecker(threshold=0.35)
    # Maximum entropy (prob = 0.5 everywhere)
    prob = np.ones((16, 16, 16), dtype=np.float32) * 0.5
    score = checker.check(prob)
    assert score <= 0.0
    assert not checker.passes(prob)


# ─── VolumeChecker ───────────────────────────────────────────────────────────

def test_volume_checker_in_range_passes():
    checker = VolumeChecker(min_voxels=100, max_voxels=10_000)
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[:5, :5, :5] = 1.0   # 125 voxels — within [100, 10000]
    assert checker.passes(mask)
    assert checker.check(mask) == 1.0


def test_volume_checker_too_small_fails():
    checker = VolumeChecker(min_voxels=500, max_voxels=10_000)
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[0, 0, 0] = 1.0  # 1 voxel — too small
    assert not checker.passes(mask)
    assert checker.check(mask) < 1.0


def test_volume_checker_too_large_fails():
    checker = VolumeChecker(min_voxels=100, max_voxels=50)
    mask = np.ones((32, 32, 32), dtype=np.float32)  # 32768 voxels >> max
    assert not checker.passes(mask)


# ─── TopologyChecker ─────────────────────────────────────────────────────────

def test_topology_checker_single_component_passes():
    checker = TopologyChecker(max_cc=3)
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[5:15, 5:15, 5:15] = 1.0  # Single blob
    assert checker.passes(mask)
    assert checker.check(mask) == 1.0


def test_topology_checker_many_fragments_fails():
    checker = TopologyChecker(max_cc=2)
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    # Create 5 isolated single-voxel blobs
    for i in range(5):
        mask[i * 5, 0, 0] = 1.0
    assert not checker.passes(mask)
    assert checker.check(mask) < 1.0


# ─── ConvexityChecker ────────────────────────────────────────────────────────

def test_convexity_checker_solid_sphere_passes():
    checker = ConvexityChecker(min_convexity=0.60)
    # A solid cube is perfectly convex
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[8:24, 8:24, 8:24] = 1.0
    score = checker.check(mask)
    assert score >= 0.95   # near-perfect convexity
    assert checker.passes(mask)


def test_convexity_checker_empty_mask_fails():
    checker = ConvexityChecker(min_convexity=0.70)
    mask = np.zeros((16, 16, 16), dtype=np.float32)
    assert not checker.passes(mask)


# ─── QCGate Integration ───────────────────────────────────────────────────────

def test_qcgate_passes_clean_prediction():
    gate = QCGate(
        uncertainty_threshold=0.60,  # relaxed for test
        volume_min_voxels=50,
        volume_max_voxels=100_000,
        max_cc_count=5,
        surface_convexity_min=0.50,
    )
    prob = np.ones((16, 16, 16), dtype=np.float32) * 0.95
    mask = (prob > 0.5).astype(np.float32)
    report = gate.evaluate("vol_clean", prob, mask)
    assert report.passed
    assert report.score > 0.0
    assert len(report.flags) == 0
    assert len(report.sha256_hash) == 64


def test_qcgate_fails_on_high_uncertainty():
    gate = QCGate(uncertainty_threshold=0.10)  # very strict
    prob = np.ones((16, 16, 16), dtype=np.float32) * 0.5  # max entropy
    mask = (prob > 0.5).astype(np.float32)
    report = gate.evaluate("vol_uncertain", prob, mask)
    assert not report.passed
    assert "uncertainty" in report.flags


def test_qcgate_fails_on_volume_violation():
    gate = QCGate(volume_min_voxels=10_000)  # impossible for 16^3 mask
    prob = np.ones((16, 16, 16), dtype=np.float32) * 0.99
    mask = (prob > 0.5).astype(np.float32)  # 4096 voxels < 10000 min
    report = gate.evaluate("vol_small", prob, mask)
    assert not report.passed
    assert "volume" in report.flags


def test_qcgate_evaluate_tensor():
    gate = QCGate(
        uncertainty_threshold=0.60,
        volume_min_voxels=10,
        volume_max_voxels=100_000,
        max_cc_count=5,
        surface_convexity_min=0.40,
    )
    logit = torch.ones(1, 1, 8, 8, 8) * 3.0  # high logits → high confidence
    report = gate.evaluate_tensor("vol_tensor", logit)
    assert isinstance(report, QCReport)
    assert report.score >= 0.0


def test_sha256_is_deterministic():
    h1 = _sha256_report("vol_001", 0.85, ["topology"])
    h2 = _sha256_report("vol_001", 0.85, ["topology"])
    assert h1 == h2
    assert len(h1) == 64


def test_qcgate_medsam_crosscheck():
    gate = QCGate(
        uncertainty_threshold=0.60,
        volume_min_voxels=10,
        volume_max_voxels=100_000,
        max_cc_count=5,
        surface_convexity_min=0.40,
        medsam_dice_threshold=0.80,  # strict — should fail with random mask
    )
    prob = np.ones((8, 8, 8), dtype=np.float32) * 0.95
    mask = (prob > 0.5).astype(np.float32)
    # Opposite medsam mask → Dice ≈ 0
    medsam_mask = np.zeros((8, 8, 8), dtype=np.float32)
    report = gate.evaluate("vol_medsam", prob, mask, medsam_mask=medsam_mask)
    assert "medsam_dice" in report.flags
