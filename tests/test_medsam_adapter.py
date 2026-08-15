"""Tests for hassl/utils/medsam_adapter.py"""
import numpy as np
import pytest

from hassl.utils.medsam_adapter import MedSAMAdapter


def test_medsam_adapter_init_valid_variant():
    adapter = MedSAMAdapter(variant="sam_med3d", checkpoint_dir="/tmp/medsam_test")
    assert adapter.variant == "sam_med3d"


def test_medsam_adapter_init_invalid_variant():
    with pytest.raises(ValueError, match="Unknown MedSAM variant"):
        MedSAMAdapter(variant="nonexistent_model")


def test_generate_bbox_prompt_from_mask():
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[10:20, 10:20, 10:20] = 1.0
    bbox_min, bbox_max = MedSAMAdapter.generate_bbox_prompt(rough_mask=mask, margin=2)
    assert bbox_min.shape == (3,)
    assert bbox_max.shape == (3,)
    # With margin=2 the bounding box should expand 2 voxels beyond the mask
    assert bbox_min[0] <= 10
    assert bbox_max[0] >= 19


def test_generate_bbox_prompt_from_volume_shape():
    bbox_min, bbox_max = MedSAMAdapter.generate_bbox_prompt(volume_shape=(32, 32, 32))
    assert all(bbox_min < bbox_max)
    # Should be central third
    assert bbox_min[0] == 8
    assert bbox_max[0] == 24


def test_infer_volume_returns_dummy_mask_when_no_checkpoint(tmp_path):
    """Without a real checkpoint, adapter should gracefully return a zero mask."""
    adapter = MedSAMAdapter(
        variant="sam_med3d",
        checkpoint_dir=str(tmp_path / "nonexistent"),
    )
    volume = np.random.rand(16, 16, 16).astype(np.float32)
    mask = adapter.infer_volume(volume)
    assert mask.shape == (16, 16, 16)
    assert mask.dtype == np.float32
    # Should be all zeros (graceful fallback)
    assert mask.sum() == 0.0


def test_get_dice_vs_returns_float(tmp_path):
    """Dice cross-check should return a float in [0, 1]."""
    adapter = MedSAMAdapter(
        variant="sam_med3d",
        checkpoint_dir=str(tmp_path / "nonexistent"),
    )
    hassl_mask = np.ones((16, 16, 16), dtype=np.float32)
    volume = np.random.rand(16, 16, 16).astype(np.float32)
    dice = adapter.get_dice_vs(hassl_mask, volume)
    assert isinstance(dice, float)
    assert 0.0 <= dice <= 1.0


def test_sha256_hash_is_deterministic():
    mask = np.ones((8, 8, 8), dtype=np.float32)
    h1 = MedSAMAdapter.compute_sha256("vol_001", mask)
    h2 = MedSAMAdapter.compute_sha256("vol_001", mask)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_hash_differs_for_different_volume_ids():
    mask = np.ones((8, 8, 8), dtype=np.float32)
    h1 = MedSAMAdapter.compute_sha256("vol_001", mask)
    h2 = MedSAMAdapter.compute_sha256("vol_002", mask)
    assert h1 != h2
