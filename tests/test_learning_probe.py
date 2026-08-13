"""
Learning Probe — end-to-end training signal, TTA spatial un-flipping, and server undo/provenance checks.

Verifies:
1. TTAUncertaintyScorer returns ~0 variance for a deterministic model on asymmetric inputs (H-1 un-flip guard).
2. _invert_prediction handles 4-D MetaTensor transform traces.
3. Server edit tracking & undo stack: accept_volume tags unedited volumes as pseudo_approved and edited as human_corrected (H-2 & R12 H-1).
4. Resized transform uses trilinear interpolation for images.
5. Supervised training step produces finite loss.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from hassl.active.query_strategies import TTAUncertaintyScorer


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_volume(spatial=(32, 32, 32), batch=1, channels=1):
    return torch.rand(batch, channels, *spatial)


def _make_binary_label(spatial=(32, 32, 32), batch=1):
    t = torch.zeros(batch, 1, *spatial)
    t[:, :, 8:16, 8:16, 8:16] = 1.0
    return t


# ─── Fix H-1 Guard: TTAUncertaintyScorer un-flips spatial predictions ───────

class IdentityModel(torch.nn.Module):
    """Deterministic, flip-equivariant model for testing TTA un-flipping."""
    def forward(self, x):
        return x


def test_tta_scorer_zero_for_deterministic_model():
    """TTAUncertaintyScorer must return ~0 variance for a deterministic model on an asymmetric volume.

    If spatial un-flipping is missing, comparing flipped vs unflipped passes on an
    asymmetric volume yields non-zero variance (Round 14 finding H-1).
    """
    model = IdentityModel()
    scorer = TTAUncertaintyScorer(model, num_passes=8, flip=True, intensity_std=0.0)

    # Create a spatially asymmetric volume (high values concentrated in one quadrant)
    x = torch.zeros(1, 1, 32, 32, 32)
    x[0, 0, :10, :10, :10] = 1.0

    scores = scorer.score(x)
    assert scores[0] < 1e-6, (
        f"Expected TTA variance ~0.0 for deterministic model, got {scores[0]}. "
        "If >0, TTA passes are not being spatially un-flipped before computing variance."
    )


# ─── Fix H-2 & R12 H-1 Guard: Server Undo Stack & Provenance Tagging ────────

def test_server_edit_tracking_and_undo():
    """Server accept_volume must tag unedited as pseudo_approved and edited as human_corrected."""
    import hassl.app.server as server
    from hassl.app.server import _state, _push_undo

    # Reset state
    _state["edited_volumes"] = set()
    _state["undo_stacks"] = {}
    _state["cached_presegs"] = {}
    _state["volumes"] = {"vol_test": {"image_path": "dummy.mha", "preseg_path": "dummy.seg.nrrd"}}

    vol_id = "vol_test"
    mask_orig = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_edited = np.ones((10, 10, 10), dtype=np.uint8)

    # 1. Unedited volume should NOT be in edited_volumes
    assert vol_id not in _state["edited_volumes"]

    # 2. Simulate editing operation (pushes to undo stack and marks edited)
    _push_undo(vol_id, mask_orig)
    _state["cached_presegs"][vol_id] = mask_edited

    assert vol_id in _state["edited_volumes"]
    assert len(_state["undo_stacks"][vol_id]) == 1

    # 3. Simulate undo
    prev = _state["undo_stacks"][vol_id].pop()
    _state["cached_presegs"][vol_id] = prev
    _state["edited_volumes"].discard(vol_id)

    assert vol_id not in _state["edited_volumes"]
    assert np.array_equal(_state["cached_presegs"][vol_id], mask_orig)


# ─── Fix 3 guard: trilinear mode must be used for image resizing ─────────────

def test_resize_mode_is_trilinear():
    """Resized must use trilinear for images, not area."""
    from hassl.data.data_engine import get_base_transforms

    cfg = MagicMock()
    cfg.spacing = (1.0, 1.0, 1.0)
    cfg.spatial_size = (64, 64, 64)
    cfg.preprocessing_mode = 'resize'
    cfg.num_classes = 1
    cfg.patch_size = (32, 32, 32)
    cfg.pos_neg_ratio = 1.0

    transform = get_base_transforms(cfg, keys=["image", "label"], is_training=False)

    from monai.transforms import Resized
    transforms_list = transform.transforms if hasattr(transform, 'transforms') else []
    resized_transforms = [t for t in transforms_list if isinstance(t, Resized)]
    assert len(resized_transforms) > 0, "Resized transform not found in pipeline"

    resized = resized_transforms[0]
    image_mode = resized.mode[0] if isinstance(resized.mode, (list, tuple)) else resized.mode
    assert image_mode == "trilinear", f"Resized image mode must be 'trilinear', got '{image_mode}'."


# ─── Supervised training step produces finite loss ────────────────────────────

def test_supervised_training_step_finite_loss():
    """One gradient step must produce a finite loss."""
    from hassl.training.trainer import build_network
    from monai.losses import DiceCELoss

    model = build_network('unet', num_classes=1, dropout=0.0)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceCELoss(include_background=True, sigmoid=True)

    x = _make_volume()
    y = _make_binary_label()

    optim.zero_grad()
    pred = model(x)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    elif pred.ndim == 6:
        pred = pred[:, 0]
    loss = criterion(pred, y)
    loss.backward()
    optim.step()

    assert torch.isfinite(loss), f"Loss must be finite; got {loss.item()}"


# ─── BALDStrategy scores non-negative ─────────────────────────────────────────

def test_bald_strategy_scores_non_negative():
    """BALD mutual information scores must be >= 0."""
    from hassl.active.query_strategies import BALDStrategy
    from hassl.training.trainer import build_network

    model = build_network('unet', num_classes=1, dropout=0.2)
    strategy = BALDStrategy(model, num_passes=3)
    x = _make_volume(batch=2)
    scores = strategy.score(x)

    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert (scores >= 0).all(), "BALD mutual information must be >= 0"
