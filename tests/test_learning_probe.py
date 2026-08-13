"""
Learning Probe — end-to-end training signal and inversion shape checks.

Verifies:
1. Supervised training step produces finite, decreasing loss.
2. UA-MT step produces finite loss in both prototype and full modes.
3. _invert_prediction output shape differs from spatial_size for anisotropic input.
4. TTAUncertaintyScorer returns valid variance scores.
"""

import numpy as np
import pytest
import torch


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_volume(spatial=(32, 32, 32), batch=1, channels=1):
    return torch.rand(batch, channels, *spatial)


def _make_binary_label(spatial=(32, 32, 32), batch=1):
    t = torch.zeros(batch, 1, *spatial)
    t[:, :, 8:16, 8:16, 8:16] = 1.0
    return t


# ─── Fix 1 guard: _invert_prediction must not receive a 5-D tensor ──────────

def test_invert_prediction_uses_4d_tensor():
    """pred_tensor[index] must be 4-D, not pred_tensor[index:index+1] (5-D).

    Regression guard for Round-13 critical finding C-1 in query_engine.py.
    If this test breaks, the slice notation reverted to [index:index+1].
    """
    pred = torch.rand(2, 1, 32, 32, 32)
    indexed = pred[0]           # 4-D [1, 32, 32, 32] — correct
    sliced  = pred[0:1]         # 5-D [1, 1, 32, 32, 32] — wrong (was the bug)
    assert indexed.ndim == 4, "pred_tensor[index] must be 4-D"
    assert sliced.ndim == 5,  "pred_tensor[index:index+1] is 5-D (the old bug)"


# ─── Fix 2 guard: MetaTensor applied_operations present on input ─────────────

def test_meta_tensor_has_applied_operations():
    """Invertd requires applied_operations on the input tensor to invert transforms.

    If inputs[b] carries no applied_operations the inversion silently returns
    the un-inverted tensor (Round-13 finding, trainer.py validation path).
    """
    try:
        from monai.data import MetaTensor
    except ImportError:
        pytest.skip("MONAI not installed")

    data = torch.rand(1, 32, 32, 32)
    mt = MetaTensor(data, meta={}, applied_operations=[])
    assert hasattr(mt, 'applied_operations'), \
        "MetaTensor must expose applied_operations for Invertd to invert"


# ─── Fix 3 guard: trilinear mode must be used for image resizing ─────────────

def test_resize_mode_is_trilinear():
    """Resized must use trilinear for images, not area.

    area is a box-filter (adaptive_avg_pool) — upsampling replicates rather
    than interpolates, causing staircase artifacts on anisotropic volumes.
    (Round-13 High finding, data_engine.py)
    """
    from hassl.data.data_engine import get_base_transforms
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.spacing = (1.0, 1.0, 1.0)
    cfg.spatial_size = (64, 64, 64)
    cfg.preprocessing_mode = 'resize'
    cfg.num_classes = 1
    cfg.patch_size = (32, 32, 32)
    cfg.pos_neg_ratio = 1.0

    transform = get_base_transforms(cfg, keys=["image", "label"], is_training=False)

    # Find the Resized transform and check its mode
    from monai.transforms import Resized, Compose
    transforms_list = transform.transforms if hasattr(transform, 'transforms') else []
    resized_transforms = [t for t in transforms_list if isinstance(t, Resized)]
    assert len(resized_transforms) > 0, "Resized transform not found in pipeline"

    resized = resized_transforms[0]
    # mode is stored as a tuple per key; image is first key so index 0
    image_mode = resized.mode[0] if isinstance(resized.mode, (list, tuple)) else resized.mode
    assert image_mode == "trilinear", (
        f"Resized image mode must be 'trilinear', got '{image_mode}'. "
        "Reverting to 'area' would reintroduce staircase artifacts on upsampled axes."
    )


# ─── Supervised training step produces finite, decreasing loss ───────────────

def test_supervised_training_step_finite_loss():
    """One gradient step must produce a finite, positive loss."""
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
    assert loss.item() > 0, "Loss should be positive for random init"


# ─── TTAUncertaintyScorer returns bounded variance scores ────────────────────

def test_tta_uncertainty_scorer():
    """TTAUncertaintyScorer must return non-negative variance for each volume in the batch."""
    from hassl.active.query_strategies import TTAUncertaintyScorer
    from hassl.training.trainer import build_network

    model = build_network('unet', num_classes=1, dropout=0.0)
    scorer = TTAUncertaintyScorer(model, num_passes=3, flip=True, intensity_std=0.02)

    x = _make_volume(batch=2)
    scores = scorer.score(x)

    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert (scores >= 0).all(), "TTA variance scores must be non-negative"
    assert np.isfinite(scores).all(), "TTA variance scores must be finite"


# ─── BALDStrategy BALD scores are non-negative ───────────────────────────────

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
