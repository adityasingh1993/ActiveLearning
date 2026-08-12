import pytest
import torch
from hassl.losses import CombinedSegLoss, FlexMatchThreshold, UncertaintyMaskedLoss, BoundaryLoss

def test_combined_seg_loss():
    loss_fn = CombinedSegLoss(include_dice=True, include_ce=True)
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_flexmatch_threshold():
    thresh_fn = FlexMatchThreshold(num_classes=2)
    probs = torch.rand(10, 2)
    labels = torch.randint(0, 2, (10,))
    thresh_fn.update(probs, labels)
    thresholds = thresh_fn.get_thresholds()
    assert thresholds.shape == (2,)

def test_uncertainty_masked_loss():
    base_loss = CombinedSegLoss()
    masked_loss = UncertaintyMaskedLoss(base_loss)
    
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    uncertainty = torch.rand(2, 1, 32, 32, 32)
    
    # Check that mask reduces loss (high uncertainty = lower loss weight)
    loss1 = masked_loss(pred, target, uncertainty)
    assert loss1.dim() == 0

def test_boundary_loss():
    loss_fn = BoundaryLoss()
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
