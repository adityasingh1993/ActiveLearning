import pytest
import torch
from hassl.training.losses import CombinedSegLoss, FlexMatchThreshold, UncertaintyMaskedLoss, BoundaryLoss

def test_combined_seg_loss_generalized_dice_focal():
    loss_fn = CombinedSegLoss(num_classes=1, loss_type="generalized_dice_focal")
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_combined_seg_loss_dice_ce():
    loss_fn = CombinedSegLoss(num_classes=1, loss_type="dice_ce")
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_flexmatch_threshold():
    thresh_fn = FlexMatchThreshold(num_classes=2, threshold_base=0.8)
    probs = torch.rand(10, 2, 16, 16, 16)
    thresholds = thresh_fn.get_threshold(probs)
    assert thresholds.shape == (2,)

def test_uncertainty_masked_loss():
    base_loss = CombinedSegLoss(num_classes=1)
    masked_loss = UncertaintyMaskedLoss(base_loss)
    
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    mask = (torch.rand(2, 1, 32, 32, 32) > 0.5).float()
    
    loss1 = masked_loss(pred, target, mask)
    assert loss1.dim() == 0
    assert loss1.item() >= 0

def test_boundary_loss():
    loss_fn = BoundaryLoss(num_classes=1)
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_soft_cldice_loss():
    from hassl.training.losses import SoftCLDiceLoss
    cldice_fn = SoftCLDiceLoss(num_classes=1)
    pred = torch.rand(2, 1, 16, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
    loss = cldice_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_combined_seg_loss_with_cldice():
    loss_fn = CombinedSegLoss(num_classes=1, include_cldice=True, cldice_weight=0.3)
    pred = torch.rand(2, 1, 16, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
