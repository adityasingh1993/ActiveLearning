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


def test_flexmatch_threshold_binary_counts_two_sided_confidence():
    thresh_fn = FlexMatchThreshold(num_classes=1, threshold_base=0.95)
    probs = torch.tensor([[[[[0.99, 0.01, 0.50]]]]])
    thresh = thresh_fn.get_threshold(probs)
    assert thresh.shape == (1,)
    assert thresh.item() == pytest.approx(0.95)
    assert thresh_fn.sigma[0].item() == pytest.approx(2.0)


def test_uncertainty_masked_loss():
    base_loss = CombinedSegLoss(num_classes=1)
    masked_loss = UncertaintyMaskedLoss(base_loss)

    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    mask = (torch.rand(2, 1, 32, 32, 32) > 0.5).float()

    loss1 = masked_loss(pred, target, mask)
    assert loss1.dim() == 0
    assert loss1.item() >= 0


def test_uncertainty_masked_loss_preserves_caller_mask():
    base_loss = CombinedSegLoss(num_classes=1)
    masked_loss = UncertaintyMaskedLoss(base_loss)

    pred = torch.tensor([[[[[4.0, -4.0, 0.0]]]]], requires_grad=True)
    target = torch.tensor([[[[[1.0, 0.0, 1.0]]]]])
    # Only the first voxel is selected by the caller. The loss must not replace
    # this mask based on its own confidence calculation.
    caller_mask = torch.tensor([[[[[1.0, 0.0, 0.0]]]]])

    loss = masked_loss(pred, target, caller_mask)
    loss.backward()

    assert pred.grad is not None
    assert abs(pred.grad[0, 0, 0, 0, 0].item()) > 0
    assert pred.grad[0, 0, 0, 0, 1].item() == pytest.approx(0.0)
    assert pred.grad[0, 0, 0, 0, 2].item() == pytest.approx(0.0)


def test_boundary_loss_uses_probabilities_and_stays_bounded():
    loss_fn = BoundaryLoss(num_classes=1)
    pred = torch.tensor([[[[[-12.0, 12.0], [8.0, -8.0]]]]], requires_grad=True)
    target = torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]])
    loss = loss_fn(pred, target)
    loss.backward()

    assert loss.dim() == 0
    assert 0.0 <= loss.item() <= 1.0
    assert torch.isfinite(pred.grad).all()


def test_boundary_loss_empty_target_is_zero_and_differentiable():
    loss_fn = BoundaryLoss(num_classes=1)
    pred = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert pred.grad is not None
