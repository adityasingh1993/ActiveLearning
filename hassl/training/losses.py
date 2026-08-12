import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
from monai.losses import DiceCELoss


class FlexMatchThreshold(nn.Module):
    """Dynamic per-class confidence thresholding (FlexMatch, M-7 fix)."""

    def __init__(self, num_classes: int, threshold_base: float = 0.95):
        super().__init__()
        self.num_classes = num_classes
        self.threshold_base = threshold_base
        self.register_buffer("sigma", torch.zeros(num_classes))

    def get_threshold(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute per-class thresholds normalized by max learning status (M-7 fix)."""
        with torch.no_grad():
            if self.num_classes == 1:
                pred = predictions.detach()
                high_conf = (pred > self.threshold_base).float().sum()
                self.sigma[0] += high_conf
                beta = 1.0  # Single class normalized
                threshold = self.threshold_base * beta
                return torch.tensor([threshold], device=predictions.device)
            else:
                pred_class = predictions.argmax(dim=1)
                pred_prob = predictions.max(dim=1).values

                for c in range(self.num_classes):
                    mask = (pred_class == c) & (pred_prob > self.threshold_base)
                    self.sigma[c] += mask.float().sum()

                # Normalize per-class counts by max count across all classes (FlexMatch paper)
                max_sigma = max(1.0, float(self.sigma.max().item()))
                beta = self.sigma / max_sigma
                thresholds = self.threshold_base * beta
                return thresholds.to(predictions.device)


class UncertaintyMaskedLoss(nn.Module):
    """Wraps base loss and applies a binary mask suppressing high-uncertainty voxels."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base_loss_per_voxel = self.base_loss(pred, target)

        if mask.shape != base_loss_per_voxel.shape:
            mask = mask.view_as(base_loss_per_voxel)

        masked_loss = base_loss_per_voxel * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        return torch.tensor(0.0, device=pred.device)


class BoundaryLoss(nn.Module):
    """Distance-transform weighted Dice loss (M-8 fix)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def compute_edt(self, target: np.ndarray) -> np.ndarray:
        """Compute Euclidean Distance Transform weights (M-8 fix)."""
        res = np.zeros_like(target, dtype=np.float32)
        if target.ndim == 5:
            # Shape [B, C, D, H, W]
            for b in range(target.shape[0]):
                for c in range(target.shape[1]):
                    t = target[b, c]
                    if t.any() and not t.all():
                        edt_fg = ndimage.distance_transform_edt(t)
                        edt_bg = ndimage.distance_transform_edt(1 - t)
                        res[b, c] = edt_bg + edt_fg
                    else:
                        res[b, c] = 1.0
        elif target.ndim == 4:
            # Shape [B, D, H, W]
            for b in range(target.shape[0]):
                t = target[b]
                if t.any() and not t.all():
                    edt_fg = ndimage.distance_transform_edt(t)
                    edt_bg = ndimage.distance_transform_edt(1 - t)
                    res[b] = edt_bg + edt_fg
                else:
                    res[b] = 1.0
        return res

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            edt = self.compute_edt(target.cpu().numpy())
            edt_weights = torch.from_numpy(edt).to(pred.device)
            edt_weights = 1.0 + edt_weights / (edt_weights.max() + 1e-5)

        dims = list(range(2, pred.ndim))
        intersection = (pred * target * edt_weights).sum(dim=dims)
        union = (pred * edt_weights).sum(dim=dims) + (target * edt_weights).sum(dim=dims)
        dice = 2.0 * intersection / (union + 1e-5)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes: int, include_boundary: bool = False, boundary_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.include_boundary = include_boundary
        self.boundary_weight = boundary_weight

        sigmoid = num_classes == 1
        softmax = num_classes > 1

        self.dice_ce = DiceCELoss(
            include_background=False if num_classes > 1 else True,
            sigmoid=sigmoid, softmax=softmax,
            reduction='none'
        )

        if include_boundary:
            self.boundary_loss = BoundaryLoss(num_classes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_ce(pred, target)
        if loss.ndim > 0:
            loss = loss.mean()
        if self.include_boundary:
            loss = loss + self.boundary_weight * self.boundary_loss(pred, target)
        return loss
