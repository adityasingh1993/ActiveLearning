import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
from monai.losses import DiceLoss, DiceCELoss

class FlexMatchThreshold:
    """Dynamic per-class confidence thresholding (FlexMatch)."""
    def __init__(self, num_classes: int, threshold_base: float = 0.95):
        self.num_classes = num_classes
        self.threshold_base = threshold_base
        self.register = torch.zeros(num_classes)
        self.max_register = torch.ones(num_classes) * 1e-5

    def get_threshold(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        predictions: probabilities of shape (B, C, ...) or (B, 1, ...)
        Returns per-class thresholds
        """
        if self.num_classes == 1:
            pred = predictions.detach().flatten()
            high_conf = (pred > self.threshold_base).sum()
            self.register[0] += high_conf.item()
            self.max_register[0] = max(self.max_register[0].item(), self.register[0].item())
            
            beta = self.register[0] / self.max_register[0]
            threshold = self.threshold_base * beta
            return threshold.view(1).to(predictions.device)
        else:
            C = self.num_classes
            pred_class = predictions.argmax(dim=1)
            pred_prob = predictions.max(dim=1).values
            
            for c in range(C):
                mask = (pred_class == c) & (pred_prob > self.threshold_base)
                self.register[c] += mask.sum().item()
                self.max_register[c] = max(self.max_register[c].item(), self.register[c].item())
                
            beta = self.register / self.max_register
            thresholds = self.threshold_base * beta
            return thresholds.to(predictions.device)

class UncertaintyMaskedLoss(nn.Module):
    """Wraps any base loss, applies a binary mask suppressing high-uncertainty voxels."""
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base_loss_per_voxel = self.base_loss(pred, target)
        
        # Make sure mask matches shape
        if mask.shape != base_loss_per_voxel.shape:
            mask = mask.view_as(base_loss_per_voxel)
            
        masked_loss = base_loss_per_voxel * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        return masked_loss.sum() * 0.0

class BoundaryLoss(nn.Module):
    """Distance-transform weighted Dice loss."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def compute_edt(self, target: np.ndarray) -> np.ndarray:
        res = np.zeros_like(target, dtype=np.float32)
        for i in range(target.shape[0]):
            for c in range(target.shape[1] if len(target.shape) > 4 else 1):
                t = target[i, c] if len(target.shape) > 4 else target[i]
                if t.any() and not t.all():
                    edt_fg = ndimage.distance_transform_edt(t)
                    edt_bg = ndimage.distance_transform_edt(1 - t)
                    res[i, c if len(target.shape) > 4 else i] = edt_bg + edt_fg
                else:
                    res[i, c if len(target.shape) > 4 else i] = 1.0
        return res
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            edt = self.compute_edt(target.cpu().numpy())
            edt_weights = torch.from_numpy(edt).to(pred.device)
            edt_weights = 1.0 + edt_weights / (edt_weights.max() + 1e-5)
            
        intersection = (pred * target * edt_weights).sum(dim=list(range(2, pred.ndim)))
        union = (pred * edt_weights).sum(dim=list(range(2, pred.ndim))) + (target * edt_weights).sum(dim=list(range(2, pred.ndim)))
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
        
        self.dice_ce = DiceCELoss(include_background=False if num_classes>1 else True,
                                  sigmoid=sigmoid, softmax=softmax, 
                                  reduction='none')
        
        if include_boundary:
            self.boundary_loss = BoundaryLoss(num_classes)
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_ce(pred, target)
        if self.include_boundary:
            # We might want to just mean the dice_ce first, then add boundary
            if loss.ndim > 0:
                loss = loss.mean()
            loss = loss + self.boundary_weight * self.boundary_loss(pred, target)
        return loss
