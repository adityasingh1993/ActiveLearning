import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
from monai.losses import DiceCELoss, GeneralizedDiceFocalLoss


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
    """Applies a voxel-wise confidence mask to suppress uncertain pseudo-label regions.

    Uses BCE-with-logits (reduction='none') as the per-voxel loss so that the spatial
    mask can be applied before averaging.  DiceLoss inherently reduces to a scalar per
    volume, so it cannot be masked per-voxel; CE is a strictly per-voxel signal and
    sufficient for the unsupervised consistency term.

    The supervised loss (clean GT labels) continues using the full DiceCELoss / GeneralizedDiceFocalLoss — no
    masking is needed there because ground-truth masks are reliable.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss  # kept for API compatibility, not used in forward
        self._bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked BCE loss over confident voxels only.

        Args:
            pred:   Logit predictions  [B, C, D, H, W]  (NOT sigmoid-activated)
            target: Binary pseudo-labels [B, C, D, H, W] float {0, 1}
            mask:   Confidence mask      [B, C, D, H, W] or broadcastable; 1=confident 0=uncertain

        Returns:
            Scalar loss averaged over confident voxels only.
            Returns 0 when no confident voxels exist (avoids NaN on fully-masked batches).
        """
        # Expand mask to match pred if needed (e.g. mask is [B, 1, D, H, W])
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)

        bce_per_voxel = self._bce(pred, target.float())
        masked_loss = bce_per_voxel * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


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
        # Guard against zero-foreground or all-background predictions/targets
        if pred.sum() == 0 or target.sum() == 0:
            return torch.tensor(1.0, device=pred.device, requires_grad=True)

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
    """Configurable segmentation loss for supervised training (Generalized Dice Focal Loss or DiceCE).

    Fixes applied:
    - include_background=False always: background Dice is trivially high (~0.97) for
      sparse foreground structures (bladder 2-5% of volume), drowning the gradient
      signal for the actual target class.
    - to_onehot_y=True for multiclass: MONAI DiceLoss requires one-hot encoded targets.
      Without this, integer class indices (e.g. 2) are treated as probability activations
      (2.0), making Dice computation mathematically invalid for multi-class tasks.
    - Default loss_type="generalized_dice_focal": combines Generalized Dice Loss (class weighting)
      with Focal Loss for improved convergence on difficult boundaries and high class imbalance.
    - lambda_focal=0.25 & lambda_gdl=1.0: rebalances Focal Loss vs GDL. In 3D medical volumes with
      99% background, unweighted Focal Loss (lambda_focal=1.0) accumulates massive background
      gradients that push all predictions to 0 (pred_fg = 0.00001). Lowering lambda_focal allows GDL
      to pull predictions into foreground regions.
    """

    def __init__(
        self,
        num_classes: int,
        loss_type: str = "generalized_dice_focal",
        include_boundary: bool = False,
        boundary_weight: float = 0.5,
        reduction: str = 'mean',
        lambda_gdl: float = 1.0,
        lambda_focal: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.include_boundary = include_boundary
        self.boundary_weight = boundary_weight
        self.lambda_gdl = lambda_gdl
        self.lambda_focal = lambda_focal
        self.gamma = gamma

        sigmoid = num_classes == 1
        softmax = num_classes > 1
        to_onehot_y = num_classes > 1  # Required for multiclass: converts int targets to one-hot

        # MONAI loss reduction accepts 'mean' or 'sum'
        monai_reduction = reduction if reduction in ('mean', 'sum') else 'mean'

        if loss_type == "generalized_dice_focal":
            self.dice_ce = GeneralizedDiceFocalLoss(
                include_background=False,  # Always exclude background — foreground Dice is what matters
                to_onehot_y=to_onehot_y,   # Required for multiclass integer-label targets
                sigmoid=sigmoid,
                softmax=softmax,
                reduction=monai_reduction,
                lambda_gdl=lambda_gdl,
                lambda_focal=lambda_focal,
                gamma=gamma,
            )
        else:
            self.dice_ce = DiceCELoss(
                include_background=False,  # Always exclude background — foreground Dice is what matters
                to_onehot_y=to_onehot_y,   # Required for multiclass integer-label targets
                sigmoid=sigmoid,
                softmax=softmax,
                reduction=monai_reduction,
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
