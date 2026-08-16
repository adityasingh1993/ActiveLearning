import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Compute per-class thresholds normalized by max learning status.

        Binary segmentation is special: confidence is two-sided. A foreground
        probability of 0.99 and a background probability of 0.01 are both 99%
        confident. Track binary confidence symmetrically so diagnostics do not
        silently count only foreground-confidence.
        """
        with torch.no_grad():
            if self.num_classes == 1:
                pred = predictions.detach()
                confidence = torch.maximum(pred, 1.0 - pred)
                high_conf = (confidence > self.threshold_base).float().sum().to(self.sigma.device)
                self.sigma[0] += high_conf
                return torch.tensor([self.threshold_base], device=predictions.device)
            else:
                pred_class = predictions.argmax(dim=1)
                pred_prob = predictions.max(dim=1).values

                for c in range(self.num_classes):
                    mask = (pred_class == c) & (pred_prob > self.threshold_base)
                    self.sigma[c] += mask.float().sum().to(self.sigma.device)

                # Normalize per-class counts by max count across all classes (FlexMatch paper)
                max_sigma = max(1.0, float(self.sigma.max().item()))
                beta = self.sigma / max_sigma
                thresholds = self.threshold_base * beta
                return thresholds.to(predictions.device)


class UncertaintyMaskedLoss(nn.Module):
    """Applies a voxel-wise confidence mask to pseudo-label consistency loss.

    Uses BCE-with-logits (reduction='none') as the per-voxel loss so that the
    spatial mask can be applied before averaging. Dice-style losses inherently
    reduce spatially and therefore cannot implement a true voxel-wise mask.

    Important binary safeguard:
    the existing CPS caller historically constructs ``mask = probs > threshold``.
    For a one-channel sigmoid this is foreground-only and drops confident
    background voxels, which can drive an all-foreground collapse. For binary
    predictions we therefore rebuild the mask symmetrically from the receiving
    network's probability: ``max(p, 1-p) > threshold``. This keeps both confident
    foreground and confident background in the consistency objective.
    """

    def __init__(self, base_loss: nn.Module, binary_confidence_threshold: float = 0.95):
        super().__init__()
        self.base_loss = base_loss  # kept for API compatibility, not used in forward
        self.binary_confidence_threshold = binary_confidence_threshold
        self._bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked BCE loss over confident voxels only.

        Args:
            pred:   Logit predictions [B, C, D, H, W] (NOT activated).
            target: Pseudo-labels, binary for C=1 or one-hot/broadcastable for C>1.
            mask:   Caller-provided confidence mask. For binary predictions this
                    is replaced with a symmetric confidence mask to avoid the
                    foreground-only selection bug in the CPS caller.
        """
        if pred.ndim < 3:
            raise ValueError(f"Expected segmentation logits with >=3 dims, got shape {tuple(pred.shape)}")

        if pred.shape[1] == 1:
            with torch.no_grad():
                probs = torch.sigmoid(pred.detach())
                confidence = torch.maximum(probs, 1.0 - probs)
                mask = (confidence > self.binary_confidence_threshold).to(dtype=pred.dtype)
        elif mask.shape != pred.shape:
            mask = mask.expand_as(pred)

        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)

        bce_per_voxel = self._bce(pred, target.float())
        masked_loss = bce_per_voxel * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        return pred.sum() * 0.0


class BoundaryLoss(nn.Module):
    """Distance-transform weighted Dice loss operating on probabilities.

    Network outputs are logits. Using raw logits inside a Dice ratio makes the
    intersection/union unbounded and can produce invalid gradients, especially
    early in training when logits are negative. Convert logits to probabilities
    before the weighted Dice calculation.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def compute_edt(self, target: np.ndarray) -> np.ndarray:
        """Compute Euclidean Distance Transform weights."""
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
        if self.num_classes == 1:
            probs = torch.sigmoid(pred)
            target_prob = target.float()
            if target_prob.ndim == probs.ndim - 1:
                target_prob = target_prob.unsqueeze(1)
        else:
            probs = torch.softmax(pred, dim=1)
            if target.ndim == pred.ndim and target.shape[1] == self.num_classes:
                target_prob = target.float()
            else:
                target_idx = target.long()
                if target_idx.ndim == pred.ndim and target_idx.shape[1] == 1:
                    target_idx = target_idx.squeeze(1)
                target_prob = F.one_hot(target_idx, num_classes=self.num_classes)
                target_prob = target_prob.movedim(-1, 1).float()

        # A completely empty GT mask provides no meaningful boundary target.
        if target_prob.sum() == 0:
            return pred.sum() * 0.0

        with torch.no_grad():
            edt = self.compute_edt(target_prob.detach().cpu().numpy())
            edt_weights = torch.from_numpy(edt).to(device=pred.device, dtype=probs.dtype)
            edt_weights = 1.0 + edt_weights / (edt_weights.max() + 1e-5)

        dims = list(range(2, probs.ndim))
        intersection = (probs * target_prob * edt_weights).sum(dim=dims)
        union = (probs * edt_weights).sum(dim=dims) + (target_prob * edt_weights).sum(dim=dims)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Configurable segmentation loss for supervised training.

    ``GeneralizedDiceFocalLoss`` is the default for the highly imbalanced 3D
    foreground task. For a single sigmoid output there is no explicit background
    channel; false-positive background voxels are penalized by the focal/CE term,
    while multi-class targets are converted to one-hot for MONAI losses.
    """

    def __init__(
        self,
        num_classes: int,
        loss_type: str = "generalized_dice_focal",
        include_boundary: bool = False,
        boundary_weight: float = 0.5,
        reduction: str = 'mean',
        lambda_gdl: float = 1.0,
        lambda_focal: float = 0.5,
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
        to_onehot_y = num_classes > 1

        # With one sigmoid output MONAI has no separate background channel, so
        # include_background has no practical effect. Keep True for binary and
        # exclude background only in genuine multi-class softmax segmentation.
        include_background = (num_classes == 1)

        monai_reduction = reduction if reduction in ('mean', 'sum') else 'mean'

        if loss_type == "generalized_dice_focal":
            self.dice_ce = GeneralizedDiceFocalLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                reduction=monai_reduction,
                lambda_gdl=lambda_gdl,
                lambda_focal=lambda_focal,
                gamma=gamma,
            )
        else:
            self.dice_ce = DiceCELoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
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
