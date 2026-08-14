import copy
import logging
import os
import warnings
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from hassl.compat import (
    MONAI_GE_15,
    build_invertd,
    build_hd95_metric,
    build_confusion_metric,
)

# Suppress MONAI 1.5+ internal deprecation warnings that are non-actionable for users
if MONAI_GE_15:
    warnings.filterwarnings("ignore", message=".*always_return_as_numpy.*")

logger = logging.getLogger(__name__)
import torch.nn as nn
import monai
from monai.networks.nets import UNet, DynUNet, SwinUNETR
from monai.metrics import DiceMetric

from .ema import EMATeacher
from .losses import CombinedSegLoss, UncertaintyMaskedLoss, FlexMatchThreshold
from ..tracking import ExperimentTracker


def build_network(backbone: str, num_classes: int, dropout: float) -> nn.Module:
    """Build network architecture based on backbone selection."""
    if backbone == 'unet':
        return UNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            num_res_units=2, dropout=dropout
        )
    elif backbone == 'dynunet':
        return DynUNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes,
            kernel_size=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 4,
            filters=[16, 32, 64, 128, 256],
            dropout=dropout, norm_name='instance',
            deep_supervision=True
        )
    elif backbone == 'swinunetr':
        return SwinUNETR(
            in_channels=1, out_channels=num_classes,
            feature_size=48, use_checkpoint=True
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def compute_multiscale_loss(criterion, preds, target):
    """Compute deep-supervision loss if network outputs multi-scale heads (H-8 fix)."""
    import torch.nn.functional as F

    if isinstance(preds, (list, tuple)):
        # Handle list of multi-scale outputs
        loss = 0.0
        weights = [1.0 / (2 ** i) for i in range(len(preds))]
        total_w = sum(weights)
        for p, w in zip(preds, weights):
            t_down = F.interpolate(target.float(), size=p.shape[2:], mode='nearest') if p.shape[2:] != target.shape[2:] else target
            l = criterion(p, t_down)
            if l.ndim > 0:
                l = l.mean()
            loss += (w / total_w) * l
        return loss
    elif torch.is_tensor(preds) and preds.ndim == 6:
        # Handle 6D tensor [B, heads, C, D, H, W]
        loss = 0.0
        num_heads = preds.shape[1]
        weights = [1.0 / (2 ** i) for i in range(num_heads)]
        total_w = sum(weights)
        for i in range(num_heads):
            p = preds[:, i]
            t_down = F.interpolate(target.float(), size=p.shape[2:], mode='nearest') if p.shape[2:] != target.shape[2:] else target
            l = criterion(p, t_down)
            if l.ndim > 0:
                l = l.mean()
            loss += (weights[i] / total_w) * l
        return loss
    else:
        l = criterion(preds, target)
        return l.mean() if l.ndim > 0 else l


def apply_keep_largest_cc(pred_tensor: torch.Tensor, min_size_voxels: int = 100) -> torch.Tensor:
    """Keep only the largest CC and remove satellite predictions smaller than min_size_voxels.

    Args:
        pred_tensor: Binary prediction tensor [B, C, D, H, W].
        min_size_voxels: Any connected component with fewer voxels than this is discarded,
            even if it is the only component. Default 100 is safe for medium-sized targets
            (bladder/prostate at 128^3 are 1000-10000 voxels). Set via config.lcc_min_size_voxels.

    Two-pass approach:
      Pass 1 — find and keep the largest CC.
      Pass 2 — zero out any component (including the largest) below min_size_voxels threshold.
    This removes satellite noise blobs that inflate FP rates and bias volume estimates upward.
    """
    import scipy.ndimage as ndi

    device = pred_tensor.device
    arr = pred_tensor.detach().cpu().numpy()
    out = np.zeros_like(arr)

    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            mask = arr[b, c] > 0.5
            if not mask.any():
                continue
            labeled_arr, num = ndi.label(mask)
            if num == 0:
                continue
            counts = np.bincount(labeled_arr.ravel())
            counts[0] = 0  # exclude background
            if counts.max() == 0:
                continue
            max_lab = counts.argmax()
            largest_size = counts[max_lab]
            # Keep largest CC only if it meets the minimum size threshold
            if largest_size >= min_size_voxels:
                out[b, c] = (labeled_arr == max_lab).astype(arr.dtype)
            # else: all components are too small — output stays zero (no prediction)

    return torch.from_numpy(out).to(device)


class EarlyStopping:
    """Early stopping monitor for validation metric plateaus."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if np.isnan(val_score):
            return False

        if self.best_score is None:
            self.best_score = val_score
            return False

        if self.mode == 'max':
            improved = val_score > (self.best_score + self.min_delta)
        else:
            improved = val_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = val_score
            self.counter = 0
            self.early_stop = False  # Clear sticky flag on improvement (V10-1 fix)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping monitor for a new AL round (V10-1 fix)."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def state_dict(self) -> dict:
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
        }

    def load_state_dict(self, state: dict):
        self.counter = state.get('counter', 0)
        self.best_score = state.get('best_score', None)
        self.early_stop = state.get('early_stop', False)


class HASSLTrainer:
    """Unified trainer supporting UA-Mean Teacher (prototype) and CPS (full) modes."""

    def __init__(self, config, labeled_loader, unlabeled_loader, val_loader,
                 tracker: ExperimentTracker, pretrained_weights=None, val_transform=None):
        self.config = config
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader
        self.tracker = tracker

        # V-10 fix: Set seed determinism
        monai.utils.set_determinism(seed=config.seed)

        self.device_type = 'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
        self.device = torch.device(self.device_type)

        self.num_classes = config.num_classes
        self.mode = config.compute_mode

        # AMP Scaler setup
        if self.device_type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)

        if self.mode == 'prototype':
            self.net_A = build_network(config.unet_backbone, self.num_classes, config.dropout).to(self.device)
            self.teacher = EMATeacher(self.net_A).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.net_A.parameters(), lr=config.train_lr, weight_decay=config.train_weight_decay
            )
            self.scheduler = self._build_scheduler(self.optimizer)
        else:
            self.net_A = build_network(config.unet_backbone, self.num_classes, config.dropout).to(self.device)
            self.net_B = build_network('swinunetr', self.num_classes, 0.0).to(self.device)
            self.optimizer_A = torch.optim.AdamW(
                self.net_A.parameters(), lr=config.train_lr, weight_decay=config.train_weight_decay
            )
            self.optimizer_B = torch.optim.AdamW(
                self.net_B.parameters(), lr=config.train_lr, weight_decay=config.train_weight_decay
            )
            self.scheduler_A = self._build_scheduler(self.optimizer_A)
            self.scheduler_B = self._build_scheduler(self.optimizer_B)
            self.flex_match = FlexMatchThreshold(self.num_classes)

        use_es = getattr(config, 'use_early_stopping', True)
        patience = getattr(config, 'early_stopping_patience', 30)
        min_delta = getattr(config, 'early_stopping_min_delta', 1e-4)
        self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='max') if use_es else None

        loss_type = getattr(config, 'loss_type', 'generalized_dice_focal')
        lambda_gdl = getattr(config, 'loss_lambda_gdl', 1.0)
        lambda_focal = getattr(config, 'loss_lambda_focal', 0.25)
        gamma = getattr(config, 'loss_focal_gamma', 2.0)
        self.criterion = CombinedSegLoss(
            self.num_classes,
            loss_type=loss_type,
            include_boundary=False,
            lambda_gdl=lambda_gdl,
            lambda_focal=lambda_focal,
            gamma=gamma,
        )
        self.masked_criterion = UncertaintyMaskedLoss(self.criterion)
        self.best_dice = 0.0
        self.dice_metric = DiceMetric(include_background=False if self.num_classes > 1 else True, reduction="mean")
        self.start_epoch = 0

        # Pre-instantiate spatial and intensity augmentations for spatially aligned Mean Teacher view (V7-1, V8-1 fix)
        from ..data.augmentations import get_spatial_augmentation, get_intensity_augmentation
        self.spatial_aug = get_spatial_augmentation(keys=["image"])
        self.intensity_aug = get_intensity_augmentation(keys=["image"])

        # Store val transform for Invertd-based original-space volume calculation.
        # CRITICAL: must be the EXACT same Compose instance passed to the val DataLoader.
        # Invertd replays applied_operations by matching transform entries; a separately-
        # constructed Compose (even with identical config) has different object IDs and
        # will fail to find a matching entry → 'validate Invertd failed for sample 0'.
        # build_dataloaders() now returns val_transforms; callers should pass it here.
        if val_transform is not None:
            self.val_transform = val_transform
        else:
            # Fallback for callers that don't yet pass val_transform (backward-compatible).
            from ..data.data_engine import get_base_transforms
            self.val_transform = get_base_transforms(config, keys=["image", "label"], is_training=False)

        # Load pre-trained SSL weights if available
        if pretrained_weights and os.path.exists(pretrained_weights):
            self._load_pretrained(pretrained_weights)

    def _build_scheduler(self, optimizer):
        """Construct learning rate scheduler based on config with optional linear warmup (V10-4 fix)."""
        scheduler_type = getattr(self.config, 'lr_scheduler', 'cosine')
        min_lr = getattr(self.config, 'min_lr', 1e-6)
        train_epochs = getattr(self.config, 'train_epochs', 200)
        warmup_epochs = getattr(self.config, 'lr_warmup_epochs', 0)

        if scheduler_type == 'none' or train_epochs <= 0:
            return None

        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10, min_lr=min_lr
            )

        if scheduler_type == 'cosine':
            if warmup_epochs > 0 and warmup_epochs < train_epochs:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
                )
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, train_epochs - warmup_epochs), eta_min=min_lr
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
                )
            else:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=train_epochs, eta_min=min_lr
                )

        return None

    def _make_unlabeled_views(self, inputs_u: torch.Tensor):
        """Generate spatially aligned teacher and student views (V7-1, V8-1, V8-5 fix)."""
        spatial_img = [self.spatial_aug({"image": inputs_u[b]})["image"] for b in range(inputs_u.size(0))]
        spatial_tensor = torch.stack(spatial_img)
        inputs_u_teacher = spatial_tensor.clone()  # Decouple teacher tensor memory from student intensity transforms (V8-5 fix)
        inputs_u_student = torch.stack([self.intensity_aug({"image": spatial_tensor[b]})["image"] for b in range(spatial_tensor.size(0))])
        return inputs_u_teacher, inputs_u_student

    def get_rampup_weight(self, epoch: int) -> float:
        if epoch < self.config.consistency_rampup_epochs:
            return float(epoch / max(1, self.config.consistency_rampup_epochs))
        return 1.0

    def train_one_epoch_uamt(self, epoch: int):
        self.net_A.train()
        unsup_weight = self.get_rampup_weight(epoch)
        pseudo_weight = getattr(self.config, 'pseudo_label_weight', 0.5)
        total_loss, total_sup, total_unsup, total_uncert = 0.0, 0.0, 0.0, 0.0
        # Teacher diagnostic accumulators: confidence and foreground fraction of pseudo-labels
        total_pseudo_conf = 0.0   # Mean teacher sigmoid probability on unlabeled voxels
        total_pseudo_fg   = 0.0   # Fraction of voxels the teacher labels as foreground
        n_unlabeled_steps = 0

        if self.unlabeled_loader is None or len(self.unlabeled_loader.dataset) == 0:
            iter_unlabeled = None
        else:
            iter_unlabeled = iter(self.unlabeled_loader)

        for batch_idx, batch_data in enumerate(self.labeled_loader):
            inputs_l = batch_data['image'].to(self.device)
            targets_l = batch_data['label'].to(self.device)
            provenance_list = batch_data.get('provenance', ['human'] * inputs_l.size(0))

            # V6-3 fix: Human-corrected masks are weighted 1.0 (same as human gold annotations)
            sample_weights = torch.tensor(
                [1.0 if p in ['human', 'human_corrected'] else pseudo_weight for p in provenance_list],
                device=self.device, dtype=torch.float32
            )

            inputs_u = None
            if iter_unlabeled is not None:
                try:
                    batch_u = next(iter_unlabeled)
                except StopIteration:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    batch_u = next(iter_unlabeled)
                inputs_u = batch_u['image'].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(self.device_type, enabled=(self.device_type == 'cuda')):
                # 1. Supervised Loss with exact per-sample weighting (V-8 fix)
                preds_l = self.net_A(inputs_l)
                loss_sup_list = []
                for b in range(inputs_l.size(0)):
                    p_b = preds_l[b:b+1] if torch.is_tensor(preds_l) else [p[b:b+1] for p in preds_l]
                    l_b = compute_multiscale_loss(self.criterion, p_b, targets_l[b:b+1])
                    loss_sup_list.append(l_b)
                loss_sup_tensor = torch.stack(loss_sup_list)
                loss_sup = (loss_sup_tensor * sample_weights).sum() / (sample_weights.sum() + 1e-8)

                # 2. Unsupervised Loss via MC Dropout Teacher with input perturbation asymmetry (N-5 & V7-1 fix)
                loss_unsup = torch.tensor(0.0, device=self.device)
                uncert_val = 0.0

                if inputs_u is not None:
                    # V7-1 & V8-1 fix: Use production _make_unlabeled_views helper for shared spatial transform
                    try:
                        inputs_u_teacher, inputs_u_student = self._make_unlabeled_views(inputs_u)
                    except Exception as e:
                        # V7-2 fix: Log explicit warning when augmentation fails instead of silent fallback
                        print(f"[HASSL Warning] Augmentation failed in train_one_epoch_uamt: {e}. Falling back to unaugmented inputs.")
                        inputs_u_teacher = inputs_u
                        inputs_u_student = inputs_u

                    preds_u = self.net_A(inputs_u_student)
                    if isinstance(preds_u, (list, tuple)):
                        preds_u = preds_u[0]
                    elif preds_u.ndim == 6:
                        preds_u = preds_u[:, 0]

                    # Perform genuine stochastic MC dropout through teacher (H-1 fix)
                    pseudo_probs, uncertainty = self.teacher.forward_mc_dropout(
                        inputs_u_teacher, num_passes=self.config.mc_dropout_passes
                    )

                    thresh = torch.quantile(uncertainty.detach().float(), 0.75)
                    mask = (uncertainty < thresh).float()

                    if self.num_classes == 1:
                        pseudo_labels = (pseudo_probs > 0.5).float()
                    else:
                        pseudo_labels = torch.argmax(pseudo_probs, dim=1, keepdim=True)

                    if pseudo_labels.sum() > 0:
                        loss_unsup = self.masked_criterion(preds_u, pseudo_labels, mask)
                        if loss_unsup.ndim > 0:
                            loss_unsup = loss_unsup.mean()
                    else:
                        loss_unsup = torch.tensor(0.0, device=self.device)

                    uncert_val = uncertainty.mean().item()

                    # --- Teacher diagnostics (no grad, detached) ---
                    with torch.no_grad():
                        pp = pseudo_probs.detach().float()
                        total_pseudo_conf += float(pp.mean().item())           # Mean confidence [0,1]
                        total_pseudo_fg   += float((pp > 0.5).float().mean().item())  # FG fraction
                        n_unlabeled_steps += 1

                loss = loss_sup + unsup_weight * loss_unsup

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net_A.parameters(), max_norm=5.0)  # V10-5 fix: Gradient clipping
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.teacher.update(self.net_A, decay=self.config.ema_decay)

            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_unsup += loss_unsup.item()
            total_uncert += uncert_val

        N = max(1, len(self.labeled_loader))
        M = max(1, n_unlabeled_steps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            total_loss / N,
            total_sup / N,
            total_unsup / N,
            total_uncert / N,
            total_pseudo_conf / M,   # NEW: teacher mean confidence on unlabeled
            total_pseudo_fg / M,     # NEW: teacher foreground fraction on unlabeled
        )

    def train_one_epoch_cps(self, epoch: int):
        self.net_A.train()
        self.net_B.train()
        unsup_weight = self.get_rampup_weight(epoch)
        pseudo_weight = getattr(self.config, 'pseudo_label_weight', 0.5)
        total_loss, total_sup, total_unsup = 0.0, 0.0, 0.0

        if self.unlabeled_loader is None or len(self.unlabeled_loader.dataset) == 0:
            iter_unlabeled = None
        else:
            iter_unlabeled = iter(self.unlabeled_loader)

        for batch_idx, batch_data in enumerate(self.labeled_loader):
            inputs_l = batch_data['image'].to(self.device)
            targets_l = batch_data['label'].to(self.device)
            provenance_list = batch_data.get('provenance', ['human'] * inputs_l.size(0))

            # V6-3 fix: Human-corrected masks are weighted 1.0
            sample_weights = torch.tensor(
                [1.0 if p in ['human', 'human_corrected'] else pseudo_weight for p in provenance_list],
                device=self.device, dtype=torch.float32
            )

            inputs_u = None
            if iter_unlabeled is not None:
                try:
                    batch_u = next(iter_unlabeled)
                except StopIteration:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    batch_u = next(iter_unlabeled)
                inputs_u = batch_u['image'].to(self.device)

            self.optimizer_A.zero_grad()
            self.optimizer_B.zero_grad()

            with torch.amp.autocast(self.device_type, enabled=(self.device_type == 'cuda')):
                preds_A_l = self.net_A(inputs_l)
                preds_B_l = self.net_B(inputs_l)

                loss_sup_A_list = [compute_multiscale_loss(self.criterion, preds_A_l[b:b+1] if torch.is_tensor(preds_A_l) else [p[b:b+1] for p in preds_A_l], targets_l[b:b+1]) for b in range(inputs_l.size(0))]
                loss_sup_B_list = [compute_multiscale_loss(self.criterion, preds_B_l[b:b+1] if torch.is_tensor(preds_B_l) else [p[b:b+1] for p in preds_B_l], targets_l[b:b+1]) for b in range(inputs_l.size(0))]

                loss_sup_A = (torch.stack(loss_sup_A_list) * sample_weights).sum() / (sample_weights.sum() + 1e-8)
                loss_sup_B = (torch.stack(loss_sup_B_list) * sample_weights).sum() / (sample_weights.sum() + 1e-8)

                loss_cps_A = torch.tensor(0.0, device=self.device)
                loss_cps_B = torch.tensor(0.0, device=self.device)

                if inputs_u is not None:
                    # CPS requires perturbation-induced consistency: each network must see a
                    # differently-augmented view of the same volume so their predictions diverge
                    # enough to provide meaningful cross-supervision.  Without augmentation both
                    # networks immediately agree → pseudo-labels collapse to the same bias.
                    # _make_unlabeled_views produces:
                    #   view_A: shared spatial aug + intensity aug (student A view)
                    #   view_B: shared spatial aug only             (student B view)
                    # We generate two independent augmentation pairs so A and B get distinct views.
                    try:
                        inputs_u_A, inputs_u_A_alt = self._make_unlabeled_views(inputs_u)
                        inputs_u_B, inputs_u_B_alt = self._make_unlabeled_views(inputs_u)
                        # net_A gets view_A, net_B gets view_B (independently augmented)
                        inputs_for_A = inputs_u_A.to(self.device)
                        inputs_for_B = inputs_u_B.to(self.device)
                    except Exception as e:
                        print(f"[HASSL Warning] CPS augmentation failed: {e}. Using unaugmented inputs.")
                        inputs_for_A = inputs_u
                        inputs_for_B = inputs_u

                    preds_A_u = self.net_A(inputs_for_A)
                    preds_B_u = self.net_B(inputs_for_B)

                    if isinstance(preds_A_u, (list, tuple)): preds_A_u = preds_A_u[0]
                    elif preds_A_u.ndim == 6: preds_A_u = preds_A_u[:, 0]

                    if isinstance(preds_B_u, (list, tuple)): preds_B_u = preds_B_u[0]
                    elif preds_B_u.ndim == 6: preds_B_u = preds_B_u[:, 0]

                    if self.num_classes == 1:
                        probs_A = torch.sigmoid(preds_A_u)
                        probs_B = torch.sigmoid(preds_B_u)
                        pseudo_A = (probs_A > 0.5).float()
                        pseudo_B = (probs_B > 0.5).float()
                        thresh_A = self.flex_match.get_threshold(probs_A)
                        thresh_B = self.flex_match.get_threshold(probs_B)
                        mask_A = (probs_A > thresh_A).float()
                        mask_B = (probs_B > thresh_B).float()
                    else:
                        probs_A = torch.softmax(preds_A_u, dim=1)
                        probs_B = torch.softmax(preds_B_u, dim=1)
                        pseudo_A = torch.argmax(probs_A, dim=1, keepdim=True)
                        pseudo_B = torch.argmax(probs_B, dim=1, keepdim=True)
                        thresh_A = self.flex_match.get_threshold(probs_A)
                        thresh_B = self.flex_match.get_threshold(probs_B)
                        mask_A = (probs_A.max(dim=1, keepdim=True).values > thresh_A.view(1, -1, 1, 1, 1).max(dim=1, keepdim=True).values).float()
                        mask_B = (probs_B.max(dim=1, keepdim=True).values > thresh_B.view(1, -1, 1, 1, 1).max(dim=1, keepdim=True).values).float()

                    loss_cps_A = self.masked_criterion(preds_A_u, pseudo_B, mask_B)
                    loss_cps_B = self.masked_criterion(preds_B_u, pseudo_A, mask_A)

                    if loss_cps_A.ndim > 0: loss_cps_A = loss_cps_A.mean()
                    if loss_cps_B.ndim > 0: loss_cps_B = loss_cps_B.mean()

                loss = loss_sup_A + loss_sup_B + unsup_weight * (loss_cps_A + loss_cps_B)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer_A)
            self.scaler.unscale_(self.optimizer_B)
            torch.nn.utils.clip_grad_norm_(self.net_A.parameters(), max_norm=5.0)  # V10-5 fix: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.net_B.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer_A)
            self.scaler.step(self.optimizer_B)
            self.scaler.update()

            total_loss += loss.item()
            total_sup += (loss_sup_A + loss_sup_B).item() / 2
            total_unsup += (loss_cps_A + loss_cps_B).item() / 2

        N = max(1, len(self.labeled_loader))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss / N, total_sup / N, total_unsup / N, 0.0

    @torch.no_grad()
    def validate(self, epoch: int = 0, should_log_image: bool = False) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader.dataset) == 0:
            return {
                'val_dice': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_rve_pct': 0.0,
                'val_volume_r2': 0.0,
                'val_hd95': float('nan'),
            }

        self.net_A.eval()
        self.dice_metric.reset()

        from monai.inferers import SlidingWindowInferer
        from monai.metrics import DiceMetric

        dice_metric_lcc = DiceMetric(include_background=False if self.num_classes > 1 else True, reduction="mean")
        confusion_metric = build_confusion_metric(
            include_background=False if self.num_classes > 1 else True
        )
        confusion_metric_lcc = build_confusion_metric(
            include_background=False if self.num_classes > 1 else True
        )
        hd95_metric = build_hd95_metric(
            include_background=False if self.num_classes > 1 else True
        )
        # Teacher Dice metric — prototype mode only (EMA shadow model)
        # Runs alongside student in the same val loop so there is no extra DataLoader pass.
        teacher_dice_metric = DiceMetric(include_background=False if self.num_classes > 1 else True, reduction="mean")
        has_teacher = self.mode == 'prototype' and hasattr(self, 'teacher')
        if has_teacher:
            self.teacher.shadow.eval()

        pred_vols_mm3 = []
        pred_vols_mm3_lcc = []
        gt_vols_mm3 = []
        first_batch_sample = None
        first_batch_native_sample = None

        sw_bs = 1 if self.mode == 'prototype' else getattr(self.config, 'sw_batch_size', 2)
        inferer = SlidingWindowInferer(roi_size=self.config.spatial_size, sw_batch_size=sw_bs, overlap=0.25)

        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs = batch_data['image'].to(self.device)
            targets = batch_data['label'].to(self.device)

            with torch.amp.autocast(self.device_type, enabled=(self.device_type == 'cuda')):
                # --- Student forward ---
                preds = inferer(inputs, self.net_A)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                elif preds.ndim == 6:
                    preds = preds[:, 0]

                if self.num_classes == 1:
                    preds_binary = (torch.sigmoid(preds) > 0.5).float()
                else:
                    preds_binary = torch.argmax(preds, dim=1, keepdim=True).float()

                # --- Teacher forward (prototype mode only, no grad already via @torch.no_grad) ---
                if has_teacher:
                    t_preds = inferer(inputs, self.teacher.shadow)
                    if isinstance(t_preds, (list, tuple)): t_preds = t_preds[0]
                    elif t_preds.ndim == 6: t_preds = t_preds[:, 0]
                    if self.num_classes == 1:
                        t_binary = (torch.sigmoid(t_preds) > 0.5).float()
                    else:
                        t_binary = torch.argmax(t_preds, dim=1, keepdim=True).float()
                    teacher_dice_metric(y_pred=t_binary, y=targets)

            lcc_min_size_voxels = getattr(self.config, 'lcc_min_size_voxels', 100)
            preds_binary_lcc = apply_keep_largest_cc(preds_binary, min_size_voxels=lcc_min_size_voxels)

            if first_batch_sample is None and should_log_image:
                # Compute MC Dropout epistemic variance & TTA aleatoric variance 3D maps for sample 0
                mc_var_t = torch.zeros_like(preds_binary[0])
                tta_var_t = torch.zeros_like(preds_binary[0])
                try:
                    from hassl.active.query_strategies import TTAUncertaintyScorer
                    from hassl.training.ema import enable_dropout

                    # 1. MC Dropout variance
                    enable_dropout(self.net_A)
                    with torch.no_grad():
                        mc_preds = []
                        for _ in range(min(5, getattr(self.config, 'mc_dropout_passes', 5))):
                            p = inferer(inputs[0:1], self.net_A)
                            if isinstance(p, (list, tuple)): p = p[0]
                            elif p.ndim == 6: p = p[:, 0]
                            p_prob = torch.sigmoid(p) if self.num_classes == 1 else torch.softmax(p, dim=1)
                            mc_preds.append(p_prob)
                        mc_var_t = torch.stack(mc_preds, dim=0).var(dim=0)[0]

                    # 2. TTA Aleatoric variance
                    self.net_A.eval()
                    tta_scorer = TTAUncertaintyScorer(self.net_A, num_passes=4)
                    with torch.no_grad():
                        tta_preds = []
                        for _ in range(4):
                            aug_in, flipped_dims = tta_scorer._augment(inputs[0:1])
                            p = inferer(aug_in, self.net_A)
                            if isinstance(p, (list, tuple)): p = p[0]
                            elif p.ndim == 6: p = p[:, 0]
                            p_prob = torch.sigmoid(p) if self.num_classes == 1 else torch.softmax(p, dim=1)
                            if flipped_dims:
                                p_prob = torch.flip(p_prob, dims=tuple(flipped_dims))
                            tta_preds.append(p_prob)
                        tta_var_t = torch.stack(tta_preds, dim=0).var(dim=0)[0]
                except Exception as e:
                    logger.warning("[HASSL] Could not compute validation uncertainty maps: %s", e)

                first_batch_sample = (
                    inputs[0].detach().cpu(),
                    targets[0].detach().cpu(),
                    preds_binary[0].detach().cpu(),
                    preds_binary_lcc[0].detach().cpu(),
                    mc_var_t.detach().cpu(),
                    tta_var_t.detach().cpu(),
                )

            self.dice_metric(y_pred=preds_binary, y=targets)
            dice_metric_lcc(y_pred=preds_binary_lcc, y=targets)

            confusion_metric(y_pred=preds_binary, y=targets)
            confusion_metric_lcc(y_pred=preds_binary_lcc, y=targets)

            from monai.data import decollate_batch, MetaTensor

            decollated_samples = decollate_batch(batch_data)

            # Metrics inversion: pred, pred_lcc, label only — nearest_interp=True (scalar, all MONAI versions)
            # nearest_interp as a list is not universally supported; keep it scalar here.
            inv_transform = build_invertd(
                keys=["pred", "pred_lcc", "label"],
                transform=self.val_transform,
                orig_keys=["image", "image", "label"],
                nearest_interp=True,
                to_tensor=True,
            )

            # Image-only inversion for native-space visualization preview (built once per batch,
            # used only for sample 0 on image-logging epochs). Separate call to avoid:
            #   1. list-form nearest_interp compatibility issues
            #   2. applied_operations exhaustion when all 4 keys share the same ops trace
            inv_image_transform = build_invertd(
                keys=["image"],
                transform=self.val_transform,
                orig_keys=["image"],
                nearest_interp=False,  # bilinear/trilinear for smooth image inversion
                to_tensor=True,
            ) if should_log_image else None

            # Fix 1: extract affine-derived voxel volume ONCE per sample, BEFORE try/except.
            # Previously, the fallback path used config.spacing (default 1.0,1.0,1.0 mm) which
            # produced wrong mm³ units and made val_volume_r2 meaningless. Now both the Invertd
            # success path and the fallback use the same scanner-native voxel volume.
            lcc_min_size = getattr(self.config, 'lcc_min_size_voxels', 100)

            for b, sample in enumerate(decollated_samples):
                # --- Affine-derived voxel volume (scanner-native mm³) ---
                voxel_vol_mm3 = None

                # Priority 1: original_affine from MetaTensor meta (most accurate — pre-transform)
                if hasattr(inputs, 'meta'):
                    raw = inputs.meta.get('original_affine', None)
                    if raw is not None:
                        af = raw[b] if (torch.is_tensor(raw) and raw.ndim > 2) else raw
                        if torch.is_tensor(af) and af.shape == (4, 4):
                            voxel_vol_mm3 = float(torch.abs(torch.det(af[:3, :3].float())).item())

                # Priority 2: current affine from MetaTensor meta (post-transform, still usable)
                if voxel_vol_mm3 is None and hasattr(sample.get('image', None), 'meta'):
                    raw = sample['image'].meta.get('affine', None)
                    if raw is not None and torch.is_tensor(raw) and raw.shape == (4, 4):
                        voxel_vol_mm3 = float(torch.abs(torch.det(raw[:3, :3].float())).item())

                # Priority 3: image_meta_dict (older MONAI data format)
                if voxel_vol_mm3 is None and 'image_meta_dict' in batch_data:
                    meta = batch_data['image_meta_dict']
                    for key in ('original_affine', 'affine'):
                        if key in meta:
                            raw = meta[key]
                            af = raw[b] if (torch.is_tensor(raw) and raw.ndim > 2) else raw
                            if torch.is_tensor(af) and af.shape == (4, 4):
                                voxel_vol_mm3 = float(torch.abs(torch.det(af[:3, :3].float())).item())
                                break

                # Fallback: config.spacing (still wrong for micro-spacing but better than silence)
                if voxel_vol_mm3 is None or voxel_vol_mm3 <= 0:
                    voxel_vol_mm3 = float(
                        self.config.spacing[0] * self.config.spacing[1] * self.config.spacing[2]
                    )
                    logger.debug(
                        "[HASSL] validate sample %d: affine not found, using config.spacing for voxel volume", b
                    )

                try:
                    # Wrap predictions as MetaTensors carrying the input's transform trace.
                    # Use deepcopy on applied_operations so popping operations during Invertd
                    # does not mutate the reference list shared across keys.
                    ops_copy1 = copy.deepcopy(sample['image'].applied_operations) if hasattr(sample['image'], 'applied_operations') else []
                    ops_copy2 = copy.deepcopy(sample['image'].applied_operations) if hasattr(sample['image'], 'applied_operations') else []
                    meta_copy = copy.deepcopy(sample['image'].meta) if hasattr(sample['image'], 'meta') else {}

                    pred_mt = MetaTensor(
                        preds_binary[b].clone(),
                        meta=meta_copy,
                        applied_operations=ops_copy1,
                    )
                    pred_lcc_mt = MetaTensor(
                        preds_binary_lcc[b].clone(),
                        meta=copy.deepcopy(meta_copy),
                        applied_operations=ops_copy2,
                    )
                    sample["pred"] = pred_mt
                    sample["pred_lcc"] = pred_lcc_mt
                    if "label" not in sample:
                        sample["label"] = targets[b].clone()

                    inv_out = inv_transform(sample)
                    inv_pred = inv_out["pred"]
                    inv_pred_lcc = inv_out["pred_lcc"]
                    inv_gt = inv_out["label"]

                    pv_mm3 = float(inv_pred.sum().item()) * voxel_vol_mm3
                    pv_mm3_lcc = float(inv_pred_lcc.sum().item()) * voxel_vol_mm3
                    gv_mm3 = float(inv_gt.sum().item()) * voxel_vol_mm3

                    # Capture native-space tensors for sample 0 visualization.
                    # Image is inverted via a SEPARATE single-key Invertd (inv_image_transform)
                    # using a fresh ops deepcopy — this avoids the ops-exhaustion bug where
                    # inverting image first consumes applied_operations, leaving pred/label empty.
                    if b == 0 and first_batch_native_sample is None and should_log_image and inv_image_transform is not None:
                        try:
                            img_ops = copy.deepcopy(sample['image'].applied_operations) if hasattr(sample['image'], 'applied_operations') else []
                            img_meta = copy.deepcopy(sample['image'].meta) if hasattr(sample['image'], 'meta') else {}
                            from monai.data import MetaTensor as _MT
                            img_mt = _MT(sample['image'].clone(), meta=img_meta, applied_operations=img_ops)
                            inv_img_out = inv_image_transform({"image": img_mt})
                            first_batch_native_sample = (
                                inv_img_out["image"].detach().cpu(),
                                inv_gt.detach().cpu(),
                                inv_pred.detach().cpu(),
                                inv_pred_lcc.detach().cpu(),
                            )
                        except Exception as img_e:
                            logger.warning("[HASSL] Native-space image inversion failed (preview only): %s", img_e)
                except Exception as e:
                    logger.warning(
                        "[HASSL] validate Invertd failed for sample %d, "
                        "falling back to resized-space volume: %s", b, e
                    )
                    # voxel_vol_mm3 already computed above — same correct units in fallback
                    pv_mm3 = float(preds_binary[b].sum().item()) * voxel_vol_mm3
                    pv_mm3_lcc = float(preds_binary_lcc[b].sum().item()) * voxel_vol_mm3
                    gv_mm3 = float(targets[b].sum().item()) * voxel_vol_mm3

                spacing_b = self.config.spacing
                if hasattr(inputs, 'meta') and 'affine' in inputs.meta:
                    try:
                        affine_b = inputs.meta['affine'][b]
                        spacing_b = tuple(
                            float(torch.linalg.norm(affine_b[:3, i]).item()) for i in range(3)
                        )
                    except Exception:
                        pass
                try:
                    hd95_metric(y_pred=preds_binary[b:b+1], y=targets[b:b+1], spacing=spacing_b)
                except Exception:
                    pass

                pred_vols_mm3.append(pv_mm3)
                pred_vols_mm3_lcc.append(pv_mm3_lcc)
                gt_vols_mm3.append(gv_mm3)

        val_dice = self.dice_metric.aggregate().item()
        val_dice = float('nan') if torch.isnan(torch.tensor(val_dice)) else float(val_dice)

        val_dice_lcc = dice_metric_lcc.aggregate().item()
        val_dice_lcc = float('nan') if torch.isnan(torch.tensor(val_dice_lcc)) else float(val_dice_lcc)

        cm_res = confusion_metric.aggregate()
        val_prec = float('nan') if torch.isnan(cm_res[0]).any() else float(cm_res[0].item())
        val_rec  = float('nan') if torch.isnan(cm_res[1]).any() else float(cm_res[1].item())

        cm_res_lcc = confusion_metric_lcc.aggregate()
        val_prec_lcc = float('nan') if torch.isnan(cm_res_lcc[0]).any() else float(cm_res_lcc[0].item())
        val_rec_lcc  = float('nan') if torch.isnan(cm_res_lcc[1]).any() else float(cm_res_lcc[1].item())

        try:
            raw_hd95 = hd95_metric.aggregate().item()
            raw_t = torch.tensor(raw_hd95)
            val_hd95 = float('nan') if (torch.isnan(raw_t) or torch.isinf(raw_t)) else float(raw_hd95)
        except Exception:
            val_hd95 = float('nan')

        pred_arr = np.array(pred_vols_mm3)
        pred_arr_lcc = np.array(pred_vols_mm3_lcc)
        gt_arr = np.array(gt_vols_mm3)

        rve_list = np.abs(pred_arr - gt_arr) / (gt_arr + 1e-8) * 100.0
        val_rve_pct = float(np.mean(rve_list)) if len(rve_list) > 0 else 0.0

        rve_list_lcc = np.abs(pred_arr_lcc - gt_arr) / (gt_arr + 1e-8) * 100.0
        val_rve_pct_lcc = float(np.mean(rve_list_lcc)) if len(rve_list_lcc) > 0 else 0.0

        if len(gt_arr) > 1 and np.var(gt_arr) > 1e-6:
            ss_res = np.sum((gt_arr - pred_arr) ** 2)
            ss_res_lcc = np.sum((gt_arr - pred_arr_lcc) ** 2)
            ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
            val_volume_r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
            val_volume_r2_lcc = float(1.0 - (ss_res_lcc / (ss_tot + 1e-8)))
        else:
            val_volume_r2 = 1.0 if np.allclose(pred_arr, gt_arr) else 0.0
            val_volume_r2_lcc = 1.0 if np.allclose(pred_arr_lcc, gt_arr) else 0.0

        # Teacher Dice (prototype mode only)
        if has_teacher:
            try:
                raw_t = teacher_dice_metric.aggregate().item()
                val_dice_teacher = float('nan') if torch.isnan(torch.tensor(raw_t)) else float(raw_t)
            except Exception:
                val_dice_teacher = float('nan')
            self.teacher.shadow.train()  # restore teacher to train-compatible state
        else:
            val_dice_teacher = float('nan')

        if should_log_image and first_batch_sample is not None:
            self.log_validation_samples(
                epoch=epoch,
                model_sample=first_batch_sample,
                native_sample=first_batch_native_sample
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'val_dice': val_dice,
            'val_dice_teacher': val_dice_teacher,  # EMA teacher Dice (NaN in CPS mode)
            'val_dice_lcc': val_dice_lcc,
            'val_precision': val_prec,
            'val_precision_lcc': val_prec_lcc,
            'val_recall': val_rec,
            'val_recall_lcc': val_rec_lcc,
            'val_rve_pct': val_rve_pct,
            'val_rve_pct_lcc': val_rve_pct_lcc,
            'val_volume_r2': val_volume_r2,
            'val_volume_r2_lcc': val_volume_r2_lcc,
            'val_hd95': val_hd95,
            'val_pred_vol_mm3_mean': float(np.mean(pred_arr)) if len(pred_arr) > 0 else 0.0,
            'val_pred_vol_mm3_mean_lcc': float(np.mean(pred_arr_lcc)) if len(pred_arr_lcc) > 0 else 0.0,
            'val_gt_vol_mm3_mean': float(np.mean(gt_arr)) if len(gt_arr) > 0 else 0.0,
        }

    def log_data_preview(self, epoch: int):
        """Log 3-axis (axial/coronal/sagittal) views of a preprocessed val batch after resize.

        Helps verify:
        - AspectRatioResizeWithPadd is preserving anatomy correctly
        - Image intensity normalisation looks reasonable (not flat/dark)
        - Label mask is correctly co-registered with the image
        - Foreground fraction is non-trivially small
        Called at epoch 0 and every log_image_every_n_epochs thereafter.
        """
        try:
            if self.val_loader is None or len(self.val_loader.dataset) == 0:
                return
            batch = next(iter(self.val_loader))
            img_np = batch['image'][0, 0].cpu().numpy()   # [D, H, W]
            lbl_np = batch['label'][0, 0].cpu().numpy()   # [D, H, W]

            D, H, W = img_np.shape
            fg_frac = float(lbl_np.mean())

            # Pick best slice per axis (highest GT foreground area)
            ax_idx = int(lbl_np.sum(axis=(1, 2)).argmax())   # axial
            co_idx = int(lbl_np.sum(axis=(0, 2)).argmax())   # coronal
            sa_idx = int(lbl_np.sum(axis=(0, 1)).argmax())   # sagittal

            def make_panel(img_slice, lbl_slice):
                """Per-slice 1-99th percentile contrast stretch + solid mask overlay at actual pixel size."""
                lo, hi = np.percentile(img_slice, [1, 99])
                norm = np.clip((img_slice - lo) / (hi - lo + 1e-8), 0, 1)
                gray = (norm * 255).astype(np.uint8)
                rgb  = np.stack([gray, gray, gray], axis=-1)  # H x W x 3

                p_img = rgb.copy()

                p_lbl = rgb.copy()
                mask  = lbl_slice > 0.5
                if mask.any():
                    overlay = p_lbl.copy()
                    overlay[mask] = [0, 220, 100]
                    p_lbl = (p_lbl * 0.5 + overlay * 0.5).astype(np.uint8)

                h, w = gray.shape
                header = np.full((20, w, 3), 40, dtype=np.uint8)
                panel  = np.concatenate([header, p_img, p_lbl], axis=0)
                return panel

            panel_ax = make_panel(img_np[ax_idx],          lbl_np[ax_idx])
            panel_co = make_panel(img_np[:, co_idx, :],    lbl_np[:, co_idx, :])
            panel_sa = make_panel(img_np[:, :, sa_idx],    lbl_np[:, :, sa_idx])

            # Pad all panels to same height before concatenating horizontally
            max_h = max(p.shape[0] for p in [panel_ax, panel_co, panel_sa])
            def pad_h(p):
                diff = max_h - p.shape[0]
                return np.pad(p, ((0, diff), (0, 0), (0, 0)), mode='constant') if diff > 0 else p

            grid = np.concatenate([pad_h(panel_ax), pad_h(panel_co), pad_h(panel_sa)], axis=1)
            out_h, out_w = grid.shape[:2]

            self.tracker.log_image(
                grid,
                name="data_preview_after_resize",
                step=epoch,
                caption=(
                    f"Epoch {epoch} — Preprocessed val sample after resize "
                    f"(Axial | Coronal | Sagittal) — each axis: [Image | Label overlay] "
                    f"| Volume: {D}x{H}x{W} | FG fraction: {fg_frac:.4f} "
                    f"| GT slices: ax={ax_idx} co={co_idx} sa={sa_idx} "
                    f"| Logged: {out_h}x{out_w}px"
                ),
                save_local_dir=os.path.join(self.config.log_dir, "data_previews")
            )
        except Exception as e:
            logger.warning("[HASSL] Failed to generate data preview: %s", e)

    def log_validation_samples(
        self,
        epoch: int,
        model_sample: Tuple[torch.Tensor, ...],
        native_sample: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        """Log 3-axis x 5-panel validation preview strips for BOTH model space and native physical space.

        Previews logged:
          1. `val_slice_preview_model_space`: Slices at exact model resolution (128x128x128)
          2. `val_slice_preview_native_space`: Slices at actual scanner voxel resolution (after Invertd)
          3. `val_uncertainty_preview`: MC/TTA epistemic and aleatoric heatmaps (model space)

        Each strip shows 3 rows (Axial, Coronal, Sagittal) x 5 columns:
          [Image | GT (green) | Raw Pred (cyan) | LCC Pred (magenta) | Error Map (TP green/FP red/FN blue)]

        Images are logged at the exact pixel dimensions of the underlying volume — no artificial
        upscaling or downscaling. Model space panels are 128px per slice; native space panels are
        whatever the actual scanner voxel grid is.
        """
        try:
            from hassl.utils.visualization import render_uncertainty_slice_grid

            def clahe_norm(img_slice: np.ndarray) -> np.ndarray:
                """Per-slice 1-99th percentile contrast stretch -> uint8 [0,255]."""
                lo, hi = np.percentile(img_slice, [1, 99])
                norm = np.clip((img_slice - lo) / (hi - lo + 1e-8), 0, 1)
                return (norm * 255).astype(np.uint8)

            def alpha_overlay(gray_rgb: np.ndarray, mask: np.ndarray, color: list, alpha: float = 0.55) -> np.ndarray:
                """Blend solid color over mask region with alpha transparency."""
                out = gray_rgb.copy().astype(np.float32)
                overlay = np.zeros_like(out)
                overlay[mask] = color
                out[mask] = out[mask] * (1 - alpha) + overlay[mask] * alpha
                return np.clip(out, 0, 255).astype(np.uint8)

            def build_3axis_grid(img_t, gt_t, pred_t, lcc_t):
                """Build 3-row (axial, coronal, sagittal) x 5-col preview grid.
                Logged at the exact pixel dimensions of the volume — no rescaling.
                """
                img_np  = img_t[0].numpy()    # [D, H, W]
                gt_np   = gt_t[0].numpy()
                pred_np = pred_t[0].numpy()
                lcc_np  = lcc_t[0].numpy()

                D, H, W = img_np.shape

                def make_row(img_slice, gt_slice, pred_slice, lcc_slice):
                    gray = clahe_norm(img_slice)
                    rgb  = np.stack([gray, gray, gray], axis=-1)

                    gt_m = gt_slice > 0.5
                    pr_m = pred_slice > 0.5
                    lc_m = lcc_slice > 0.5

                    p1 = rgb.copy()
                    p2 = alpha_overlay(rgb, gt_m, [30, 230, 80], alpha=0.6)
                    p3 = alpha_overlay(rgb, pr_m, [0, 200, 230], alpha=0.6)
                    p4 = alpha_overlay(rgb, lc_m, [230, 50, 230], alpha=0.6)

                    p5 = np.zeros_like(rgb)
                    p5[gt_m & lc_m]    = [0, 210, 60]    # TP green
                    p5[(~gt_m) & lc_m] = [230, 40, 40]   # FP red
                    p5[gt_m & (~lc_m)] = [40, 100, 230]  # FN blue

                    return np.concatenate([p1, p2, p3, p4, p5], axis=1)

                ax_sums = gt_np.sum(axis=(1, 2))
                co_sums = gt_np.sum(axis=(0, 2))
                sa_sums = gt_np.sum(axis=(0, 1))

                ax_idx = int(ax_sums.argmax()) if ax_sums.max() > 0 else D // 2
                co_idx = int(co_sums.argmax()) if co_sums.max() > 0 else H // 2
                sa_idx = int(sa_sums.argmax()) if sa_sums.max() > 0 else W // 2

                row_ax = make_row(img_np[ax_idx],       gt_np[ax_idx],       pred_np[ax_idx],       lcc_np[ax_idx])
                row_co = make_row(img_np[:, co_idx, :], gt_np[:, co_idx, :], pred_np[:, co_idx, :], lcc_np[:, co_idx, :])
                row_sa = make_row(img_np[:, :, sa_idx], gt_np[:, :, sa_idx], pred_np[:, :, sa_idx], lcc_np[:, :, sa_idx])

                max_w = max(r.shape[1] for r in [row_ax, row_co, row_sa])
                def pad_w(r):
                    diff = max_w - r.shape[1]
                    return np.pad(r, ((0, 0), (0, diff), (0, 0)), mode='constant') if diff > 0 else r

                sep  = np.full((3, max_w, 3), 80, dtype=np.uint8)
                grid = np.concatenate([pad_w(row_ax), sep, pad_w(row_co), sep, pad_w(row_sa)], axis=0)
                out_h, out_w = grid.shape[:2]
                return grid, (ax_idx, co_idx, sa_idx), (D, H, W), float(gt_np.mean()), float(lcc_np.mean()), (out_h, out_w)

            # -------------------------------------------------------------------
            # 1. Model Space Preview — exact 128³ pixel dimensions
            #    Shows precisely what the model operates on.
            # -------------------------------------------------------------------
            img_m, gt_m, pred_m, lcc_m = model_sample[0], model_sample[1], model_sample[2], model_sample[3]
            mc_var_t  = model_sample[4] if len(model_sample) > 4 else None
            tta_var_t = model_sample[5] if len(model_sample) > 5 else None

            grid_m, (ax_i, co_i, sa_i), shape_m, gt_f_m, pr_f_m, (out_h_m, out_w_m) = build_3axis_grid(
                img_m, gt_m, pred_m, lcc_m
            )

            self.tracker.log_image(
                grid_m,
                name="val_slice_preview_model_space",
                step=epoch,
                caption=(
                    f"Epoch {epoch} [Model Space {shape_m[0]}x{shape_m[1]}x{shape_m[2]} — {out_h_m}x{out_w_m}px] "
                    f"| Slices: ax={ax_i} co={co_i} sa={sa_i} "
                    f"| Cols: Image / GT(green) / RawPred(cyan) / LCC(magenta) / Error(TP-G FP-R FN-B) "
                    f"| GT fg={gt_f_m:.3f} Pred fg={pr_f_m:.3f}"
                ),
                save_local_dir=os.path.join(self.config.log_dir, "val_previews_model_space")
            )

            # -------------------------------------------------------------------
            # 2. Native Physical Space Preview — actual scanner voxel resolution
            #    Image + segmentation both inverted through Invertd (orientation→
            #    spacing→resize), logged at their true pixel dimensions.
            # -------------------------------------------------------------------
            if native_sample is not None:
                img_n, gt_n, pred_n, lcc_n = native_sample[0], native_sample[1], native_sample[2], native_sample[3]
                grid_n, (ax_in, co_in, sa_in), shape_n, gt_f_n, pr_f_n, (out_h_n, out_w_n) = build_3axis_grid(
                    img_n, gt_n, pred_n, lcc_n
                )

                self.tracker.log_image(
                    grid_n,
                    name="val_slice_preview_native_space",
                    step=epoch,
                    caption=(
                        f"Epoch {epoch} [Native Scanner Space {shape_n[0]}x{shape_n[1]}x{shape_n[2]} — {out_h_n}x{out_w_n}px] "
                        f"| Slices: ax={ax_in} co={co_in} sa={sa_in} "
                        f"| Cols: Image / GT(green) / RawPred(cyan) / LCC(magenta) / Error(TP-G FP-R FN-B) "
                        f"| GT fg={gt_f_n:.3f} Pred fg={pr_f_n:.3f}"
                    ),
                    save_local_dir=os.path.join(self.config.log_dir, "val_previews_native_space")
                )

            # -------------------------------------------------------------------
            # 3. Model Space Uncertainty Heatmaps (MC & TTA)
            # -------------------------------------------------------------------
            if mc_var_t is not None and tta_var_t is not None:
                slice_mc  = mc_var_t[0, ax_i].numpy()
                slice_tta = tta_var_t[0, ax_i].numpy()

                grid_uncert = render_uncertainty_slice_grid(
                    slice_img=img_m[0, ax_i].numpy(),
                    slice_gt=gt_m[0, ax_i].numpy() > 0.5,
                    slice_pred=pred_m[0, ax_i].numpy() > 0.5,
                    slice_mc_var=slice_mc,
                    slice_tta_var=slice_tta,
                    slice_lcc=lcc_m[0, ax_i].numpy() > 0.5,
                )

                self.tracker.log_image(
                    grid_uncert,
                    name="val_uncertainty_preview",
                    step=epoch,
                    caption=(
                        f"Epoch {epoch} Model Space Axial z={ax_i} "
                        "(Image | GT | RawPred | MC Epistemic | TTA Aleatoric | Error)"
                    ),
                    save_local_dir=os.path.join(self.config.log_dir, "val_uncertainty_previews")
                )
        except Exception as e:
            logger.warning("[HASSL] Failed to generate validation slice preview: %s", e)

    def train(self, num_epochs: int):
        end_epoch = self.start_epoch + num_epochs
        log_img_freq = getattr(self.config, 'log_image_every_n_epochs', 10)

        for epoch in range(self.start_epoch, end_epoch):
            if self.mode == 'prototype':
                train_loss, sup_loss, unsup_loss, uncert, pseudo_conf, pseudo_fg = self.train_one_epoch_uamt(epoch)
            else:
                train_loss, sup_loss, unsup_loss, uncert = self.train_one_epoch_cps(epoch)
                pseudo_conf, pseudo_fg = float('nan'), float('nan')

            # --- Per-epoch diagnostics: foreground fraction & zero-prediction alert ---
            try:
                fg_fracs, pred_fracs = [], []
                for bd in self.labeled_loader:
                    fg_fracs.append(float(bd['label'].float().mean().item()))
                    break  # only first batch for speed
                if self.val_loader is not None:
                    for bd in self.val_loader:
                        self.net_A.eval()
                        with torch.no_grad():
                            _p = self.net_A(bd['image'].to(self.device))
                            if isinstance(_p, (list, tuple)): _p = _p[0]
                            elif _p.ndim == 6: _p = _p[:, 0]
                            _pb = (torch.sigmoid(_p) > 0.5).float()
                            pred_fracs.append(float(_pb.mean().item()))
                        self.net_A.train()
                        break
                fg_mean  = np.mean(fg_fracs)  if fg_fracs  else float('nan')
                pf_mean  = np.mean(pred_fracs) if pred_fracs else float('nan')
                if not np.isnan(pf_mean) and pf_mean < 1e-4:
                    print(f"  [HASSL WARN] Epoch {epoch}: model producing near-ZERO predictions "
                          f"(pred_fg={pf_mean:.5f}). Possible mode collapse or extreme class imbalance.")
                self.tracker.log_metrics({
                    'train_fg_fraction': fg_mean,
                    'val_pred_fg_fraction': pf_mean,
                }, step=epoch)
            except Exception:
                pass

            should_log_image = ((epoch + 1) % log_img_freq == 0) or (epoch == end_epoch - 1)

            # Log post-resize data preview at epoch 0 and on image-log epochs
            if epoch == self.start_epoch or should_log_image:
                self.log_data_preview(epoch)

            val_metrics = self.validate(epoch=epoch, should_log_image=should_log_image)
            val_dice = val_metrics['val_dice']

            # Step LR scheduler and get current learning rate
            current_lr = self.config.train_lr
            if self.mode == 'prototype' and getattr(self, 'scheduler', None) is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            elif self.mode == 'full':
                if getattr(self, 'scheduler_A', None) is not None:
                    if isinstance(self.scheduler_A, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler_A.step(val_dice)
                    else:
                        self.scheduler_A.step()
                if getattr(self, 'scheduler_B', None) is not None:
                    if isinstance(self.scheduler_B, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler_B.step(val_dice)
                    else:
                        self.scheduler_B.step()
                current_lr = self.optimizer_A.param_groups[0]['lr']

            metrics = {
                'train_loss': train_loss,
                'supervised_loss': sup_loss,
                'unsupervised_loss': unsup_loss,
                'learning_rate': current_lr,
                'uncertainty_mean': uncert,
                # Teacher diagnostic curves (prototype mode; NaN in CPS mode)
                'teacher_pseudo_conf': pseudo_conf,       # Mean teacher sigmoid confidence on unlabeled [0,1]
                'teacher_pseudo_fg_frac': pseudo_fg,      # Fraction of unlabeled voxels labelled as FG by teacher
                'consistency_rampup_weight': self.get_rampup_weight(epoch),  # Unsupervised loss rampup progress
                'epoch': epoch,
                **val_metrics
            }
            self.tracker.log_metrics(metrics, step=epoch)

            hd95_str    = f"{val_metrics['val_hd95']:.2f}mm" if not np.isnan(val_metrics['val_hd95']) else "N/A"
            dice_str    = f"{val_dice:.4f}"                    if not np.isnan(val_dice)                    else "N/A"
            t_dice_str  = f"{val_metrics['val_dice_teacher']:.4f}" if not np.isnan(val_metrics['val_dice_teacher']) else "N/A"
            prec_str    = f"{val_metrics['val_precision']:.4f}" if not np.isnan(val_metrics['val_precision']) else "N/A"
            rec_str     = f"{val_metrics['val_recall']:.4f}"    if not np.isnan(val_metrics['val_recall'])    else "N/A"

            print(f"  Epoch {epoch:3d}/{end_epoch} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Dice(S): {dice_str} | Dice(T): {t_dice_str} | "
                  f"Prec: {prec_str} | Rec: {rec_str} | "
                  f"RVE: {val_metrics['val_rve_pct']:.1f}% | R²: {val_metrics['val_volume_r2']:.3f} | "
                  f"HD95: {hd95_str} | LR: {current_lr:.6f}")

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth'),
                    epoch=epoch,
                )

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth'),
                    epoch=epoch,
                )

            # Early Stopping Check
            if self.early_stopper and self.early_stopper(val_dice):
                print(f"  [HASSL Early Stopping] Validation Dice did not improve for {self.early_stopper.patience} consecutive epochs. Early stopping at epoch {epoch + 1}.")
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth'),
                    epoch=epoch,
                )
                break

        print(f"  Best validation Dice: {self.best_dice:.4f}")

    def _load_pretrained(self, path: str):
        """Load SSL pre-trained encoder weights (partial load)."""
        print(f"  Loading SSL pre-trained weights from {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        model_dict = self.net_A.state_dict()
        pretrained_dict = {k: v for k, v in state.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.net_A.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pre-trained weights")

    def resume(self, path: str, weights_only: bool = False):
        """Resume training from a checkpoint (M-9 fix, V10-1/V10-2 fix)."""
        if not os.path.exists(path):
            print(f"  No checkpoint found at {path}, training from scratch.")
            return
        self.load_checkpoint(path, weights_only=weights_only)
        print(f"  Resumed from checkpoint: {path} at epoch {self.start_epoch} (best_dice={self.best_dice:.4f})")

    def save_checkpoint(self, path: str, epoch: int = 0):
        """Save training checkpoint including current epoch, scheduler, and early stopper (M-9 fix)."""
        state = {
            'net_A': self.net_A.state_dict(),
            'best_dice': self.best_dice,
            'epoch': epoch + 1,  # Resume from next epoch
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }
        if self.mode == 'prototype':
            state['teacher'] = self.teacher.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            if getattr(self, 'scheduler', None) is not None:
                state['scheduler'] = self.scheduler.state_dict()
        else:
            state['net_B'] = self.net_B.state_dict()
            state['optimizer_A'] = self.optimizer_A.state_dict()
            state['optimizer_B'] = self.optimizer_B.state_dict()
            if getattr(self, 'scheduler_A', None) is not None:
                state['scheduler_A'] = self.scheduler_A.state_dict()
            if getattr(self, 'scheduler_B', None) is not None:
                state['scheduler_B'] = self.scheduler_B.state_dict()

        if getattr(self, 'early_stopper', None) is not None:
            state['early_stopper'] = self.early_stopper.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        if self.tracker is not None:
            self.tracker.log_artifact(path, 'checkpoint')

    def load_checkpoint(self, path: str, weights_only: bool = False):
        """Load training checkpoint and restore state.

        When weights_only=True (e.g. starting a new AL round or model query),
        loads model weights and best_dice but resets optimizer, scheduler,
        and early stopper so each AL round runs a fresh optimization cycle (V10-1, V10-2 fix).
        """
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.net_A.load_state_dict(state['net_A'])
        self.best_dice = state.get('best_dice', 0.0)

        if self.mode == 'prototype':
            if 'teacher' in state:
                self.teacher.load_state_dict(state['teacher'])
        elif self.mode == 'full':
            if 'net_B' in state:
                self.net_B.load_state_dict(state['net_B'])

        if weights_only:
            # Reset epoch numbering, scheduler, and early stopper for new AL round
            self.start_epoch = 0
            if self.mode == 'prototype':
                self.scheduler = self._build_scheduler(self.optimizer)
            else:
                self.scheduler_A = self._build_scheduler(self.optimizer_A)
                self.scheduler_B = self._build_scheduler(self.optimizer_B)
            if self.early_stopper is not None:
                self.early_stopper.reset()
        else:
            self.start_epoch = state.get('epoch', 0)
            if self.mode == 'prototype':
                if 'optimizer' in state:
                    self.optimizer.load_state_dict(state['optimizer'])
                if 'scheduler' in state and getattr(self, 'scheduler', None) is not None:
                    self.scheduler.load_state_dict(state['scheduler'])
            elif self.mode == 'full':
                if 'optimizer_A' in state:
                    self.optimizer_A.load_state_dict(state['optimizer_A'])
                if 'optimizer_B' in state:
                    self.optimizer_B.load_state_dict(state['optimizer_B'])
                if 'scheduler_A' in state and getattr(self, 'scheduler_A', None) is not None:
                    self.scheduler_A.load_state_dict(state['scheduler_A'])
                if 'scheduler_B' in state and getattr(self, 'scheduler_B', None) is not None:
                    self.scheduler_B.load_state_dict(state['scheduler_B'])

            if 'early_stopper' in state and getattr(self, 'early_stopper', None) is not None:
                self.early_stopper.load_state_dict(state['early_stopper'])

    def get_models(self):
        """Return trained models for active learning query."""
        if self.mode == 'prototype':
            return self.net_A, self.teacher.shadow
        else:
            return self.net_A, self.net_B
