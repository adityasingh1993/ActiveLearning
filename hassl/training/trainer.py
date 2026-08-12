import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import monai
from monai.networks.nets import UNet, DynUNet, SwinUNETR
from monai.metrics import DiceMetric

from .ema import EMATeacher
from .losses import CombinedSegLoss, UncertaintyMaskedLoss, FlexMatchThreshold
from ..tracking import ExperimentTracker
from ..data.augmentations import get_weak_augmentation, get_strong_augmentation


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
            img_size=(128, 128, 128), in_channels=1, out_channels=num_classes,
            feature_size=48, use_checkpoint=True
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def compute_multiscale_loss(criterion, preds, target):
    """Compute deep-supervision loss if network outputs multi-scale heads (H-8 fix)."""
    if isinstance(preds, (list, tuple)):
        # Handle list of multi-scale outputs
        loss = 0.0
        weights = [1.0 / (2 ** i) for i in range(len(preds))]
        total_w = sum(weights)
        for p, w in zip(preds, weights):
            l = criterion(p, target)
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
            l = criterion(p, target)
            if l.ndim > 0:
                l = l.mean()
            loss += (weights[i] / total_w) * l
        return loss
    else:
        l = criterion(preds, target)
        return l.mean() if l.ndim > 0 else l


class HASSLTrainer:
    """Unified trainer supporting UA-Mean Teacher (prototype) and CPS (full) modes."""

    def __init__(self, config, labeled_loader, unlabeled_loader, val_loader,
                 tracker: ExperimentTracker, pretrained_weights=None):
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
        else:
            self.net_A = build_network(config.unet_backbone, self.num_classes, config.dropout).to(self.device)
            self.net_B = build_network('swinunetr', self.num_classes, 0.0).to(self.device)
            self.optimizer_A = torch.optim.AdamW(
                self.net_A.parameters(), lr=config.train_lr, weight_decay=config.train_weight_decay
            )
            self.optimizer_B = torch.optim.AdamW(
                self.net_B.parameters(), lr=config.train_lr, weight_decay=config.train_weight_decay
            )
            self.flex_match = FlexMatchThreshold(self.num_classes)

        self.criterion = CombinedSegLoss(self.num_classes, include_boundary=False)
        self.masked_criterion = UncertaintyMaskedLoss(self.criterion)
        self.best_dice = 0.0
        self.dice_metric = DiceMetric(include_background=False if self.num_classes > 1 else True, reduction="mean")
        self.start_epoch = 0

        # Pre-instantiate weak and strong augmentations for asymmetric Mean Teacher view (N-5 fix)
        self.weak_aug = get_weak_augmentation(keys=["image"])
        self.strong_aug = get_strong_augmentation(keys=["image"])

        # Load pre-trained SSL weights if available
        if pretrained_weights and os.path.exists(pretrained_weights):
            self._load_pretrained(pretrained_weights)

    def get_rampup_weight(self, epoch: int) -> float:
        if epoch < self.config.consistency_rampup_epochs:
            return float(epoch / max(1, self.config.consistency_rampup_epochs))
        return 1.0

    def train_one_epoch_uamt(self, epoch: int):
        self.net_A.train()
        unsup_weight = self.get_rampup_weight(epoch)
        pseudo_weight = getattr(self.config, 'pseudo_label_weight', 0.5)
        total_loss, total_sup, total_unsup, total_uncert = 0.0, 0.0, 0.0, 0.0

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

                # 2. Unsupervised Loss via MC Dropout Teacher with input perturbation asymmetry (N-5 fix)
                loss_unsup = torch.tensor(0.0, device=self.device)
                uncert_val = 0.0

                if inputs_u is not None:
                    # N-5 fix: Execute weak augmentation for teacher and strong augmentation for student
                    try:
                        inputs_u_teacher = torch.stack([self.weak_aug({"image": inputs_u[b]})["image"] for b in range(inputs_u.size(0))])
                        inputs_u_student = torch.stack([self.strong_aug({"image": inputs_u[b]})["image"] for b in range(inputs_u.size(0))])
                    except Exception:
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

                    loss_unsup = self.masked_criterion(preds_u, pseudo_labels, mask)
                    if loss_unsup.ndim > 0:
                        loss_unsup = loss_unsup.mean()

                    uncert_val = uncertainty.mean().item()

                loss = loss_sup + unsup_weight * loss_unsup

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.teacher.update(self.net_A, decay=self.config.ema_decay)

            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_unsup += loss_unsup.item()
            total_uncert += uncert_val

        N = max(1, len(self.labeled_loader))
        return total_loss / N, total_sup / N, total_unsup / N, total_uncert / N

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
                    # Both networks see aligned input (N-5 fix)
                    preds_A_u = self.net_A(inputs_u)
                    preds_B_u = self.net_B(inputs_u)

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
            self.scaler.step(self.optimizer_A)
            self.scaler.step(self.optimizer_B)
            self.scaler.update()

            total_loss += loss.item()
            total_sup += (loss_sup_A + loss_sup_B).item() / 2
            total_unsup += (loss_cps_A + loss_cps_B).item() / 2

        N = max(1, len(self.labeled_loader))
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
        from monai.metrics import ConfusionMatrixMetric, HausdorffDistanceMetric

        confusion_metric = ConfusionMatrixMetric(
            include_background=False if self.num_classes > 1 else True,
            metric_name=["precision", "recall"],
            reduction="mean"
        )
        # V-6 fix: Pass physical spacing to HD95 metric
        hd95_metric = HausdorffDistanceMetric(
            include_background=False if self.num_classes > 1 else True,
            percentile=95,
            reduction="mean"
        )

        pred_vols_mm3 = []
        gt_vols_mm3 = []
        first_batch_sample = None

        inferer = SlidingWindowInferer(roi_size=self.config.spatial_size, sw_batch_size=2, overlap=0.25)
        default_voxel_vol_mm3 = float(self.config.spacing[0] * self.config.spacing[1] * self.config.spacing[2])

        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs = batch_data['image'].to(self.device)
            targets = batch_data['label'].to(self.device)

            with torch.amp.autocast(self.device_type, enabled=(self.device_type == 'cuda')):
                preds = inferer(inputs, self.net_A)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                elif preds.ndim == 6:
                    preds = preds[:, 0]

                if self.num_classes == 1:
                    preds_binary = (torch.sigmoid(preds) > 0.5).float()
                else:
                    preds_binary = torch.argmax(preds, dim=1, keepdim=True).float()

            if first_batch_sample is None and should_log_image:
                first_batch_sample = (inputs[0].detach().cpu(), targets[0].detach().cpu(), preds_binary[0].detach().cpu())

            self.dice_metric(y_pred=preds_binary, y=targets)
            confusion_metric(y_pred=preds_binary, y=targets)

            # V-5 & V-6 fix: Extract physical voxel volume and spacing directly from current post-resize MetaTensor affine
            for b in range(inputs.size(0)):
                voxel_vol = default_voxel_vol_mm3
                spacing_b = self.config.spacing

                affine_b = None
                if hasattr(inputs, 'meta') and 'affine' in inputs.meta:
                    affine_b = inputs.meta['affine'][b]
                elif 'image_meta_dict' in batch_data and 'affine' in batch_data['image_meta_dict']:
                    affine_b = batch_data['image_meta_dict']['affine'][b]

                if affine_b is not None and torch.is_tensor(affine_b):
                    try:
                        voxel_vol = float(torch.abs(torch.det(affine_b[:3, :3])).item())
                        spacing_b = tuple(float(torch.linalg.norm(affine_b[:3, i]).item()) for i in range(3))
                    except Exception:
                        pass

                try:
                    # Pass post-resize physical spacing to HD95 metric (V-6 fix)
                    hd95_metric(y_pred=preds_binary[b:b+1], y=targets[b:b+1], spacing=spacing_b)
                except Exception:
                    pass

                pv_mm3 = float(preds_binary[b].sum().item()) * voxel_vol
                gv_mm3 = float(targets[b].sum().item()) * voxel_vol
                pred_vols_mm3.append(pv_mm3)
                gt_vols_mm3.append(gv_mm3)

        val_dice = self.dice_metric.aggregate().item()
        val_dice = 0.0 if torch.isnan(torch.tensor(val_dice)) else float(val_dice)

        cm_res = confusion_metric.aggregate()
        val_prec = float(cm_res[0].item()) if not torch.isnan(cm_res[0]).any() else 0.0
        val_rec = float(cm_res[1].item()) if not torch.isnan(cm_res[1]).any() else 0.0

        # V-7 fix: Propagate float('nan') when metric aggregation fails or produces NaN
        try:
            raw_hd95 = hd95_metric.aggregate().item()
            val_hd95 = float('nan') if torch.isnan(torch.tensor(raw_hd95)) else float(raw_hd95)
        except Exception:
            val_hd95 = float('nan')

        pred_arr = np.array(pred_vols_mm3)
        gt_arr = np.array(gt_vols_mm3)

        rve_list = np.abs(pred_arr - gt_arr) / (gt_arr + 1e-8) * 100.0
        val_rve_pct = float(np.mean(rve_list)) if len(rve_list) > 0 else 0.0

        if len(gt_arr) > 1 and np.var(gt_arr) > 1e-6:
            ss_res = np.sum((gt_arr - pred_arr) ** 2)
            ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
            val_volume_r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
        else:
            val_volume_r2 = 1.0 if np.allclose(pred_arr, gt_arr) else 0.0

        if should_log_image and first_batch_sample is not None:
            self.log_validation_samples(epoch, *first_batch_sample)

        return {
            'val_dice': val_dice,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_rve_pct': val_rve_pct,
            'val_volume_r2': val_volume_r2,
            'val_hd95': val_hd95,
        }

    def log_validation_samples(self, epoch: int, img_t: torch.Tensor, gt_t: torch.Tensor, pred_t: torch.Tensor):
        """Generate and log 4-panel slice preview grid (Image | GT | Pred | Error Map)."""
        try:
            img_np = img_t[0].numpy()
            gt_np = gt_t[0].numpy()
            pred_np = pred_t[0].numpy()

            gt_sums = gt_np.sum(axis=(1, 2))
            slice_idx = int(gt_sums.argmax()) if gt_sums.max() > 0 else img_np.shape[0] // 2

            slice_img = img_np[slice_idx]
            slice_gt = (gt_np[slice_idx] > 0.5).astype(np.float32)
            slice_pred = (pred_np[slice_idx] > 0.5).astype(np.float32)

            p_min, p_max = slice_img.min(), slice_img.max()
            slice_norm = (slice_img - p_min) / (p_max - p_min + 1e-8)
            base_gray = (slice_norm * 255).astype(np.uint8)
            base_rgb = np.stack([base_gray] * 3, axis=-1)

            # Panel 1: Original Image
            p1 = base_rgb.copy()

            # Panel 2: Ground Truth (Green overlay)
            p2 = base_rgb.copy()
            p2[slice_gt > 0, 1] = np.clip(p2[slice_gt > 0, 1].astype(np.int32) + 120, 0, 255).astype(np.uint8)

            # Panel 3: Prediction (Cyan overlay)
            p3 = base_rgb.copy()
            p3[slice_pred > 0, 0] = np.clip(p3[slice_pred > 0, 0].astype(np.int32) + 120, 0, 255).astype(np.uint8)
            p3[slice_pred > 0, 2] = np.clip(p3[slice_pred > 0, 2].astype(np.int32) + 120, 0, 255).astype(np.uint8)

            # Panel 4: Composite Error (Green=TP, Red=FP, Blue=FN)
            p4 = base_rgb.copy()
            tp = (slice_gt > 0) & (slice_pred > 0)
            fp = (slice_gt == 0) & (slice_pred > 0)
            fn = (slice_gt > 0) & (slice_pred == 0)

            p4[tp, 1] = 255  # Green for True Positive
            p4[fp, 0] = 255  # Red for False Positive
            p4[fn, 2] = 255  # Blue for False Negative

            grid_img = np.concatenate([p1, p2, p3, p4], axis=1)

            self.tracker.log_image(
                grid_img,
                name="val_slice_preview",
                step=epoch,
                caption=f"Epoch {epoch} Slice {slice_idx} (Image | GT | Pred | Error: TP-Green, FP-Red, FN-Blue)",
                save_local_dir=os.path.join(self.config.log_dir, "val_previews")
            )
        except Exception as e:
            print(f"  [Warning] Failed to generate validation slice preview: {e}")

    def train(self, num_epochs: int):
        end_epoch = self.start_epoch + num_epochs
        log_img_freq = getattr(self.config, 'log_image_every_n_epochs', 10)

        for epoch in range(self.start_epoch, end_epoch):
            if self.mode == 'prototype':
                train_loss, sup_loss, unsup_loss, uncert = self.train_one_epoch_uamt(epoch)
            else:
                train_loss, sup_loss, unsup_loss, uncert = self.train_one_epoch_cps(epoch)

            should_log_image = ((epoch + 1) % log_img_freq == 0) or (epoch == end_epoch - 1)
            val_metrics = self.validate(epoch=epoch, should_log_image=should_log_image)
            val_dice = val_metrics['val_dice']

            metrics = {
                'train_loss': train_loss,
                'supervised_loss': sup_loss,
                'unsupervised_loss': unsup_loss,
                'learning_rate': self.config.train_lr,
                'uncertainty_mean': uncert,
                'epoch': epoch,
                **val_metrics
            }
            self.tracker.log_metrics(metrics, step=epoch)

            hd95_str = f"{val_metrics['val_hd95']:.2f}mm" if not np.isnan(val_metrics['val_hd95']) else "N/A"

            print(f"  Epoch {epoch:3d}/{end_epoch} | "
                  f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | "
                  f"Prec: {val_metrics['val_precision']:.4f} | Rec: {val_metrics['val_recall']:.4f} | "
                  f"RVE: {val_metrics['val_rve_pct']:.1f}% | R²: {val_metrics['val_volume_r2']:.3f} | "
                  f"HD95: {hd95_str}")

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

    def resume(self, path: str):
        """Resume training from a full checkpoint (M-9 fix)."""
        if not os.path.exists(path):
            print(f"  No checkpoint found at {path}, training from scratch.")
            return
        self.load_checkpoint(path)
        print(f"  Resumed from checkpoint: {path} at epoch {self.start_epoch} (best_dice={self.best_dice:.4f})")

    def save_checkpoint(self, path: str, epoch: int = 0):
        """Save training checkpoint including current epoch (M-9 fix)."""
        state = {
            'net_A': self.net_A.state_dict(),
            'best_dice': self.best_dice,
            'epoch': epoch + 1,  # Resume from next epoch
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }
        if self.mode == 'prototype':
            state['teacher'] = self.teacher.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
        else:
            state['net_B'] = self.net_B.state_dict()
            state['optimizer_A'] = self.optimizer_A.state_dict()
            state['optimizer_B'] = self.optimizer_B.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        self.tracker.log_artifact(path, 'checkpoint')

    def load_checkpoint(self, path: str):
        """Load training checkpoint and restore start_epoch (M-9 fix)."""
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.net_A.load_state_dict(state['net_A'])
        self.best_dice = state.get('best_dice', 0.0)
        self.start_epoch = state.get('epoch', 0)  # Restore epoch numbering (M-9 fix)

        if self.mode == 'prototype':
            if 'teacher' in state:
                self.teacher.load_state_dict(state['teacher'])
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
        elif self.mode == 'full':
            if 'net_B' in state:
                self.net_B.load_state_dict(state['net_B'])
            if 'optimizer_A' in state:
                self.optimizer_A.load_state_dict(state['optimizer_A'])
            if 'optimizer_B' in state:
                self.optimizer_B.load_state_dict(state['optimizer_B'])

    def get_models(self):
        """Return trained models for active learning query."""
        if self.mode == 'prototype':
            return self.net_A, self.teacher.shadow
        else:
            return self.net_A, self.net_B
