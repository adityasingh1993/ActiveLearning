import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, DynUNet
import numpy as np
from typing import Tuple


class SSLPretrainer:
    """Self-Supervised Pre-trainer utilizing Masked Volume Inpainting, Rotation Prediction, and Patch-based InfoNCE."""

    def __init__(self, config, dataloader, tracker):
        self.config = config
        self.dataloader = dataloader
        self.tracker = tracker
        self.device_type = 'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
        self.device = torch.device(self.device_type)

        out_channels = getattr(config, 'output_channels', 1)

        if config.unet_backbone == 'unet':
            self.model = UNet(
                spatial_dims=3, in_channels=1, out_channels=out_channels,
                channels=getattr(config, 'unet_channels', (16, 32, 64, 128, 256)),
                strides=getattr(config, 'unet_strides', (2, 2, 2, 2)),
                num_res_units=2, dropout=config.dropout
            )
        elif config.unet_backbone == 'dynunet':
            self.model = DynUNet(
                spatial_dims=3, in_channels=1, out_channels=out_channels,
                kernel_size=[[3, 3, 3]] * 5,
                strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                upsample_kernel_size=[[2, 2, 2]] * 4,
                filters=[16, 32, 64, 128, 256],
                norm_name='instance',
                deep_supervision=config.dynunet_deep_supervision
            )
        else:
            raise ValueError(f"Unknown backbone: {config.unet_backbone}")

        self.model = self.model.to(self.device)

        # The controlled Final62 SSL experiment uses 4^3=64 spatial tokens instead of 2^3=8.
        # This keeps the contrastive task non-trivial for batch_size=1 and is configurable for
        # future experiments without changing the task implementation.
        self.contrastive_grid_size = int(getattr(config, 'ssl_contrastive_grid_size', 4))
        if self.contrastive_grid_size < 2:
            raise ValueError("ssl_contrastive_grid_size must be >= 2")

        # Mild, geometry-preserving ultrasound intensity perturbations. The two contrastive
        # views are generated independently while retaining voxel-to-voxel spatial alignment,
        # so token i remains the positive pair for token i.
        self.contrastive_gamma_min = float(getattr(config, 'ssl_contrastive_gamma_min', 0.8))
        self.contrastive_gamma_max = float(getattr(config, 'ssl_contrastive_gamma_max', 1.2))
        self.contrastive_scale_min = float(getattr(config, 'ssl_contrastive_scale_min', 0.9))
        self.contrastive_scale_max = float(getattr(config, 'ssl_contrastive_scale_max', 1.1))
        self.contrastive_shift_abs = float(getattr(config, 'ssl_contrastive_shift_abs', 0.05))
        self.contrastive_noise_std_max = float(getattr(config, 'ssl_contrastive_noise_std_max', 0.05))

        if self.contrastive_gamma_min <= 0 or self.contrastive_gamma_max < self.contrastive_gamma_min:
            raise ValueError("Invalid SSL contrastive gamma range")
        if self.contrastive_scale_min <= 0 or self.contrastive_scale_max < self.contrastive_scale_min:
            raise ValueError("Invalid SSL contrastive intensity-scale range")
        if self.contrastive_shift_abs < 0 or self.contrastive_noise_std_max < 0:
            raise ValueError("SSL contrastive shift/noise settings must be non-negative")

        # Task heads operating on bottleneck features (H-4 fix)
        self.rot_head = nn.LazyLinear(4).to(self.device)
        self.proj_head = nn.LazyLinear(getattr(config, 'ssl_embedding_dim', 128)).to(self.device)

        # Dry run to initialize LazyLinear heads on bottleneck feature dimensions. 64^3 is used
        # here so a 4^3 token grid is represented naturally at the deepest DynUNet feature map.
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, 64, 64, 64, device=self.device)
            dummy_bottleneck = self._extract_bottleneck_features(dummy_x)
            dummy_rot_feat = F.adaptive_avg_pool3d(dummy_bottleneck, 1).view(1, -1)
            g = self.contrastive_grid_size
            dummy_patch_feat = (
                F.adaptive_avg_pool3d(dummy_bottleneck, (g, g, g))
                .permute(0, 2, 3, 4, 1)
                .reshape(-1, dummy_bottleneck.size(1))
            )
            self.rot_head(dummy_rot_feat)
            self.proj_head(dummy_patch_feat)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.rot_head.parameters()) + list(self.proj_head.parameters()),
            lr=getattr(config, 'ssl_lr', 1e-4),
            weight_decay=getattr(config, 'ssl_weight_decay', 1e-5)
        )

        ssl_epochs = getattr(config, 'ssl_epochs', 100)
        min_lr = getattr(config, 'min_lr', 1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=ssl_epochs, eta_min=min_lr
        )

        from ..training.trainer import EarlyStopping
        use_es = getattr(config, 'ssl_use_early_stopping', True)
        patience = getattr(config, 'ssl_early_stopping_patience', 20)
        self.early_stopper = EarlyStopping(patience=patience, min_delta=1e-4, mode='min') if use_es else None

        if self.device_type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def _extract_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deep bottleneck features from encoder (H-4 fix)."""
        if isinstance(self.model, UNet):
            h = x
            for block in self.model.model:
                h = block(h)
                if h.shape[1] >= 128:
                    break
            return h
        elif isinstance(self.model, DynUNet):
            h = self.model.input_block(x)
            for down in self.model.downsamples:
                h = down(h)
            return self.model.bottleneck(h)
        else:
            out = self.model(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    def _apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random 3D sub-cube mask (30% volume masking by default)."""
        mask = torch.ones_like(x)
        b, c, d, h, w = x.shape
        cube_sz = getattr(self.config, 'ssl_mask_cube_size', 16)

        num_cubes_d = max(1, d // cube_sz)
        num_cubes_h = max(1, h // cube_sz)
        num_cubes_w = max(1, w // cube_sz)
        total_cubes = num_cubes_d * num_cubes_h * num_cubes_w
        num_masked = max(1, int(getattr(self.config, 'ssl_mask_ratio', 0.3) * total_cubes))

        for i in range(b):
            indices = np.random.choice(total_cubes, min(num_masked, total_cubes), replace=False)
            for idx in indices:
                dz = (idx // (num_cubes_h * num_cubes_w)) * cube_sz
                dy = ((idx // num_cubes_w) % num_cubes_h) * cube_sz
                dx = (idx % num_cubes_w) * cube_sz
                mask[i, :, min(dz, d - 1):min(dz + cube_sz, d),
                     min(dy, h - 1):min(dy + cube_sz, h),
                     min(dx, w - 1):min(dx + cube_sz, w)] = 0
        return x * mask, mask

    def _apply_rotation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random 90-degree in-plane rotations for rotation prediction."""
        b = x.size(0)
        rotated_x = x.clone()
        labels = torch.zeros(b, dtype=torch.long, device=x.device)
        for i in range(b):
            k = np.random.randint(0, 4)
            rotated_x[i] = torch.rot90(x[i], k, dims=(1, 2))
            labels[i] = k
        return rotated_x, labels

    def _make_contrastive_view(self, x: torch.Tensor) -> torch.Tensor:
        """Create an independently perturbed but spatially aligned ultrasound view.

        Spatial transforms are intentionally excluded here because InfoNCE positives are paired
        by spatial token index. Gamma, scale, shift, and noise make the appearance different
        without invalidating that correspondence.
        """
        b = x.size(0)
        param_shape = (b, 1, 1, 1, 1)
        dtype = x.dtype
        device = x.device

        gamma = torch.empty(param_shape, device=device, dtype=dtype).uniform_(
            self.contrastive_gamma_min, self.contrastive_gamma_max
        )
        scale = torch.empty(param_shape, device=device, dtype=dtype).uniform_(
            self.contrastive_scale_min, self.contrastive_scale_max
        )
        shift = torch.empty(param_shape, device=device, dtype=dtype).uniform_(
            -self.contrastive_shift_abs, self.contrastive_shift_abs
        )
        noise_std = torch.empty(param_shape, device=device, dtype=dtype).uniform_(
            0.0, self.contrastive_noise_std_max
        )

        view = torch.clamp(x, 0.0, 1.0).pow(gamma)
        view = view * scale + shift
        if self.contrastive_noise_std_max > 0:
            view = view + torch.randn_like(view) * noise_std
        return torch.clamp(view, 0.0, 1.0)

    def _infonce_loss(self, feat1: torch.Tensor, feat2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Symmetric spatial-token InfoNCE supporting batch size one."""
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        sim_matrix = torch.matmul(feat1, feat2.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=feat1.device)
        loss_12 = self.ce_loss(sim_matrix, labels)
        loss_21 = self.ce_loss(sim_matrix.T, labels)
        return 0.5 * (loss_12 + loss_21)

    def train(self, num_epochs: int):
        log_img_freq = getattr(self.config, 'log_image_every_n_epochs', 10)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_inp = 0.0
            epoch_rot = 0.0
            epoch_cont = 0.0
            first_batch_sample = None

            should_log_image = ((epoch + 1) % log_img_freq == 0) or (epoch == num_epochs - 1) or (epoch == 0)

            for batch_idx, batch in enumerate(self.dataloader):
                x = batch['image'].to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device_type, enabled=(self.device_type == 'cuda')):
                    # 1. Masked Volume Inpainting: compute MSE only on masked voxels.
                    x_masked, mask = self._apply_masking(x)
                    out_inp = self.model(x_masked)
                    if isinstance(out_inp, (list, tuple)):
                        out_inp = out_inp[0]
                    elif out_inp.ndim == 6:
                        out_inp = out_inp[:, 0]

                    masked_count = (1 - mask).sum()
                    if masked_count > 0:
                        loss_inp = (self.mse_loss(out_inp * (1 - mask), x * (1 - mask)) * mask.numel()) / masked_count
                    else:
                        loss_inp = self.mse_loss(out_inp, x)

                    # 2 & 3. Rotation + stronger spatial-token contrastive learning.
                    x_rot, rot_labels = self._apply_rotation(x)
                    x_aug1 = self._make_contrastive_view(x)
                    x_aug2 = self._make_contrastive_view(x)

                    B = x.size(0)
                    x_combined = torch.cat([x_rot, x_aug1, x_aug2], dim=0)
                    bottlenecks = self._extract_bottleneck_features(x_combined)

                    b_rot = bottlenecks[:B]
                    b1 = bottlenecks[B:2 * B]
                    b2 = bottlenecks[2 * B:]

                    rot_feats = F.adaptive_avg_pool3d(b_rot, 1).view(B, -1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)

                    # 4x4x4 by default = 64 spatial tokens per volume rather than 8.
                    g = self.contrastive_grid_size
                    p1 = (
                        F.adaptive_avg_pool3d(b1, (g, g, g))
                        .permute(0, 2, 3, 4, 1)
                        .reshape(-1, b1.size(1))
                    )
                    p2 = (
                        F.adaptive_avg_pool3d(b2, (g, g, g))
                        .permute(0, 2, 3, 4, 1)
                        .reshape(-1, b2.size(1))
                    )

                    feat1 = self.proj_head(p1)
                    feat2 = self.proj_head(p2)
                    loss_cont = self._infonce_loss(
                        feat1,
                        feat2,
                        temperature=getattr(self.config, 'ssl_contrastive_temp', 0.07),
                    )

                    # Equal weights remain frozen for the controlled experiment.
                    loss = loss_inp + loss_rot + loss_cont

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_inp += loss_inp.item()
                epoch_rot += loss_rot.item()
                epoch_cont += loss_cont.item()

                if first_batch_sample is None and should_log_image:
                    first_batch_sample = (
                        x[0].detach().cpu(),
                        x_masked[0].detach().cpu(),
                        out_inp[0].detach().cpu()
                    )

            N = max(1, len(self.dataloader))
            avg_loss = epoch_loss / N
            avg_inp = epoch_inp / N
            avg_rot = epoch_rot / N
            avg_cont = epoch_cont / N

            contribution_total = max(avg_inp + avg_rot + avg_cont, 1e-12)
            inp_pct = 100.0 * avg_inp / contribution_total
            rot_pct = 100.0 * avg_rot / contribution_total
            cont_pct = 100.0 * avg_cont / contribution_total

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_lr = self.optimizer.param_groups[0]['lr']
            if getattr(self, 'scheduler', None) is not None:
                self.scheduler.step()

            print(
                f"  [SSL Pre-train] Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} (Inp: {avg_inp:.4f}, Rot: {avg_rot:.4f}, Cont: {avg_cont:.4f}) | "
                f"Mix: I={inp_pct:.1f}% R={rot_pct:.1f}% C={cont_pct:.1f}% | "
                f"LR: {current_lr:.6f}"
            )

            self.tracker.log_metrics({
                'ssl_loss': avg_loss,
                'ssl_loss_inpainting': avg_inp,
                'ssl_loss_rotation': avg_rot,
                'ssl_loss_contrastive': avg_cont,
                'ssl_contribution_inpainting_pct': inp_pct,
                'ssl_contribution_rotation_pct': rot_pct,
                'ssl_contribution_contrastive_pct': cont_pct,
                'ssl_contrastive_grid_size': float(self.contrastive_grid_size),
                'ssl_contrastive_tokens_per_volume': float(self.contrastive_grid_size ** 3),
                'ssl_learning_rate': current_lr,
            }, step=epoch)

            if should_log_image and first_batch_sample is not None:
                self.log_ssl_inpainting_samples(epoch, *first_batch_sample)

            if getattr(self, 'early_stopper', None) is not None and self.early_stopper(avg_loss):
                print(
                    f"  [SSL Early Stopping] SSL loss did not improve for "
                    f"{self.early_stopper.patience} consecutive epochs. "
                    f"Early stopping at epoch {epoch + 1}."
                )
                break

        ckpt_dir = getattr(self.config, 'checkpoint_dir', './experiments/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, 'ssl_pretrained.pth')
        torch.save(self.model.state_dict(), save_path)
        self.tracker.log_artifact(save_path, name="ssl_pretrained_weights")
        print(f"  Pre-trained model saved to {save_path}")

    def log_ssl_inpainting_samples(self, epoch: int, orig_t: torch.Tensor, masked_t: torch.Tensor, inp_t: torch.Tensor):
        """Generate and log 3-panel SSL inpainting preview grid (Original | Masked | Inpainted)."""
        try:
            import numpy as np
            orig_np = orig_t[0].numpy()
            masked_np = masked_t[0].numpy()
            inp_np = inp_t[0].numpy()

            slice_idx = orig_np.shape[0] // 2

            s_orig = orig_np[slice_idx]
            s_masked = masked_np[slice_idx]
            s_inp = inp_np[slice_idx]

            def norm_uint8(arr):
                mn, mx = arr.min(), arr.max()
                norm = (arr - mn) / (mx - mn + 1e-8) * 255.0
                return norm.astype(np.uint8)

            p1_gray = norm_uint8(s_orig)
            p2_gray = norm_uint8(s_masked)
            p3_gray = norm_uint8(s_inp)

            p1 = np.stack([p1_gray] * 3, axis=-1)
            p2 = np.stack([p2_gray] * 3, axis=-1)
            p3 = np.stack([p3_gray] * 3, axis=-1)

            grid_img = np.concatenate([p1, p2, p3], axis=1)
            log_dir = getattr(self.config, 'log_dir', './experiments/logs')

            self.tracker.log_image(
                grid_img,
                name="ssl_inpainting_preview",
                step=epoch,
                caption=f"SSL Epoch {epoch} Inpainting Slice {slice_idx} (Original | 30% Masked | Model Inpainting)",
                save_local_dir=os.path.join(log_dir, "ssl_previews")
            )
        except Exception as e:
            print(f"  [Warning] Failed to generate SSL inpainting preview image: {e}")

    def get_encoder(self):
        """Return the pre-trained encoder model."""
        return self.model
