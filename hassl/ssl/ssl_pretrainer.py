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

        # Task heads operating on bottleneck features (H-4 fix)
        self.rot_head = nn.LazyLinear(4).to(self.device)
        self.proj_head = nn.LazyLinear(getattr(config, 'ssl_embedding_dim', 128)).to(self.device)

        # Dry run to initialize LazyLinear heads on bottleneck feature dimensions (H-4 & Sub-Patch InfoNCE fix)
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, 32, 32, 32, device=self.device)
            dummy_bottleneck = self._extract_bottleneck_features(dummy_x)
            dummy_rot_feat = F.adaptive_avg_pool3d(dummy_bottleneck, 1).view(1, -1)
            dummy_patch_feat = F.adaptive_avg_pool3d(dummy_bottleneck, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, dummy_bottleneck.size(1))
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
            # Pass through UNet encoder layers up to bottleneck
            h = x
            for block in self.model.model:
                h = block(h)
                if h.shape[1] >= 128:  # Deepest bottleneck
                    break
            return h
        elif isinstance(self.model, DynUNet):
            # Pass through DynUNet input block + down blocks to deepest layer
            h = self.model.input_block(x)
            for down in self.model.downsamples:
                h = down(h)
            return self.model.bottleneck(h)
        else:
            out = self.model(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    def _apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random 3D sub-cube mask (30% volume masking)."""
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
        """Apply random 3D rotations for rotation prediction task."""
        b = x.size(0)
        rotated_x = x.clone()
        labels = torch.zeros(b, dtype=torch.long, device=x.device)
        for i in range(b):
            k = np.random.randint(0, 4)
            rotated_x[i] = torch.rot90(x[i], k, dims=(1, 2))
            labels[i] = k
        return rotated_x, labels

    def _infonce_loss(self, feat1: torch.Tensor, feat2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Spatial sub-patch InfoNCE contrastive loss supporting any batch size (B >= 1)."""
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        sim_matrix = torch.matmul(feat1, feat2.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=feat1.device)
        return self.ce_loss(sim_matrix, labels)

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
                    # 1. Masked Volume Inpainting (M-6 fix: compute loss only on masked voxels)
                    x_masked, mask = self._apply_masking(x)
                    out_inp = self.model(x_masked)
                    if isinstance(out_inp, (list, tuple)):
                        out_inp = out_inp[0]
                    elif out_inp.ndim == 6:
                        out_inp = out_inp[:, 0]

                    unmasked_count = (1 - mask).sum()
                    if unmasked_count > 0:
                        loss_inp = (self.mse_loss(out_inp * (1 - mask), x * (1 - mask)) * mask.numel()) / unmasked_count
                    else:
                        loss_inp = self.mse_loss(out_inp, x)

                    # 2 & 3. Rotation & Contrastive Learning via single batched bottleneck forward pass (GPU Speedup)
                    x_rot, rot_labels = self._apply_rotation(x)
                    x_aug1 = x + torch.randn_like(x) * 0.05
                    x_aug2 = x + torch.randn_like(x) * 0.05

                    B = x.size(0)
                    x_combined = torch.cat([x_rot, x_aug1, x_aug2], dim=0)
                    bottlenecks = self._extract_bottleneck_features(x_combined)

                    b_rot = bottlenecks[:B]
                    b1 = bottlenecks[B:2*B]
                    b2 = bottlenecks[2*B:]

                    rot_feats = F.adaptive_avg_pool3d(b_rot, 1).view(B, -1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)

                    # Extract 2x2x2 spatial sub-patch representations for InfoNCE contrastive pairs (8 patches per volume)
                    p1 = F.adaptive_avg_pool3d(b1, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b1.size(1))
                    p2 = F.adaptive_avg_pool3d(b2, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b2.size(1))

                    feat1 = self.proj_head(p1)
                    feat2 = self.proj_head(p2)
                    loss_cont = self._infonce_loss(feat1, feat2, temperature=getattr(self.config, 'ssl_contrastive_temp', 0.07))

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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_lr = self.optimizer.param_groups[0]['lr']
            if getattr(self, 'scheduler', None) is not None:
                self.scheduler.step()

            print(f"  [SSL Pre-train] Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} (Inp: {avg_inp:.4f}, Rot: {avg_rot:.4f}, Cont: {avg_cont:.4f}) | LR: {current_lr:.6f}")

            self.tracker.log_metrics({
                'ssl_loss': avg_loss,
                'ssl_loss_inpainting': avg_inp,
                'ssl_loss_rotation': avg_rot,
                'ssl_loss_contrastive': avg_cont,
                'ssl_learning_rate': current_lr,
            }, step=epoch)

            if should_log_image and first_batch_sample is not None:
                self.log_ssl_inpainting_samples(epoch, *first_batch_sample)

            if getattr(self, 'early_stopper', None) is not None and self.early_stopper(avg_loss):
                print(f"  [SSL Early Stopping] SSL loss did not improve for {self.early_stopper.patience} consecutive epochs. Early stopping at epoch {epoch + 1}.")
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
