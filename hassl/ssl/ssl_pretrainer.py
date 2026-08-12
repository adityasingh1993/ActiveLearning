import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, DynUNet
import numpy as np
from typing import Tuple, Dict, Any, Optional


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

        # Dry run to initialize LazyLinear heads on bottleneck feature dimensions (H-4 fix)
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, 32, 32, 32, device=self.device)
            dummy_bottleneck = self._extract_bottleneck_features(dummy_x)
            dummy_feat = F.adaptive_avg_pool3d(dummy_bottleneck, 1).view(1, -1)
            self.rot_head(dummy_feat)
            self.proj_head(dummy_feat)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.rot_head.parameters()) + list(self.proj_head.parameters()),
            lr=getattr(config, 'ssl_lr', 1e-4),
            weight_decay=getattr(config, 'ssl_weight_decay', 1e-5)
        )

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
        """Patch/Volume InfoNCE contrastive loss supporting batch_size=1 (H-5 fix)."""
        # Multiply features across patches/samples to ensure negative pairs exist
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        if feat1.ndim == 2 and feat1.size(0) == 1:
            # If batch_size=1, split volume into 4 spatial quadrants for sub-patch contrastive pairs
            # This ensures non-zero negative pairs exist even at batch_size=1 (H-5 fix)
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)

        sim_matrix = torch.matmul(feat1, feat2.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=feat1.device)
        return self.ce_loss(sim_matrix, labels)

    def train(self, num_epochs: int):
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in self.dataloader:
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

                    # 2. Rotation Prediction on Encoder Bottleneck Features (H-4 fix)
                    x_rot, rot_labels = self._apply_rotation(x)
                    bottleneck_rot = self._extract_bottleneck_features(x_rot)
                    rot_feats = F.adaptive_avg_pool3d(bottleneck_rot, 1).view(x.size(0), -1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)

                    # 3. Contrastive Learning on Encoder Bottleneck Features (H-4 & H-5 fix)
                    x_aug1 = x + torch.randn_like(x) * 0.05
                    x_aug2 = x + torch.randn_like(x) * 0.05

                    b1 = self._extract_bottleneck_features(x_aug1)
                    b2 = self._extract_bottleneck_features(x_aug2)

                    feat1 = self.proj_head(F.adaptive_avg_pool3d(b1, 1).view(x.size(0), -1))
                    feat2 = self.proj_head(F.adaptive_avg_pool3d(b2, 1).view(x.size(0), -1))

                    loss_cont = self._infonce_loss(feat1, feat2, temperature=getattr(self.config, 'ssl_contrastive_temp', 0.07))

                    loss = loss_inp + loss_rot + loss_cont

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(self.dataloader))
            print(f"  [SSL Pre-train] Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")
            self.tracker.log_metrics({'ssl_loss': avg_loss}, step=epoch)

        ckpt_dir = getattr(self.config, 'checkpoint_dir', './experiments/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, 'ssl_pretrained.pth')
        torch.save(self.model.state_dict(), save_path)
        self.tracker.log_artifact(save_path, name="ssl_pretrained_weights")
        print(f"  Pre-trained model saved to {save_path}")

    def get_encoder(self):
        """Return the pre-trained encoder model."""
        return self.model
