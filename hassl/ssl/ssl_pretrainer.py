import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, DynUNet
from torch.cuda.amp import autocast, GradScaler
import numpy as np

class SSLPretrainer:
    def __init__(self, config, dataloader, tracker):
        self.config = config
        self.dataloader = dataloader
        self.tracker = tracker
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.unet_backbone == 'unet':
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=0.2
            )
        elif config.unet_backbone == 'dynunet':
            self.model = DynUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                kernel_size=[[3,3,3]]*5,
                strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                upsample_kernel_size=[[2,2,2]]*4,
                filters=[16, 32, 64, 128, 256],
                norm_name='instance',
                deep_supervision=True
            )
        else:
            raise ValueError(f"Unknown backbone: {config.unet_backbone}")
            
        self.model = self.model.to(self.device)
        
        # Heads for SSL tasks
        self.rot_head = nn.Linear(1, 4).to(self.device)
        self.proj_head = nn.Linear(1, 128).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.rot_head.parameters()) + list(self.proj_head.parameters()), 
            lr=1e-4
        )
        self.scaler = GradScaler()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def _apply_masking(self, x):
        mask = torch.ones_like(x)
        b, c, d, h, w = x.shape
        num_cubes_d, num_cubes_h, num_cubes_w = d // 16, h // 16, w // 16
        total_cubes = num_cubes_d * num_cubes_h * num_cubes_w
        num_masked = int(0.3 * total_cubes)
        
        for i in range(b):
            indices = np.random.choice(total_cubes, num_masked, replace=False)
            for idx in indices:
                dz = (idx // (num_cubes_h * num_cubes_w)) * 16
                dy = ((idx // num_cubes_w) % num_cubes_h) * 16
                dx = (idx % num_cubes_w) * 16
                mask[i, :, dz:dz+16, dy:dy+16, dx:dx+16] = 0
        return x * mask, mask

    def _apply_rotation(self, x):
        b = x.size(0)
        rotated_x = x.clone()
        labels = torch.zeros(b, dtype=torch.long, device=x.device)
        for i in range(b):
            k = np.random.randint(0, 4)
            axis = tuple(np.random.choice([2, 3, 4], 2, replace=False))
            rotated_x[i] = torch.rot90(x[i], k, dims=(axis[0]-1, axis[1]-1))
            labels[i] = k
        return rotated_x, labels
        
    def _infonce_loss(self, feat1, feat2, temperature=0.1):
        b = feat1.size(0)
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        logits = torch.matmul(feat1, feat2.T) / temperature
        labels = torch.arange(b, device=feat1.device)
        return self.ce_loss(logits, labels)
        
    def train(self, num_epochs):
        # AMP + Gradient Checkpointing
        # Note: setting checkpointing if supported
        for name, module in self.model.named_modules():
            if hasattr(module, 'checkpointing'):
                module.checkpointing = True

        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in self.dataloader:
                x = batch['image'].to(self.device)
                self.optimizer.zero_grad()
                
                with autocast():
                    # 1. Masked Volume Inpainting
                    x_masked, mask = self._apply_masking(x)
                    out_inp = self.model(x_masked)
                    if isinstance(out_inp, list) or isinstance(out_inp, tuple):
                        out_inp = out_inp[0]
                    loss_inp = self.mse_loss(out_inp * (1-mask), x * (1-mask))
                    
                    # 2. Rotation Prediction
                    x_rot, rot_labels = self._apply_rotation(x)
                    out_rot = self.model(x_rot)
                    if isinstance(out_rot, list) or isinstance(out_rot, tuple):
                        out_rot = out_rot[0]
                    rot_feats = F.adaptive_avg_pool3d(out_rot, 1).view(x.size(0), -1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)
                    
                    # 3. Contrastive Learning (dummy augmentations for demo: add noise)
                    x_aug1 = x + torch.randn_like(x) * 0.1
                    x_aug2 = x + torch.randn_like(x) * 0.1
                    out1 = self.model(x_aug1)
                    out2 = self.model(x_aug2)
                    if isinstance(out1, list) or isinstance(out1, tuple):
                        out1 = out1[0]
                        out2 = out2[0]
                    feat1 = self.proj_head(F.adaptive_avg_pool3d(out1, 1).view(x.size(0), -1))
                    feat2 = self.proj_head(F.adaptive_avg_pool3d(out2, 1).view(x.size(0), -1))
                    loss_cont = self._infonce_loss(feat1, feat2)
                    
                    loss = loss_inp + loss_rot + loss_cont
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(self.dataloader)
            self.tracker.log({'ssl_loss': avg_loss, 'epoch': epoch})
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
