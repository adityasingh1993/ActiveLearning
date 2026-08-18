#!/usr/bin/env python3
"""Shared residual guided-refiner model utilities.

The residual refiner receives two channels:
  0: ultrasound ROI
  1: coarse EMA probability ROI

Instead of predicting a segmentation from scratch, the DynUNet predicts a logit correction
(delta). The final model logit is:

    corrected_logit = logit(coarse_probability) + delta_logit

All segmentation output heads in the delta DynUNet are zero-initialized. Therefore the model
starts by reproducing the coarse probability exactly (up to clipping) and must learn only
corrections supported by the ROI image/guidance pair.
"""

from pathlib import Path

import torch
from torch import nn
from monai.networks.nets import DynUNet


COARSE_EPS = 1e-4


class ResidualGuidedDynUNet(nn.Module):
    """Two-channel DynUNet that learns a residual correction to coarse EMA logits."""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        if int(num_classes) != 1:
            raise RuntimeError("Residual guided feasibility experiment currently supports binary segmentation only")

        self.delta_net = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=int(num_classes),
            kernel_size=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 4,
            filters=[16, 32, 64, 128, 256],
            dropout=float(dropout),
            norm_name="instance",
            deep_supervision=True,
        )
        self._zero_initialize_delta_heads()

    def _zero_initialize_delta_heads(self):
        """Make the initial corrected output equal to the supplied coarse probability.

        DynUNet's main/deep-supervision segmentation heads are 1x1x1 Conv3d layers with
        out_channels equal to the requested segmentation channels. Zeroing all such heads keeps
        the residual branch at delta=0 on the first forward pass while preserving normal gradient
        flow into the output heads immediately.
        """
        n_zeroed = 0
        for module in self.delta_net.modules():
            if (
                isinstance(module, nn.Conv3d)
                and int(module.out_channels) == 1
                and tuple(int(x) for x in module.kernel_size) == (1, 1, 1)
            ):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                n_zeroed += 1
        if n_zeroed < 1:
            raise RuntimeError("Could not identify DynUNet segmentation head(s) for zero residual initialization")
        self.n_zero_initialized_heads = int(n_zeroed)

    @staticmethod
    def _add_coarse_logit(delta, coarse_logit):
        if isinstance(delta, (list, tuple)):
            return type(delta)(ResidualGuidedDynUNet._add_coarse_logit(x, coarse_logit) for x in delta)
        if not torch.is_tensor(delta):
            raise TypeError(f"Unexpected residual DynUNet output type: {type(delta)}")
        if delta.ndim == 6:
            # MONAI DynUNet deep-supervision training output: [B, N, C, D, H, W].
            return delta + coarse_logit.unsqueeze(1)
        if delta.ndim == 5:
            return delta + coarse_logit
        raise RuntimeError(f"Unexpected residual DynUNet output shape: {tuple(delta.shape)}")

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 2:
            raise RuntimeError(f"Residual guided refiner expects [B,2,D,H,W], got {tuple(x.shape)}")
        coarse_prob = torch.clamp(x[:, 1:2], min=COARSE_EPS, max=1.0 - COARSE_EPS)
        coarse_logit = torch.logit(coarse_prob)
        delta = self.delta_net(x)
        return self._add_coarse_logit(delta, coarse_logit)


def build_residual_guided_dynunet(backbone, num_classes, dropout):
    if str(backbone) != "dynunet":
        raise RuntimeError("Residual guided feasibility experiment is frozen to DynUNet")
    return ResidualGuidedDynUNet(num_classes=int(num_classes), dropout=float(dropout))


def load_residual_teacher(config, checkpoint: Path, device):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "teacher" not in state:
        raise RuntimeError(f"Residual guided checkpoint has no EMA teacher: {checkpoint}")
    teacher = build_residual_guided_dynunet(
        config.unet_backbone,
        config.num_classes,
        config.dropout,
    ).to(device)
    teacher.load_state_dict(state["teacher"])
    teacher.eval()
    return teacher
