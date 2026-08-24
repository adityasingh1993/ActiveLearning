"""Exact single-object 3D CenterNet used by the Final91-QC bladder experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU(),
        )

    def forward(self, value):
        return self.block(value)


class CenterNet3D(nn.Module):
    """Center heatmap + sub-voxel offset + normalized 3D box size."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 24, stride=2)
        self.enc2 = ConvBlock(24, 48, stride=2)
        self.enc3 = ConvBlock(48, 96, stride=2)
        self.enc4 = ConvBlock(96, 192, stride=2)
        self.up3 = nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2)
        self.fuse3 = ConvBlock(192, 96)
        self.up2 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.fuse2 = ConvBlock(96, 64)
        self.head_features = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.heatmap_head = nn.Conv3d(64, 1, 1)
        self.offset_head = nn.Conv3d(64, 3, 1)
        self.size_head = nn.Conv3d(64, 3, 1)
        nn.init.constant_(self.heatmap_head.bias, -2.19)

    @staticmethod
    def coordinate_channels(image):
        batch, _, depth, height, width = image.shape
        z = torch.linspace(-1.0, 1.0, depth, device=image.device, dtype=image.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coordinates = torch.stack([zz, yy, xx], dim=0).unsqueeze(0)
        return coordinates.expand(batch, -1, -1, -1, -1)

    def forward(self, image):
        value = torch.cat([image, self.coordinate_channels(image)], dim=1)
        c1 = self.enc1(value)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        p3 = self.fuse3(torch.cat([self.up3(c4), c3], dim=1))
        p2 = self.fuse2(torch.cat([self.up2(p3), c2], dim=1))
        features = self.head_features(p2)
        return {
            "heatmap_logits": self.heatmap_head(features),
            "offset": torch.sigmoid(self.offset_head(features)),
            "size": 0.01 + 0.99 * torch.sigmoid(self.size_head(features)),
        }


def make_centernet_targets(center, size, output_shape):
    device, dtype = center.device, center.dtype
    shape = torch.as_tensor(output_shape, device=device, dtype=dtype)
    grid_center = center * torch.clamp(shape - 1.0, min=1.0)
    center_index = torch.floor(grid_center).long()
    upper = torch.as_tensor(output_shape, device=device, dtype=torch.long) - 1
    center_index = torch.minimum(torch.maximum(center_index, torch.zeros_like(center_index)), upper)
    offset = grid_center - center_index.to(dtype)
    heatmap = torch.zeros((center.shape[0], 1, *output_shape), device=device, dtype=dtype)
    zz = torch.arange(output_shape[0], device=device, dtype=dtype).view(-1, 1, 1)
    yy = torch.arange(output_shape[1], device=device, dtype=dtype).view(1, -1, 1)
    xx = torch.arange(output_shape[2], device=device, dtype=dtype).view(1, 1, -1)
    for index in range(center.shape[0]):
        sigma = torch.clamp(size[index] * shape / 6.0, min=1.0)
        gaussian = torch.exp(-0.5 * (
            ((zz - grid_center[index, 0]) / sigma[0]) ** 2
            + ((yy - grid_center[index, 1]) / sigma[1]) ** 2
            + ((xx - grid_center[index, 2]) / sigma[2]) ** 2
        ))
        heatmap[index, 0] = gaussian
        iz, iy, ix = (int(x) for x in center_index[index].tolist())
        heatmap[index, 0, iz, iy, ix] = 1.0
    return heatmap, center_index, offset, size


def gather_at(head, center_index):
    batch, channels, _, height, width = head.shape
    linear = (
        center_index[:, 0] * height * width
        + center_index[:, 1] * width
        + center_index[:, 2]
    )
    return head.reshape(batch, channels, -1).gather(
        2, linear.view(batch, 1, 1).expand(-1, channels, 1)
    ).squeeze(2)


def modified_focal_loss(logits, target):
    prediction = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    positive = target.eq(1.0).to(logits.dtype)
    negative = target.lt(1.0).to(logits.dtype)
    negative_weight = torch.pow(1.0 - target, 4.0)
    positive_loss = torch.log(prediction) * torch.pow(1.0 - prediction, 2.0) * positive
    negative_loss = (
        torch.log(1.0 - prediction) * torch.pow(prediction, 2.0)
        * negative_weight * negative
    )
    return -(positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1.0)


def box_iou(center_a, size_a, center_b, size_b):
    lo_a, hi_a = center_a - 0.5 * size_a, center_a + 0.5 * size_a
    lo_b, hi_b = center_b - 0.5 * size_b, center_b + 0.5 * size_b
    intersection = torch.relu(torch.minimum(hi_a, hi_b) - torch.maximum(lo_a, lo_b)).prod(dim=1)
    volume_a = torch.clamp(size_a, min=1e-6).prod(dim=1)
    volume_b = torch.clamp(size_b, min=1e-6).prod(dim=1)
    return intersection / torch.clamp(volume_a + volume_b - intersection, min=1e-6)


def decode_at_indices(outputs, center_index):
    output_shape = outputs["heatmap_logits"].shape[-3:]
    shape = torch.as_tensor(
        output_shape, device=outputs["heatmap_logits"].device,
        dtype=outputs["heatmap_logits"].dtype,
    )
    offset = gather_at(outputs["offset"], center_index)
    size = gather_at(outputs["size"], center_index)
    center = (center_index.to(offset.dtype) + offset) / torch.clamp(shape - 1.0, min=1.0)
    return center, size


def decode_peak(outputs):
    heatmap = torch.sigmoid(outputs["heatmap_logits"])
    batch, _, _, height, width = heatmap.shape
    flattened = heatmap.reshape(batch, -1)
    top_values, top_indices = torch.topk(flattened, k=min(2, flattened.shape[1]), dim=1)
    linear = top_indices[:, 0]
    center_index = torch.stack([
        linear // (height * width), (linear // width) % height, linear % width,
    ], dim=1)
    center, size = decode_at_indices(outputs, center_index)
    confidence = top_values[:, 0]
    peak_margin = top_values[:, 0] - top_values[:, 1] if top_values.shape[1] == 2 else confidence
    return center, size, confidence, peak_margin


def centernet_loss(outputs, target_center, target_size):
    heatmap_target, center_index, offset_target, size_target = make_centernet_targets(
        target_center, target_size, outputs["heatmap_logits"].shape[-3:]
    )
    heatmap_loss = modified_focal_loss(outputs["heatmap_logits"], heatmap_target)
    offset_prediction = gather_at(outputs["offset"], center_index)
    size_prediction = gather_at(outputs["size"], center_index)
    offset_loss = F.smooth_l1_loss(offset_prediction, offset_target)
    size_loss = F.smooth_l1_loss(size_prediction, size_target)
    predicted_center, _ = decode_at_indices(outputs, center_index)
    iou_loss = 1.0 - box_iou(
        predicted_center, size_prediction, target_center, target_size
    ).mean()
    total = heatmap_loss + offset_loss + size_loss + 2.0 * iou_loss
    return total, heatmap_loss, offset_loss, size_loss, iou_loss
