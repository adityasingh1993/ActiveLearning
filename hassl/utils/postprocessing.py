"""
HASSL Post-Processing Module.

Connected component filtering and morphological post-processing utilities for 3D medical volume segmentation.
"""

import numpy as np
import scipy.ndimage as ndi
import torch
from typing import Optional


def apply_keep_largest_cc(
    pred_tensor: torch.Tensor,
    min_size_voxels: int = 100,
    target_class: int = 1,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Keep only the single largest 3D connected component for Component 1 (target foreground).

    Identifies 3D connected components for class/component 1 (or foreground channel),
    retains ONLY the largest connected component by voxel count, and zeroes out all smaller
    satellite noise blobs (disconnected false positive fragments).

    Args:
        pred_tensor: Prediction tensor [B, C, D, H, W] or integer label map [B, D, H, W].
        min_size_voxels: Minimum voxel count for the largest component to be retained.
        target_class: Target component label/class index to filter (default 1).
        threshold: Binarization probability threshold for soft activations (default 0.5).

    Returns:
        Tensor of exact same shape, type, and device as pred_tensor, with only the largest
        connected component retained for component 1.
    """
    device = pred_tensor.device
    arr = pred_tensor.detach().cpu().numpy()
    out = np.zeros_like(arr)

    is_discrete = np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0

    for b in range(arr.shape[0]):
        if arr.ndim == 5:
            # Shape [B, C, D, H, W]
            num_channels = arr.shape[1]
            if num_channels == 1:
                # Binary single-channel: channel 0 is Component 1 (target foreground)
                mask = (arr[b, 0] == target_class) if is_discrete else (arr[b, 0] > threshold)
                if mask.any():
                    labeled_arr, num = ndi.label(mask)
                    if num > 0:
                        counts = np.bincount(labeled_arr.ravel())
                        counts[0] = 0  # exclude background 0
                        if counts.max() > 0:
                            max_lab = counts.argmax()
                            largest_size = counts[max_lab]
                            if largest_size >= min_size_voxels:
                                out[b, 0] = (labeled_arr == max_lab).astype(arr.dtype)
            else:
                # Multi-channel: apply to channel target_class (or all non-bg channels if target_class=None)
                for c in range(num_channels):
                    if c == 0:
                        out[b, 0] = arr[b, 0]
                        continue
                    if target_class is not None and c != target_class:
                        out[b, c] = arr[b, c]
                        continue
                    mask = (arr[b, c] == 1) if is_discrete else (arr[b, c] > threshold)
                    if mask.any():
                        labeled_arr, num = ndi.label(mask)
                        if num > 0:
                            counts = np.bincount(labeled_arr.ravel())
                            counts[0] = 0
                            if counts.max() > 0:
                                max_lab = counts.argmax()
                                largest_size = counts[max_lab]
                                if largest_size >= min_size_voxels:
                                    out[b, c] = (labeled_arr == max_lab).astype(arr.dtype)

        elif arr.ndim == 4:
            # Shape [B, D, H, W] discrete integer label map
            out[b] = arr[b].copy()  # preserve other label classes
            mask = arr[b] == target_class
            if mask.any():
                labeled_arr, num = ndi.label(mask)
                if num > 0:
                    counts = np.bincount(labeled_arr.ravel())
                    counts[0] = 0
                    if counts.max() > 0:
                        max_lab = counts.argmax()
                        largest_size = counts[max_lab]
                        out[b][mask] = 0  # zero out component 1 voxels first
                        if largest_size >= min_size_voxels:
                            out[b][labeled_arr == max_lab] = target_class

    return torch.from_numpy(out).to(device)
