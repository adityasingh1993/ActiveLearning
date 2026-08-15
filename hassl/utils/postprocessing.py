"""
HASSL Post-Processing Module.

Connected component filtering and morphological post-processing utilities for 3D medical volume segmentation.
"""

import numpy as np
import scipy.ndimage as ndi
import torch
from typing import List, Optional, Union


def apply_keep_largest_cc(
    pred_tensor: torch.Tensor,
    n_components: int = 1,
    min_size_voxels: int = 0,
    target_class: Optional[int] = None,
    threshold: float = 0.5,
    fill_holes: bool = True,
) -> torch.Tensor:
    """Keep top N (default n_components=1) largest 3D connected components across foreground classes and optionally fill 3D holes.

    Ranks 3D connected components by voxel volume and retains ONLY the top `n_components`
    largest components, zeroing out all smaller satellite fragments. If fill_holes=True,
    applies 3D binary hole filling (scipy.ndimage.binary_fill_holes) to eliminate internal
    false-negative voids inside solid organs.

    Args:
        pred_tensor: Prediction tensor [B, C, D, H, W] or integer label map [B, D, H, W].
        n_components: Number of largest connected components to keep (default 1).
        min_size_voxels: Optional minimum voxel size threshold (default 0 = no size cutoff).
        target_class: Specific class label/channel index to filter. None = filter all foreground classes (default None).
        threshold: Binarization probability threshold for soft activations (default 0.5).
        fill_holes: Whether to apply 3D binary hole filling inside kept organ masks (default True).

    Returns:
        Tensor of exact same shape, type, and device as pred_tensor with only the top n_components
        largest connected components retained per foreground class.
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
                # Binary single-channel: channel 0 is foreground
                mask = (arr[b, 0] == 1) if is_discrete else (arr[b, 0] > threshold)
                if mask.any():
                    labeled_arr, num = ndi.label(mask)
                    if num > 0:
                        counts = np.bincount(labeled_arr.ravel())
                        counts[0] = 0  # ignore background 0
                        sorted_labels = np.argsort(counts)[::-1]
                        top_labels = [
                            lbl for lbl in sorted_labels[:n_components]
                            if counts[lbl] >= min_size_voxels and counts[lbl] > 0
                        ]
                        if top_labels:
                            keep_mask = np.isin(labeled_arr, top_labels)
                            if fill_holes:
                                keep_mask = ndi.binary_fill_holes(keep_mask)
                            out[b, 0] = np.where(keep_mask, arr[b, 0] if not is_discrete else 1, 0).astype(arr.dtype)
            else:
                # Multi-channel: filter non-background channels (c >= 1) or target_class
                for c in range(num_channels):
                    if c == 0 and target_class != 0:
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
                            sorted_labels = np.argsort(counts)[::-1]
                            top_labels = [
                                lbl for lbl in sorted_labels[:n_components]
                                if counts[lbl] >= min_size_voxels and counts[lbl] > 0
                            ]
                            if top_labels:
                                keep_mask = np.isin(labeled_arr, top_labels)
                                if fill_holes:
                                    keep_mask = ndi.binary_fill_holes(keep_mask)
                                out[b, c] = np.where(keep_mask, arr[b, c] if not is_discrete else 1, 0).astype(arr.dtype)

        elif arr.ndim == 4:
            # Shape [B, D, H, W] discrete integer label map
            out[b] = arr[b].copy()  # preserve background and other classes by default
            classes_to_filter = [target_class] if target_class is not None else np.unique(arr[b])[np.unique(arr[b]) != 0]
            for cls in classes_to_filter:
                mask = arr[b] == cls
                if mask.any():
                    labeled_arr, num = ndi.label(mask)
                    if num > 0:
                        counts = np.bincount(labeled_arr.ravel())
                        counts[0] = 0
                        sorted_labels = np.argsort(counts)[::-1]
                        top_labels = [
                            lbl for lbl in sorted_labels[:n_components]
                            if counts[lbl] >= min_size_voxels and counts[lbl] > 0
                        ]
                        out[b][mask] = 0  # clear all voxels of this class first
                        if top_labels:
                            keep_mask = np.isin(labeled_arr, top_labels)
                            if fill_holes:
                                keep_mask = ndi.binary_fill_holes(keep_mask)
                            out[b][keep_mask] = cls

    return torch.from_numpy(out).to(device)
