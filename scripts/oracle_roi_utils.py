#!/usr/bin/env python3
"""Utilities for GT-derived oracle ROI experiments.

These helpers are intentionally experiment-only. A ground-truth mask is used to define the
crop, so outputs must never be interpreted as deployable localization performance.
"""

from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("oracle ROI experiments require SimpleITK") from exc


def geometry_equal(a, b, atol=1e-6):
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=1e-6, atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=1e-6, atol=atol)
    )


def oracle_bbox_from_binary(mask_zyx, full_size_xyz, margin_fraction=0.40):
    """Return clipped SimpleITK ROI index/size from a non-empty ZYX binary mask.

    margin_fraction is added on EACH side of each target-axis extent. The default 0.40
    therefore creates an ideal crop approximately 1.8x the target bounding-box extent when
    image boundaries do not clip it.
    """
    mask = np.asarray(mask_zyx) > 0
    if mask.ndim != 3 or not mask.any():
        raise RuntimeError(f"Expected non-empty 3D binary mask, got shape={mask.shape}")
    if margin_fraction < 0:
        raise ValueError("margin_fraction must be >=0")

    zz, yy, xx = np.where(mask)
    mins_xyz = np.asarray([xx.min(), yy.min(), zz.min()], dtype=int)
    maxs_xyz = np.asarray([xx.max(), yy.max(), zz.max()], dtype=int)
    extent_xyz = maxs_xyz - mins_xyz + 1
    margin_xyz = np.ceil(extent_xyz.astype(float) * float(margin_fraction)).astype(int)

    full = np.asarray(full_size_xyz, dtype=int)
    start = np.maximum(0, mins_xyz - margin_xyz)
    stop = np.minimum(full, maxs_xyz + 1 + margin_xyz)
    size = stop - start
    if np.any(size <= 0):
        raise RuntimeError(f"Invalid oracle ROI start={start.tolist()} size={size.tolist()}")
    return tuple(int(x) for x in start), tuple(int(x) for x in size)


def make_oracle_roi(image_path, label_path, margin_fraction=0.40):
    image_path = Path(image_path)
    label_path = Path(label_path)
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if not geometry_equal(image, label):
        raise RuntimeError(
            "Image/label geometry mismatch for oracle ROI:\n"
            f"  image={image_path}\n  label={label_path}"
        )

    label_arr = np.asarray(sitk.GetArrayFromImage(label))
    binary = label_arr > 0
    if not binary.any():
        raise RuntimeError(f"Oracle label has no foreground: {label_path}")

    start_xyz, size_xyz = oracle_bbox_from_binary(
        binary, image.GetSize(), margin_fraction=margin_fraction
    )
    crop_image = sitk.RegionOfInterest(image, size=size_xyz, index=start_xyz)
    crop_label = sitk.RegionOfInterest(label, size=size_xyz, index=start_xyz)

    crop_binary = np.asarray(sitk.GetArrayFromImage(crop_label)) > 0
    if int(crop_binary.sum()) != int(binary.sum()):
        raise RuntimeError(
            f"Oracle crop lost foreground voxels for {image_path.name}: "
            f"full={int(binary.sum())}, crop={int(crop_binary.sum())}"
        )

    full_vox = int(np.prod(np.asarray(image.GetSize(), dtype=np.int64)))
    crop_vox = int(np.prod(np.asarray(size_xyz, dtype=np.int64)))
    fg_vox = int(binary.sum())
    return crop_image, crop_label, {
        "oracle_margin_fraction_each_side": float(margin_fraction),
        "roi_start_x": start_xyz[0],
        "roi_start_y": start_xyz[1],
        "roi_start_z": start_xyz[2],
        "roi_size_x": size_xyz[0],
        "roi_size_y": size_xyz[1],
        "roi_size_z": size_xyz[2],
        "full_size_x": int(image.GetSize()[0]),
        "full_size_y": int(image.GetSize()[1]),
        "full_size_z": int(image.GetSize()[2]),
        "gt_foreground_voxels": fg_vox,
        "full_gt_fraction": fg_vox / max(full_vox, 1),
        "roi_gt_fraction": fg_vox / max(crop_vox, 1),
        "roi_volume_fraction_of_full": crop_vox / max(full_vox, 1),
    }


def paste_crop_into_full(crop_zyx, full_shape_zyx, start_xyz, size_xyz):
    crop = np.asarray(crop_zyx)
    sx, sy, sz = [int(x) for x in size_xyz]
    x0, y0, z0 = [int(x) for x in start_xyz]
    expected = (sz, sy, sx)
    if tuple(crop.shape) != expected:
        raise RuntimeError(f"Crop shape mismatch: expected ZYX={expected}, got={tuple(crop.shape)}")
    full = np.zeros(tuple(int(x) for x in full_shape_zyx), dtype=crop.dtype)
    full[z0:z0 + sz, y0:y0 + sy, x0:x0 + sx] = crop
    return full
