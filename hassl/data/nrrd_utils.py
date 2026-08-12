import os
import nrrd
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def parse_nrrd_segments(header: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Parse 3D Slicer segmentation metadata from a NRRD header."""
    segments = {}

    for key, value in header.items():
        if key.startswith('Segment') and '_Name' in key:
            seg_prefix = key.split('_Name')[0]

            name = value
            try:
                label_value = int(header.get(f"{seg_prefix}_LabelValue", 1))
            except ValueError:
                label_value = 1

            color = header.get(f"{seg_prefix}_Color", "0.5 0.5 0.5")
            color_array = np.array([float(x) for x in color.split()])

            segments[label_value] = {
                'name': name,
                'label_value': label_value,
                'color': color_array
            }

    return segments


def load_seg_nrrd(filepath: str) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """Load a .seg.nrrd file and return the label map as an integer numpy array along with segment metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NRRD file not found: {filepath}")

    data, header = nrrd.read(filepath)
    segments = parse_nrrd_segments(header)
    return data.astype(np.int32), segments


def write_mask_with_spatial_geometry(output_path: str, mask_arr: np.ndarray, reference_image_path: Optional[str] = None):
    """Write segmentation mask preserving physical affine spatial metadata (origin, spacing, direction) (M-1 fix)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if sitk is not None and reference_image_path and os.path.exists(reference_image_path):
        try:
            ref_img = sitk.ReadImage(reference_image_path)
            # Squeeze extra dimensions (e.g. channel dim C=1) so mask_arr is strictly 3D (Z, Y, X) / (D, H, W)
            clean_arr = np.squeeze(mask_arr).astype(np.uint8)
            if clean_arr.ndim != 3:
                raise ValueError(f"Mask array must be 3D after squeeze, got shape {mask_arr.shape}")

            mask_img = sitk.GetImageFromArray(clean_arr)

            if mask_img.GetSize() != ref_img.GetSize():
                # Compute physical spacing for preprocessed mask
                ref_size = ref_img.GetSize()
                ref_spacing = ref_img.GetSpacing()
                mask_size = mask_img.GetSize()

                mask_spacing = tuple(
                    (ref_size[i] * ref_spacing[i]) / max(1, mask_size[i])
                    for i in range(3)
                )

                mask_img.SetSpacing(mask_spacing)
                mask_img.SetOrigin(ref_img.GetOrigin())
                mask_img.SetDirection(ref_img.GetDirection())

                # Resample back into native reference grid
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(sitk.Transform())
                mask_img = resampler.Execute(mask_img)

            mask_img.CopyInformation(ref_img)
            sitk.WriteImage(mask_img, output_path)
            return
        except Exception as e:
            print(f"[NRRD Utils] Warning: SimpleITK spatial write failed ({type(e).__name__}: {e}), falling back to pynrrd")

    # Fallback to pynrrd if SimpleITK fails
    header = {
        'type': 'uint8',
        'encoding': 'gzip',
        'space': 'left-posterior-superior',
        'Segment0_Name': 'Organ',
        'Segment0_LabelValue': '1',
        'Segment0_Color': '0.3 0.9 0.3',
    }
    nrrd.write(output_path, mask_arr.astype(np.uint8), header=header)
