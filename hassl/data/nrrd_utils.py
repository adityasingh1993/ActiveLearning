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
    return data.astype(np.uint8), segments


def _slicer_segmentation_metadata(
    size_xyz,
    segment_name: str = "Bladder",
    segment_id: Optional[str] = None,
    label_value: int = 1,
    segment_color: str = "0.0 1.0 0.0",
) -> Dict[str, str]:
    """Return the 3D Slicer .seg.nrrd key/value metadata required by downstream apps."""
    segment_id = segment_id or segment_name
    sx, sy, sz = [int(v) for v in size_xyz]

    # Keep the labelmap on the full native reference extent. This matches the way HASSL writes
    # native-grid predictions and avoids a cropped-layer offset being interpreted by consumers.
    extent = f"0 {max(sx - 1, 0)} 0 {max(sy - 1, 0)} 0 {max(sz - 1, 0)}"

    return {
        "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ReferenceImageExtentOffset": "0 0 0",
        "Segment0_ID": str(segment_id),
        "Segment0_LabelValue": str(int(label_value)),
        "Segment0_Layer": "0",
        "Segment0_Color": str(segment_color),
        "Segment0_Name": str(segment_name),
        "Segment0_Extent": extent,
        "Segment0_Tags": "|",
    }


def _set_slicer_metadata(
    image,
    segment_name: str = "Bladder",
    segment_id: Optional[str] = None,
    label_value: int = 1,
    segment_color: str = "0.0 1.0 0.0",
):
    """Attach Slicer segmentation metadata to a SimpleITK image before NRRD writing."""
    metadata = _slicer_segmentation_metadata(
        image.GetSize(),
        segment_name=segment_name,
        segment_id=segment_id,
        label_value=label_value,
        segment_color=segment_color,
    )
    for key, value in metadata.items():
        image.SetMetaData(key, value)


def write_mask_with_spatial_geometry(
    output_path: str,
    mask_arr: np.ndarray,
    reference_image_path: Optional[str] = None,
    segment_name: str = "Bladder",
    segment_id: Optional[str] = None,
    label_value: int = 1,
    segment_color: str = "0.0 1.0 0.0",
):
    """Write a native-grid Slicer-compatible .seg.nrrd with spatial and segment metadata.

    Geometry fields such as sizes, space directions and space origin are inherited from the
    reference image. Slicer segmentation key/value fields are written explicitly so consumers
    can recognize the file as a Binary labelmap segmentation rather than a plain NRRD volume.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    clean_arr = np.squeeze(mask_arr).astype(np.uint8)
    if clean_arr.ndim != 3:
        raise ValueError(f"Mask array must be 3D after squeeze, got shape {mask_arr.shape}")

    if sitk is not None and reference_image_path and os.path.exists(reference_image_path):
        try:
            ref_img = sitk.ReadImage(reference_image_path)
            if ref_img.GetDimension() != 3:
                raise ValueError(
                    f"Reference image must be 3D, got dimension={ref_img.GetDimension()}"
                )

            mask_img = sitk.GetImageFromArray(clean_arr)

            if mask_img.GetSize() != ref_img.GetSize():
                # Compute physical spacing for the preprocessed mask before nearest-neighbor
                # resampling back to the exact native reference grid.
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

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(sitk.Transform())
                mask_img = resampler.Execute(mask_img)

            mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
            mask_img.CopyInformation(ref_img)
            _set_slicer_metadata(
                mask_img,
                segment_name=segment_name,
                segment_id=segment_id,
                label_value=label_value,
                segment_color=segment_color,
            )

            writer = sitk.ImageFileWriter()
            writer.SetFileName(output_path)
            writer.SetUseCompression(True)
            writer.Execute(mask_img)
            return
        except Exception as e:
            print(
                f"[NRRD Utils] Warning: SimpleITK spatial/Slicer write failed "
                f"({type(e).__name__}: {e}), falling back to pynrrd"
            )

    # Fallback to pynrrd. This path can still carry Slicer tags, but if a readable reference
    # image is unavailable it cannot guarantee native physical geometry. The normal HASSL
    # auto-label path uses the SimpleITK branch above with the source image as reference.
    size_xyz = (clean_arr.shape[2], clean_arr.shape[1], clean_arr.shape[0])
    header = {
        'type': 'uint8',
        'encoding': 'gzip',
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
    }
    header.update(
        _slicer_segmentation_metadata(
            size_xyz,
            segment_name=segment_name,
            segment_id=segment_id,
            label_value=label_value,
            segment_color=segment_color,
        )
    )
    nrrd.write(output_path, clean_arr, header=header)
