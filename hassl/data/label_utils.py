"""Label normalisation utilities for HASSL datasets.

Provides NormalizeLabelsInDatasetd — a MONAI-compatible MapTransform that remaps
arbitrary or non-contiguous label values to contiguous 0, 1, 2, ..., N indices.

Why this is needed
------------------
Multi-class NRRD segmentation files sometimes use non-contiguous label values
(e.g. 0, 1, 3, 7) because the values were assigned per-structure in a tool like
3D Slicer. AsDiscreted(threshold=0.5) only works for binary masks — it would
incorrectly binarise label=3 and label=7 into 1 (foreground).

NormalizeLabelsInDatasetd remaps an explicit label_names dict (from config) into
contiguous indices: label_names={0: 'bg', 1: 'bladder', 3: 'urethra', 7: 'sphincter'}
becomes {0->0, 1->1, 3->2, 7->3}.

When to use
-----------
Applied in get_base_transforms when config.num_classes > 1 AND
config.label_names is provided. For binary datasets (num_classes == 1),
AsDiscreted(threshold=0.5) remains the correct choice.
"""

import torch
import numpy as np
from monai.transforms import MapTransform


class NormalizeLabelsInDatasetd(MapTransform):
    """Remap non-contiguous label values to contiguous 0, 1, 2, ..., N indices.

    Args:
        keys: Keys in the data dict to apply the transform to (typically ['label']).
        label_names: Dict mapping original label value -> class name, or
                     list of (original_value, class_name) pairs.
                     Background (0) is always kept as 0.
                     Example: {0: 'background', 1: 'bladder', 3: 'urethra', 7: 'sphincter'}
        allow_missing_keys: If True, silently skip missing keys.

    Example::

        t = NormalizeLabelsInDatasetd(
            keys=['label'],
            label_names={0: 'background', 1: 'bladder', 3: 'urethra'}
        )
        # A label tensor with values [0, 1, 3] becomes [0, 1, 2]
    """

    def __init__(self, keys, label_names: dict, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        if not isinstance(label_names, dict):
            raise ValueError("label_names must be a dict mapping original_value -> class_name")

        # Build a sorted mapping: sort by original value so background (0) stays 0
        sorted_values = sorted(label_names.keys())
        # contiguous index = position in sorted list (0 -> 0, first nonzero -> 1, etc.)
        self._remap = {orig: idx for idx, orig in enumerate(sorted_values)}

        # Validate that 0 maps to 0 (background must stay background)
        if 0 in self._remap and self._remap[0] != 0:
            raise ValueError("label value 0 (background) must map to contiguous index 0")

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            is_tensor = torch.is_tensor(label)
            arr = label.numpy() if is_tensor else np.asarray(label)

            out = np.zeros_like(arr, dtype=arr.dtype)
            for orig_val, new_idx in self._remap.items():
                out[arr == orig_val] = new_idx

            d[key] = torch.from_numpy(out) if is_tensor else out
        return d

    @property
    def num_classes(self) -> int:
        """Number of distinct classes after remapping (including background)."""
        return len(self._remap)

    @property
    def remap(self) -> dict:
        """Read-only view of the original -> contiguous index mapping."""
        return dict(self._remap)
