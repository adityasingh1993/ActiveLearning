"""
MONAI cross-version compatibility shims.

This module handles API differences between MONAI 1.4 and MONAI 1.5+.
All version-conditional logic is centralised here so the rest of the
codebase stays clean.

Breaking changes addressed:
  - Invertd: meta_keys / orig_meta_keys removed in MONAI 1.5
  - HausdorffDistanceMetric: always_return_as_numpy deprecated in 1.5,
      default changed from True→False (return type now follows input type)
  - Orientationd: labels default changed from explicit RAS tuple → None in 1.5.1
  - ConfusionMatrixMetric: reduction default changed mean_batch→mean in 1.5
"""

import monai
from packaging.version import Version as _V

# ---------------------------------------------------------------------------
# Version flags
# ---------------------------------------------------------------------------
_MONAI_VERSION = _V(monai.__version__)

#: True on MONAI >= 1.5.0 (Invertd meta_keys removed, HD95 default changed)
MONAI_GE_15: bool = _MONAI_VERSION >= _V("1.5.0")

#: True on MONAI >= 1.5.1 (Orientationd labels default → None)
MONAI_GE_151: bool = _MONAI_VERSION >= _V("1.5.1")


# ---------------------------------------------------------------------------
# Orientationd labels
# ---------------------------------------------------------------------------
#: Always pass this to Orientationd(labels=...) to get consistent RAS
#: behaviour across MONAI 1.4 (where the default was this tuple) and
#: MONAI 1.5.1+ (where the default changed to None / MetaTensor-space-aware).
ORIENTATIOND_RAS_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))


# ---------------------------------------------------------------------------
# Invertd factory
# ---------------------------------------------------------------------------
def build_invertd(keys, transform, orig_keys, **kwargs):
    """
    Version-safe ``Invertd`` constructor.

    In MONAI 1.4, ``Invertd`` required explicit ``meta_keys``,
    ``orig_meta_keys``, and ``meta_key_postfix`` to locate per-key metadata
    in the data dictionary.  These were removed in MONAI 1.5 in favour of
    automatic ``MetaTensor.applied_operations`` tracking.

    Parameters
    ----------
    keys : str or list[str]
        Prediction keys to invert.
    transform : monai.transforms.Compose
        **The exact same Compose instance** passed to the DataLoader, so that
        ``applied_operations`` memory IDs match.
    orig_keys : str or list[str]
        Reference image keys (used for shape/affine).
    **kwargs
        Extra keyword arguments forwarded to ``Invertd``
        (e.g. ``nearest_interp``, ``to_tensor``).

    Returns
    -------
    monai.transforms.Invertd
    """
    from monai.transforms import Invertd

    if MONAI_GE_15:
        # MONAI 1.5+: MetaTensor carries transform history automatically.
        # meta_keys / orig_meta_keys do not exist in this version.
        return Invertd(
            keys=keys,
            transform=transform,
            orig_keys=orig_keys,
            **kwargs,
        )
    else:
        # MONAI 1.4: explicit dict-based metadata routing is required.
        _keys = [keys] if isinstance(keys, str) else list(keys)
        _orig_keys = [orig_keys] if isinstance(orig_keys, str) else list(orig_keys)
        meta_keys = [f"{k}_meta_dict" for k in _keys]
        orig_meta_keys = [f"{ok}_meta_dict" for ok in _orig_keys]
        return Invertd(
            keys=keys,
            transform=transform,
            orig_keys=orig_keys,
            meta_keys=meta_keys,
            orig_meta_keys=orig_meta_keys,
            meta_key_postfix="meta_dict",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def to_numpy(tensor_or_array):
    """
    Safely convert a metric output to a NumPy scalar / array.

    MONAI 1.4 metrics return NumPy arrays by default
    (``always_return_as_numpy=True``).  MONAI 1.5+ returns tensors by
    default.  Use this helper anywhere metric results are read.
    """
    import numpy as np
    import torch

    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def build_hd95_metric(include_background: bool = True):
    """
    Version-safe ``HausdorffDistanceMetric`` at the 95th percentile.

    MONAI 1.4 default: ``always_return_as_numpy=True`` → returns NumPy.
    MONAI 1.5 default: ``always_return_as_numpy=False`` → returns Tensor.
    The parameter is deprecated in 1.5 and will be removed in 1.7.

    We always call ``to_numpy()`` on the result, so we don't pass the
    deprecated parameter at all.
    """
    from monai.metrics import HausdorffDistanceMetric

    kwargs = dict(include_background=include_background, percentile=95, reduction="mean")
    if not MONAI_GE_15:
        # Safe to set on 1.4; avoids depending on a default that will change.
        kwargs["always_return_as_numpy"] = True

    return HausdorffDistanceMetric(**kwargs)


def build_confusion_metric(include_background: bool = True, metric_name=None):
    """
    Version-safe ``ConfusionMatrixMetric`` with an explicit reduction.

    MONAI 1.4 default reduction: ``"mean_batch"``
    MONAI 1.5 default reduction: ``"mean"``  (corrected bugfix)

    We always pass ``reduction="mean"`` to get the corrected, consistent
    behaviour on both versions.
    """
    from monai.metrics import ConfusionMatrixMetric

    if metric_name is None:
        metric_name = ["precision", "recall"]

    return ConfusionMatrixMetric(
        include_background=include_background,
        metric_name=metric_name,
        reduction="mean",  # explicit — never rely on default across versions
    )
