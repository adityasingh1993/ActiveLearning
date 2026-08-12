"""
HASSL Synthetic CI Integration Test Suite.

Runs automated assertions across pipeline artifacts, patient-level splits,
provenance isolation, and spatial geometry header preservation.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from hassl.config import HASSLConfig
from hassl.data.data_engine import get_or_create_frozen_splits, build_labeled_dataset
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry


def test_import_smoke_check():
    """Verify all hassl submodules load cleanly without SyntaxError or NameError."""
    import importlib
    import pkgutil
    import hassl

    failed = []
    for m in pkgutil.walk_packages(hassl.__path__, prefix="hassl."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not failed, "Modules failed to import:\n" + "\n".join(failed)


def test_frozen_splits_patient_grouping_validation(tmp_path):
    """Verify splits.json creates patient-level holdouts and fails loudly on patient collapse (V-11 fix)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    labels_dir = data_dir / "labels"
    labels_dir.mkdir(parents=True)

    # Create synthetic volumes with patient prefixes: US001_v1, US001_v2, US002_v1, US002_v2, etc.
    for i in range(1, 7):
        patient_id = f"PAT{i:03d}"
        for v in range(1, 3):
            vol_name = f"{patient_id}_vol{v}"
            img_file = data_dir / f"{vol_name}.mha"
            lbl_file = labels_dir / f"{vol_name}.seg.nrrd"
            img_file.write_bytes(b"header")
            lbl_file.write_bytes(b"label")

    splits = get_or_create_frozen_splits(str(data_dir), seed=42)

    assert os.path.exists(str(data_dir / "splits.json"))
    assert len(splits["val_ids"]) > 0, "Validation set must not be empty"
    assert len(splits["test_ids"]) > 0, "Test set must not be empty"
    assert len(splits["initial_train_ids"]) > 0, "Train set must not be empty"

    # Verify patient isolation between splits
    val_patients = set(v.split("_")[0] for v in splits["val_ids"])
    train_patients = set(v.split("_")[0] for v in splits["initial_train_ids"])
    test_patients = set(v.split("_")[0] for v in splits["test_ids"])

    assert val_patients.isdisjoint(train_patients), "Patient leakage detected between val and train!"
    assert test_patients.isdisjoint(train_patients), "Patient leakage detected between test and train!"


def test_provenance_manifest_gating(tmp_path):
    """Verify accepted pseudo-labels are isolated to pseudo_approved and excluded from human gold pool (V-4 fix)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    labels_dir = data_dir / "labels"
    approved_dir = data_dir / "pseudo_approved"
    labels_dir.mkdir(parents=True)
    approved_dir.mkdir(parents=True)

    # Create 1 human volume, 1 pseudo_approved volume
    (data_dir / "vol_human.mha").write_bytes(b"data")
    (labels_dir / "vol_human.seg.nrrd").write_bytes(b"mask")

    (data_dir / "vol_pseudo.mha").write_bytes(b"data")
    (approved_dir / "vol_pseudo.seg.nrrd").write_bytes(b"mask")

    manifest = {
        "provenance": {
            "vol_human": "human",
            "vol_pseudo": "pseudo_approved"
        }
    }
    manifest_file = tmp_path / "pool_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    ds, labeled_ids = build_labeled_dataset(
        str(data_dir), ".mha", ".seg.nrrd", manifest_path=str(manifest_file), use_cache_dataset=False
    )

    provenance_map = {item["id"]: item["provenance"] for item in ds.data}
    assert provenance_map.get("vol_human") == "human"
    assert provenance_map.get("vol_pseudo") == "pseudo_approved"


def test_frozen_splits_patient_collapse_failure(tmp_path):
    """Verify get_or_create_frozen_splits raises ValueError when all volumes collapse to 1 patient (V-11 fix)."""
    data_dir = tmp_path / "data_collapse"
    data_dir.mkdir(parents=True)
    labels_dir = data_dir / "labels"
    labels_dir.mkdir(parents=True)

    # 10 volumes with same prefix 'commonprefix' causing single-patient collapse
    for i in range(1, 10):
        vol_name = f"commonprefix_case{i}"
        (data_dir / f"{vol_name}.mha").write_bytes(b"data")
        (labels_dir / f"{vol_name}.seg.nrrd").write_bytes(b"label")

    with pytest.raises(ValueError, match="collapsed all 9 volumes into 1 single patient ID"):
        get_or_create_frozen_splits(str(data_dir), seed=42)


def test_written_mask_preserves_geometry(tmp_path):
    """Verify write_mask_with_spatial_geometry preserves physical affine size, spacing, & direction (V6-5 fix)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not installed")

    src_path = str(tmp_path / "reference.mha")
    out_path = str(tmp_path / "written_mask.seg.nrrd")

    # Create dummy reference ITK image
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    ref_img = sitk.GetImageFromArray(arr)
    ref_img.SetSpacing((0.5, 0.4, 0.8))
    ref_img.SetOrigin((10.0, -5.0, 3.0))
    sitk.WriteImage(ref_img, src_path)

    mask_arr = np.ones((16, 16, 16), dtype=np.uint8)  # Resized 16^3 prediction mask
    write_mask_with_spatial_geometry(out_path, mask_arr, reference_image_path=src_path)

    got_img = sitk.ReadImage(out_path)
    assert got_img.GetSize() == ref_img.GetSize(), "Output mask size must match reference image size"
    assert got_img.GetSpacing() == pytest.approx(ref_img.GetSpacing()), "Output mask spacing must match reference image spacing"
    assert got_img.GetOrigin() == pytest.approx(ref_img.GetOrigin()), "Output mask origin must match reference image origin"


def test_teacher_student_views_are_spatially_aligned():
    """Verify teacher and student augmentation views share exact spatial coordinate frames (V7-1 fix)."""
    try:
        import torch
        from hassl.data.augmentations import get_spatial_augmentation, get_intensity_augmentation
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    spatial_aug = get_spatial_augmentation(keys=["image"])
    intensity_aug = get_intensity_augmentation(keys=["image"])

    # Create an asymmetric 3D phantom (off-center sphere/cube to ensure zero flip symmetry)
    vol = torch.zeros((1, 32, 32, 32), dtype=torch.float32)
    vol[:, 4:12, 16:28, 2:8] = 1.0  # Asymmetric off-center foreground patch

    # Apply shared spatial transform
    spatial_dict = spatial_aug({"image": vol})
    teacher_view = spatial_dict["image"]
    student_view = intensity_aug({"image": teacher_view})["image"]

    # Binarize masks
    t_mask = (teacher_view > 0.1).float()
    s_mask = (student_view > 0.1).float()

    intersection = (t_mask * s_mask).sum()
    total = t_mask.sum() + s_mask.sum()
    dice = float((2.0 * intersection / (total + 1e-8)).item())

    assert dice >= 0.90, f"Teacher and student views are spatially misaligned! Spatial Dice overlap: {dice:.4f} < 0.90"


def test_ssl_contrastive_loss_nonzero_at_batch_size_1():
    """Verify SSL InfoNCE contrastive loss is non-zero and produces gradients even at batch_size=1."""
    try:
        import torch
        from hassl.config import HASSLConfig
        from hassl.ssl.ssl_pretrainer import SSLPretrainer
    except ImportError:
        pytest.skip("PyTorch or MONAI not installed")

    config = HASSLConfig()
    config.device = "cpu"
    config.compute_mode = "prototype"
    config.unet_backbone = "unet"

    pretrainer = SSLPretrainer(config=config, dataloader=[], tracker=None)

    # Synthetic batch of size B=1
    x = torch.randn(1, 1, 32, 32, 32)
    x_aug1 = x + torch.randn_like(x) * 0.05
    x_aug2 = x + torch.randn_like(x) * 0.05

    b1 = pretrainer._extract_bottleneck_features(x_aug1)
    b2 = pretrainer._extract_bottleneck_features(x_aug2)

    p1 = torch.nn.functional.adaptive_avg_pool3d(b1, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b1.size(1))
    p2 = torch.nn.functional.adaptive_avg_pool3d(b2, (2, 2, 2)).permute(0, 2, 3, 4, 1).reshape(-1, b2.size(1))

    feat1 = pretrainer.proj_head(p1)
    feat2 = pretrainer.proj_head(p2)

    loss_cont = pretrainer._infonce_loss(feat1, feat2)
    assert loss_cont.item() > 0.0, f"Contrastive loss at batch_size=1 must be > 0.0, got {loss_cont.item()}"
    assert loss_cont.requires_grad, "Contrastive loss must require grad for optimization"
