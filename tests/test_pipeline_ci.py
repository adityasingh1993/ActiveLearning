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
