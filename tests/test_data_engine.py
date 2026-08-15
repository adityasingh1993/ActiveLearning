import pytest
import os
from hassl.data.data_engine import (
    build_labeled_dataset,
    build_unlabeled_dataset,
    build_dataloaders,
    get_base_transforms,
    build_all_volumes_loader,
    _strip_suffix,
)
from hassl.config import HASSLConfig


def test_strip_suffix_simple():
    """_strip_suffix must only remove a trailing suffix, not mid-string occurrences."""
    assert _strip_suffix("volume_0.mha", ".mha") == "volume_0"


def test_strip_suffix_dotted_filename():
    """Dotted filenames like sonoeq.transducer.frameacquasition.mha must strip correctly."""
    assert _strip_suffix("sonoeq.transducer.frameacquasition.mha", ".mha") == "sonoeq.transducer.frameacquasition"


def test_strip_suffix_no_mid_replacement():
    """str.replace would break 'scan.mha.backup.mha' → only trailing .mha removed."""
    assert _strip_suffix("scan.mha.backup.mha", ".mha") == "scan.mha.backup"


def test_strip_suffix_no_match():
    """If suffix not present, filename returned unchanged."""
    assert _strip_suffix("volume_0.nrrd", ".mha") == "volume_0.nrrd"




def test_build_labeled_dataset(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    dataset, labeled_ids = build_labeled_dataset(
        data_dir=sample_config.data_dir,
        image_suffix=sample_config.image_suffix,
        label_suffix=sample_config.label_suffix,
        use_cache_dataset=False,
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert 'image' in item and 'label' in item


def test_build_unlabeled_dataset(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    dataset = build_unlabeled_dataset(
        data_dir=sample_config.data_dir,
        image_suffix=sample_config.image_suffix,
        labeled_ids={"volume_0", "volume_1"},
        use_cache_dataset=False,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert 'image' in item
    assert 'label' not in item


def test_build_dataloaders(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    sample_config.use_cache_dataset = False
    labeled, unlabeled, val, val_transforms = build_dataloaders(sample_config)
    assert labeled is not None or unlabeled is not None or val is not None
    assert val_transforms is not None  # must be the Compose instance for Invertd identity matching


def test_base_transforms(sample_config):
    transforms = get_base_transforms(sample_config, keys=["image", "label"], is_training=False)
    assert transforms is not None


def test_build_all_volumes_loader(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    sample_config.use_cache_dataset = False
    loader = build_all_volumes_loader(sample_config)
    assert len(loader.dataset) >= 1
