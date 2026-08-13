import pytest
import os
from hassl.data.data_engine import (
    build_labeled_dataset,
    build_unlabeled_dataset,
    build_dataloaders,
    get_base_transforms,
    build_all_volumes_loader
)
from hassl.config import HASSLConfig


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
    labeled, unlabeled, val = build_dataloaders(sample_config)
    assert labeled is not None or unlabeled is not None or val is not None


def test_base_transforms(sample_config):
    transforms = get_base_transforms(sample_config, keys=["image", "label"], is_training=False)
    assert transforms is not None


def test_build_all_volumes_loader(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    sample_config.use_cache_dataset = False
    loader = build_all_volumes_loader(sample_config)
    assert len(loader.dataset) >= 1
