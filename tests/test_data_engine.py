import pytest
from hassl.data.data_engine import DataEngine
from hassl.config import HASSLConfig

def test_build_labeled_dataset(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    engine = DataEngine(sample_config)
    dataset = engine.build_labeled_dataset()
    assert len(dataset) == 2
    item = dataset[0]
    assert 'image' in item and 'label' in item

def test_build_unlabeled_dataset(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    engine = DataEngine(sample_config)
    dataset = engine.build_unlabeled_dataset()
    assert len(dataset) == 1
    item = dataset[0]
    assert 'image' in item
    assert 'label' not in item

def test_build_dataloaders(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    engine = DataEngine(sample_config)
    labeled, unlabeled, val = engine.build_dataloaders()
    assert labeled is not None
    assert unlabeled is not None
    assert val is not None

def test_base_transforms(sample_config):
    engine = DataEngine(sample_config)
    transforms = engine.get_base_transforms()
    # Dummy test to verify transform chain builds
    assert transforms is not None

def test_build_all_volumes_loader(synthetic_data_dir, sample_config):
    sample_config.data_dir = str(synthetic_data_dir)
    engine = DataEngine(sample_config)
    loader = engine.build_all_volumes_loader()
    # Should load all 3 volumes
    assert len(loader.dataset) == 3
