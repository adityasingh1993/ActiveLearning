import pytest
from hassl.config import HASSLConfig


def test_default_config():
    config = HASSLConfig()
    assert config.spatial_size == (128, 128, 128)
    assert config.batch_size == 1
    assert config.num_classes == 1


def test_yaml_roundtrip(tmp_path):
    config = HASSLConfig(compute_mode="full", batch_size=2, train_lr=0.001)
    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(str(yaml_path))

    loaded_config = HASSLConfig.from_yaml(str(yaml_path))
    assert loaded_config.batch_size == 2
    assert loaded_config.train_lr == 0.001


def test_prototype_mode_overrides():
    config = HASSLConfig(compute_mode="prototype", batch_size=8, mc_dropout_passes=20)
    assert config.batch_size == 1
    assert config.mc_dropout_passes == 5


def test_full_mode_overrides():
    config = HASSLConfig(compute_mode="full", batch_size=1, mc_dropout_passes=5)
    assert config.batch_size >= 2
    assert config.mc_dropout_passes >= 10


def test_is_binary():
    config = HASSLConfig(num_classes=1)
    assert config.is_binary is True
    config = HASSLConfig(num_classes=3)
    assert config.is_binary is False


def test_output_channels():
    config = HASSLConfig(num_classes=1)
    assert config.output_channels == 1
    config = HASSLConfig(num_classes=3)
    assert config.output_channels == 3
