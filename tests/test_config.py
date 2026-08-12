import pytest
import tempfile
import os
from hassl.config import HASSLConfig

def test_default_config():
    config = HASSLConfig()
    assert config.spatial_size == (128, 128, 128)
    assert config.batch_size == 2
    assert config.num_classes == 1

def test_yaml_roundtrip(tmp_path):
    config = HASSLConfig()
    config.batch_size = 4
    config.learning_rate = 0.001
    yaml_path = tmp_path / "config.yaml"
    config.save(yaml_path)
    
    loaded_config = HASSLConfig.load(yaml_path)
    assert loaded_config.batch_size == 4
    assert loaded_config.learning_rate == 0.001

def test_prototype_mode_overrides():
    config = HASSLConfig()
    config.prototype_mode = True
    config.batch_size = 8
    config.mc_dropout_passes = 20
    config._apply_overrides()
    assert config.batch_size == 1
    assert config.mc_dropout_passes == 5

def test_full_mode_overrides():
    config = HASSLConfig()
    config.prototype_mode = False
    config.batch_size = 1
    config.mc_dropout_passes = 5
    config._apply_overrides()
    assert config.batch_size >= 2
    assert config.mc_dropout_passes >= 10

def test_is_binary():
    config = HASSLConfig()
    config.num_classes = 1
    assert config.is_binary is True
    config.num_classes = 3
    assert config.is_binary is False

def test_output_channels():
    config = HASSLConfig()
    config.num_classes = 1
    assert config.output_channels == 1
    config.num_classes = 3
    assert config.output_channels == 3
