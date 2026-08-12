import pytest
import torch
import numpy as np
import SimpleITK as sitk
import nrrd
from pathlib import Path
from hassl.config import HASSLConfig

@pytest.fixture
def sample_config(tmp_path):
    config = HASSLConfig()
    config.ssl_epochs = 2
    config.train_epochs = 2
    config.spatial_size = (32, 32, 32)
    config.batch_size = 1
    config.device = 'cpu'
    config.tracker = 'none'
    config.unet_channels = (8, 16, 32, 64)
    config.cache_dir = str(tmp_path / "cache")
    return config

@pytest.fixture
def synthetic_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for i in range(3):
        vol = np.random.rand(32, 32, 32).astype(np.float32)
        img = sitk.GetImageFromArray(vol)
        sitk.WriteImage(img, str(image_dir / f"volume_{i}.mha"))
        
        if i < 2:
            seg = np.random.randint(0, 2, (32, 32, 32)).astype(np.uint8)
            nrrd.write(str(label_dir / f"volume_{i}.seg.nrrd"), seg)
            
    return data_dir

@pytest.fixture
def sample_volume():
    return torch.rand(1, 1, 32, 32, 32)

@pytest.fixture
def sample_label():
    return torch.randint(0, 2, (1, 1, 32, 32, 32)).float()
