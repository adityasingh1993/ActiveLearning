import pytest
import torch
from hassl.trainer import HASSLTrainer
from hassl.config import HASSLConfig

def test_build_network_unet(sample_config):
    sample_config.unet_backbone = 'UNet'
    sample_config.unet_channels = (8, 16, 32, 64)
    sample_config.unet_strides = (2, 2, 2)
    sample_config.num_classes = 1
    trainer = HASSLTrainer(sample_config)
    net = trainer.build_network()
    assert net is not None

def test_build_network_dynunet(sample_config):
    sample_config.unet_backbone = 'DynUNet'
    sample_config.dynunet_filters = [8, 16, 32, 64]
    sample_config.dynunet_kernel_size = [[3,3,3]]*4
    sample_config.dynunet_strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2]]
    sample_config.dynunet_upsample_kernel_size = [[2,2,2]]*3
    trainer = HASSLTrainer(sample_config)
    net = trainer.build_network()
    assert net is not None

def test_build_network_swinunetr(sample_config):
    sample_config.unet_backbone = 'SwinUNETR'
    sample_config.swin_feature_size = 12
    trainer = HASSLTrainer(sample_config)
    net = trainer.build_network()
    assert net is not None

def test_unet_forward_pass(sample_config, sample_volume):
    sample_config.unet_backbone = 'UNet'
    sample_config.unet_channels = (8, 16, 32, 64)
    sample_config.unet_strides = (2, 2, 2)
    sample_config.num_classes = 1
    trainer = HASSLTrainer(sample_config)
    net = trainer.build_network()
    
    with torch.no_grad():
        out = net(sample_volume)
    assert out.shape == (1, 1, 32, 32, 32)

def test_dynunet_forward_pass(sample_config, sample_volume):
    sample_config.unet_backbone = 'DynUNet'
    sample_config.dynunet_filters = [8, 16, 32, 64]
    sample_config.dynunet_kernel_size = [[3,3,3]]*4
    sample_config.dynunet_strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2]]
    sample_config.dynunet_upsample_kernel_size = [[2,2,2]]*3
    sample_config.num_classes = 1
    sample_config.dynunet_deep_supervision = False
    trainer = HASSLTrainer(sample_config)
    net = trainer.build_network()
    
    with torch.no_grad():
        out = net(sample_volume)
    assert out.shape == (1, 1, 32, 32, 32)
