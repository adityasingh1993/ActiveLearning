import pytest
import torch
from hassl.training.trainer import HASSLTrainer, build_network
from hassl.config import HASSLConfig

def test_build_network_unet(sample_config):
    sample_config.unet_backbone = 'unet'
    net = build_network(sample_config.unet_backbone, sample_config.num_classes, sample_config.dropout)
    assert net is not None

def test_build_network_dynunet(sample_config):
    sample_config.unet_backbone = 'dynunet'
    net = build_network(sample_config.unet_backbone, sample_config.num_classes, sample_config.dropout)
    assert net is not None

def test_build_network_swinunetr(sample_config):
    net = build_network('swinunetr', sample_config.num_classes, sample_config.dropout)
    assert net is not None

def test_unet_forward_pass(sample_config, sample_volume):
    sample_config.unet_backbone = 'unet'
    net = build_network(sample_config.unet_backbone, sample_config.num_classes, sample_config.dropout)
    
    with torch.no_grad():
        out = net(sample_volume)
    assert out.shape[1] == sample_config.num_classes

def test_dynunet_forward_pass(sample_config, sample_volume):
    sample_config.unet_backbone = 'dynunet'
    net = build_network(sample_config.unet_backbone, sample_config.num_classes, sample_config.dropout)
    
    with torch.no_grad():
        out = net(sample_volume)
        if isinstance(out, (list, tuple)):
            out = out[0]
    assert out.shape[1] == sample_config.num_classes
