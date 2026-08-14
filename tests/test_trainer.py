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
        elif out.ndim == 6:
            out = out[:, 0]
    assert out.shape[1] == sample_config.num_classes

def test_apply_keep_largest_cc_component_1():
    from hassl.utils.postprocessing import apply_keep_largest_cc
    # Construct tensor with two disconnected Component 1 blobs: one large (size 125), one small satellite (size 8)
    t = torch.zeros(1, 1, 32, 32, 32)
    t[0, 0, 5:10, 5:10, 5:10] = 1.0  # Large component 1 (5x5x5 = 125 voxels)
    t[0, 0, 25:27, 25:27, 25:27] = 1.0  # Small satellite blob (2x2x2 = 8 voxels)

    filtered = apply_keep_largest_cc(t, min_size_voxels=10, target_class=1)

    # Large component must be retained (125 voxels)
    assert filtered[0, 0, 5:10, 5:10, 5:10].sum() == 125
    # Small satellite blob must be zeroed out (0 voxels)
    assert filtered[0, 0, 25:27, 25:27, 25:27].sum() == 0
    assert filtered.sum() == 125

