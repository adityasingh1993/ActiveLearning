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

def test_build_network_res_dynunet(sample_config):
    net = build_network('res_dynunet', sample_config.num_classes, sample_config.dropout)
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

def test_apply_keep_largest_cc_n_components_1_no_voxel_cutoff():
    from hassl.utils.postprocessing import apply_keep_largest_cc
    # Construct tensor with two small blobs: size 12 and size 4 (both < 100 voxels)
    t = torch.zeros(1, 1, 16, 16, 16)
    t[0, 0, 2:4, 2:4, 2:5] = 1.0  # Component A (2x2x3 = 12 voxels)
    t[0, 0, 10:12, 10:12, 10:11] = 1.0  # Component B (2x2x1 = 4 voxels)

    # n_components=1, min_size_voxels=0 (no hard voxel cutoff)
    filtered = apply_keep_largest_cc(t, n_components=1, min_size_voxels=0)

    # Top 1 largest component (12 voxels) must be kept even though < 100
    assert filtered[0, 0, 2:4, 2:4, 2:5].sum() == 12
    # Component B (4 voxels) must be zeroed out
    assert filtered[0, 0, 10:12, 10:12, 10:11].sum() == 0
    assert filtered.sum() == 12

def test_apply_keep_largest_cc_fill_holes():
    from hassl.utils.postprocessing import apply_keep_largest_cc
    # Construct a 3D box (5x5x5 = 125 voxels) with an internal 1x1x1 hole at center (3,3,3)
    t = torch.zeros(1, 1, 10, 10, 10)
    t[0, 0, 1:6, 1:6, 1:6] = 1.0
    t[0, 0, 3, 3, 3] = 0.0  # Internal false-negative hole

    # With fill_holes=True (default)
    filled = apply_keep_largest_cc(t, fill_holes=True)
    # The internal 1x1x1 hole at (3,3,3) must be filled (now 1.0)
    assert filled[0, 0, 3, 3, 3] == 1.0
    assert filled.sum() == 125

    # With fill_holes=False
    unfilled = apply_keep_largest_cc(t, fill_holes=False)
    # The internal hole remains 0.0
    assert unfilled[0, 0, 3, 3, 3] == 0.0
    assert unfilled.sum() == 124



