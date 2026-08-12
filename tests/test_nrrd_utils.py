import pytest
import numpy as np
import nrrd
from hassl.data.nrrd_utils import parse_nrrd_segments, load_seg_nrrd

def test_parse_nrrd_segments(tmp_path):
    # Create dummy header with segment info
    header = {
        'Segment0_Name': 'Tumor',
        'Segment0_LabelValue': '1',
        'Segment1_Name': 'Organ',
        'Segment1_LabelValue': '2',
    }
    segments = parse_nrrd_segments(header)
    assert len(segments) == 2
    assert segments[1] == 'Tumor'
    assert segments[2] == 'Organ'

def test_load_seg_nrrd(tmp_path):
    seg_data = np.zeros((32, 32, 32), dtype=np.uint8)
    seg_data[10:20, 10:20, 10:20] = 1
    
    header = {
        'type': 'unsigned char',
        'dimension': 3,
        'sizes': [32, 32, 32],
        'Segment0_Name': 'Tumor',
        'Segment0_LabelValue': '1',
    }
    file_path = tmp_path / "test.seg.nrrd"
    nrrd.write(str(file_path), seg_data, header)
    
    loaded_data, segments = load_seg_nrrd(str(file_path))
    assert loaded_data.shape == (32, 32, 32)
    assert loaded_data.dtype == np.uint8
    assert 1 in segments
    assert segments[1] == 'Tumor'
