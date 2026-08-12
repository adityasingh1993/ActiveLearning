import os
import nrrd
import numpy as np
from typing import Dict, Any, Tuple

def parse_nrrd_segments(header: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Parse 3D Slicer segmentation metadata from a NRRD header.
    Returns a dictionary mapping label_value (int) to segment metadata.
    """
    segments = {}
    
    for key, value in header.items():
        if key.startswith('Segment') and '_Name' in key:
            seg_prefix = key.split('_Name')[0]
            
            name = value
            try:
                label_value = int(header.get(f"{seg_prefix}_LabelValue", 1))
            except ValueError:
                label_value = 1
                
            color = header.get(f"{seg_prefix}_Color", "0.5 0.5 0.5")
            color_array = np.array([float(x) for x in color.split()])
            
            segments[label_value] = {
                'name': name,
                'label_value': label_value,
                'color': color_array
            }
            
    return segments

def load_seg_nrrd(filepath: str) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Load a .seg.nrrd file and return the label map as an integer numpy array
    along with the segment metadata. Handles detached headers (.nhdr) as well.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NRRD file not found: {filepath}")
        
    # nrrd library automatically handles detached headers (.nhdr) properly
    data, header = nrrd.read(filepath)
    
    segments = parse_nrrd_segments(header)
    
    return data.astype(np.int32), segments
