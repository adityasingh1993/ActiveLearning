import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_overlay(image_slice, pred_slice, gt_slice=None):
    """
    Overlays prediction and optionally ground truth on an image slice.
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Base image in grayscale
    ax.imshow(image_slice, cmap='gray')
    
    # Prediction in red (alpha 0.5)
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    ax.imshow(pred_masked, cmap='Reds', alpha=0.5)
    
    if gt_slice is not None:
        # Ground truth in green (alpha 0.5, outlines could be better but this is simple)
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax.imshow(gt_masked, cmap='Greens', alpha=0.3)
        ax.set_title('Image (Gray) | Pred (Red) | GT (Green)')
    else:
        ax.set_title('Image (Gray) | Pred (Red)')
        
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_uncertainty_map(uncertainty_volume, slice_idx, axis=0):
    """
    Plots a heatmap of the uncertainty for a specific slice along a given axis.
    """
    if axis == 0:
        slice_data = uncertainty_volume[slice_idx, :, :]
    elif axis == 1:
        slice_data = uncertainty_volume[:, slice_idx, :]
    else:
        slice_data = uncertainty_volume[:, :, slice_idx]
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(slice_data, cmap='jet')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Uncertainty Map (Axis {axis}, Slice {slice_idx})')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_al_scores(scores_dict, round_num):
    """
    Creates a bar chart of volume scores for a given active learning round.
    """
    vids = list(scores_dict.keys())
    scores = [scores_dict[v] for v in vids]
    
    # Sort for better visualization
    sorted_indices = np.argsort(scores)[::-1]
    vids = [vids[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(vids, scores, color='skyblue')
    ax.set_title(f'Active Learning Scores - Round {round_num}')
    ax.set_xlabel('Volume ID')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    return fig

def render_thermal_heatmap(map_2d: np.ndarray, max_val: float = None) -> np.ndarray:
    """Render a 2D float uncertainty/variance map as a thermal RGB heatmap [H, W, 3].

    Colormap: Blue (0.0 = low/certain) -> Green/Yellow (0.5 = moderate) -> Red (1.0 = high/uncertain).
    """
    if max_val is None or max_val <= 0:
        max_val = float(map_2d.max()) + 1e-8
    norm = np.clip(map_2d / max_val, 0.0, 1.0)

    r = np.clip(2.0 * norm - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * norm - 1.0), 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * norm, 0.0, 1.0)

    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    return rgb


def render_uncertainty_slice_grid(
    slice_img: np.ndarray,
    slice_gt: np.ndarray,
    slice_pred: np.ndarray,
    slice_mc_var: np.ndarray,
    slice_tta_var: np.ndarray,
    slice_lcc: np.ndarray = None,
) -> np.ndarray:
    """Generate a 6-panel composite preview grid.

    Panels:
        1. Original Grayscale Image
        2. Ground Truth (Green overlay)
        3. Raw Model Prediction (Cyan overlay)
        4. MC Dropout Epistemic Uncertainty Heatmap (Blue=Low, Red=High)
        5. TTA Aleatoric Uncertainty Heatmap (Blue=Low, Red=High)
        6. Error Map (Green=TP, Red=FP, Blue=FN)

    Returns:
        RGB numpy array of shape [H, W * 6, 3] ready for WandB/MLflow image logging.
    """
    slice_lcc = slice_lcc if slice_lcc is not None else slice_pred

    # Grayscale base image normalized to [0, 255]
    p_min, p_max = slice_img.min(), slice_img.max()
    slice_norm = (slice_img - p_min) / (p_max - p_min + 1e-8)
    base_gray = (slice_norm * 255.0).astype(np.uint8)
    base_rgb = np.stack([base_gray] * 3, axis=-1)

    # Panel 1: Original Image
    p1 = base_rgb.copy()

    # Panel 2: Ground Truth (Green overlay)
    p2 = base_rgb.copy()
    p2[slice_gt > 0, 1] = np.clip(p2[slice_gt > 0, 1].astype(np.int32) + 120, 0, 255).astype(np.uint8)

    # Panel 3: Raw Model Prediction (Cyan overlay)
    p3 = base_rgb.copy()
    p3[slice_pred > 0, 0] = np.clip(p3[slice_pred > 0, 0].astype(np.int32) + 120, 0, 255).astype(np.uint8)
    p3[slice_pred > 0, 2] = np.clip(p3[slice_pred > 0, 2].astype(np.int32) + 120, 0, 255).astype(np.uint8)

    # Panel 4: MC Epistemic Uncertainty Heatmap blended with base image
    mc_rgb = render_thermal_heatmap(slice_mc_var)
    p4 = (base_rgb * 0.4 + mc_rgb * 0.6).astype(np.uint8)

    # Panel 5: TTA Aleatoric Uncertainty Heatmap blended with base image
    tta_rgb = render_thermal_heatmap(slice_tta_var)
    p5 = (base_rgb * 0.4 + tta_rgb * 0.6).astype(np.uint8)

    # Panel 6: Error Map (Green=TP, Red=FP, Blue=FN)
    p6 = base_rgb.copy()
    tp = (slice_gt > 0) & (slice_lcc > 0)
    fp = (slice_gt == 0) & (slice_lcc > 0)
    fn = (slice_gt > 0) & (slice_lcc == 0)

    p6[tp, 1] = 255  # Green = TP
    p6[fp, 0] = 255  # Red = FP
    p6[fn, 2] = 255  # Blue = FN

    grid_img = np.concatenate([p1, p2, p3, p4, p5, p6], axis=1)
    return grid_img

