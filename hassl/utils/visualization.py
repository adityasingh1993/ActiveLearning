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

def save_figure_as_array(fig):
    """
    Converts a matplotlib figure to an RGB numpy array.
    """
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data
