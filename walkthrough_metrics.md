# Walkthrough: Extended Validation Metrics & Image Logging

We have successfully integrated extended validation metrics and periodic 2D validation slice preview image logging into the HASSL pipeline.

## Summary of Changes

### 1. Tracking Subsystem ([tracking.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/tracking.py#L156-L195))
- Enhanced `log_image()` to automatically write PNG previews to `./experiments/logs/val_previews/` locally in addition to logging to WandB (`wandb.Image`) or MLflow (`mlflow.log_figure`).
- Added robust array normalization to handle floating point or integer arrays cleanly.

### 2. Validation & Training Engine ([trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py#L290-L360))
- **Extended Validation Metrics Dict**: `validate()` now calculates and returns:
  - `val_dice`: Dice Similarity Coefficient.
  - `val_precision`: Positive Predictive Value ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
  - `val_recall`: Sensitivity ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
  - `val_rve_pct`: Mean Relative Volume Error percentage ($\frac{|V_{\text{pred}} - V_{\text{gt}}|}{V_{\text{gt}}} \times 100\%$).
  - `val_volume_r2`: Coefficient of determination $R^2$ of volume prediction vs ground truth volume across validation subjects.
  - `val_hd95`: 95th percentile Hausdorff Distance ($\text{mm}$).
- **Automated Slice Preview Generation (`log_validation_samples`)**:
  - Automatically identifies the 2D axial slice with the maximum ground-truth target area per validation run.
  - Constructs a 4-panel RGB preview panel:
    1. **Input CT/MRI Image**
    2. **Ground Truth Mask Overlay** (Green)
    3. **Model Prediction Mask Overlay** (Cyan)
    4. **Composite Error Overlay** (Green = True Positive, Red = False Positive, Blue = False Negative)
  - Periodically logs grid images every `log_image_every_n_epochs` (default: 10) and on the final epoch.

---

## Verification Results

1. **AST Syntax & Type Checking**: Verified clean Python AST compilation across `trainer.py` and `tracking.py`.
2. **Local Preview Generation**: Successfully verified local PNG creation under `experiments/logs/val_previews/test_preview_step000.png`.
