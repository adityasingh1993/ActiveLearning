# Implementation Plan: Extended Validation Metrics & Image Logging

Add comprehensive validation metrics (Precision, Recall, Relative Volume Error, Volume R², 95th Percentile Hausdorff Distance) and automated periodic 2D/3D slice image logging to `HASSLTrainer` and `ExperimentTracker`.

## User Review Required

> [!NOTE]
> - Validation metrics will now return a dictionary containing `val_dice`, `val_precision`, `val_recall`, `val_rve_pct`, `val_volume_r2`, and `val_hd95`.
> - Validation slice preview images will be generated every `log_image_every_n_epochs` (default: 10) and on the final epoch.
> - Preview images will be logged to WandB / MLflow via `tracker.log_image()` and saved locally in `experiments/logs/val_previews/` for offline inspection.

## Proposed Changes

### Component 1: Training & Evaluation (`hassl/training/`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
- **Update `validate()`**:
  - Integrate MONAI `ConfusionMatrixMetric` or exact PyTorch tensor calculation to compute:
    - **`val_precision`**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}$
    - **`val_recall`**: $\frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}$
  - Compute volumetric metrics across all validation volumes:
    - **`val_rve_pct`**: Mean Relative Volume Error $\frac{|V_{\text{pred}} - V_{\text{gt}}|}{V_{\text{gt}} + \epsilon} \times 100\%$
    - **`val_volume_r2`**: Coefficient of determination $R^2$ of predicted volume vs ground truth volume across validation subjects.
  - Integrate MONAI `HausdorffDistanceMetric(percentile=95)` for **`val_hd95`** (with safe exception handling for empty masks).
  - Return a dictionary of validation metrics instead of just a scalar `val_dice`.
- **Add `log_validation_samples(epoch, inputs, targets, preds)`**:
  - Extract the slice along the axial plane with the largest ground-truth foreground target area (or middle slice).
  - Construct a 4-panel RGB preview panel:
    1. **Input Image** (Grayscale normalized CT/MRI)
    2. **Ground Truth Mask** (Green mask overlay)
    3. **Model Prediction Mask** (Red/Cyan mask overlay)
    4. **Composite Overlay** (Input + GT Contour + Pred Contour + Error Map)
  - Log image to tracker using `self.tracker.log_image(grid_np, name=f"val_slice_preview", step=epoch)`.
  - Save image locally under `experiments/logs/val_previews/epoch_{epoch:03d}.png`.
- **Update `train()`**:
  - Log all extended validation metrics to `self.tracker.log_metrics(metrics, step=epoch)`.
  - Trigger `log_validation_samples` when `(epoch + 1) % self.config.log_image_every_n_epochs == 0` or on the final epoch.
  - Print formatted summary line in console:
    ```text
    Epoch  10/100 | Loss: 0.2314 | Dice: 0.8451 | Prec: 0.8620 | Rec: 0.8290 | RVE: 8.4% | R²: 0.912 | HD95: 3.42mm
    ```

---

### Component 2: Tracking Subsystem (`hassl/`)

#### [MODIFY] [tracking.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/tracking.py)
- Enhance `log_image()`:
  - Add fallback logic to write image files directly to `experiments/logs/val_previews/` if backend is `"none"`, ensuring local visibility even without WandB/MLflow.

---

## Verification Plan

### Automated Tests
1. Run pytest suite to verify trainer & tracking modifications:
   ```powershell
   pytest tests/
   ```
2. Run a 2-epoch dry run of `hassl.pipeline` in prototype mode to verify metrics logging and preview image generation:
   ```powershell
   python -m hassl.pipeline --config config.yaml --phase train --round 0
   ```

### Manual Verification
- Verify that metrics (`val_dice`, `val_precision`, `val_recall`, `val_rve_pct`, `val_volume_r2`, `val_hd95`) are correctly logged to tracker / output logs.
- Verify that validation slice preview images are saved under `./experiments/logs/val_previews/epoch_*.png` and rendered correctly in RGB grid format.
