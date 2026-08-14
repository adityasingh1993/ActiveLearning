# Implementation Plan - HASSL Round 5 Remediation

Fix all blocking bugs, data integrity issues, metric calculations, and architectural flaws identified in the Round 5 Code Re-Review.

## User Review Required

> [!IMPORTANT]
> - **Gold Label Integrity (V-4 / A-1)**: `server.py` accept endpoint will no longer write model presegmentations into `data/labels/`. Accepted masks will strictly go to `data/pseudo_approved/` with `provenance = "pseudo_approved"`, preventing model outputs from polluting the human gold label pool.
> - **Clinical Metrics Physical Scale (V-5 & V-6)**: Volume R² will now be calculated using physical volumes ($\text{mm}^3$) computed from per-case voxel spacing, rather than resampled 128³ voxel counts. HD95 will now use physical spacing ($\text{mm}$).
> - **Metric Failure Handling (V-7)**: Undefined or failed metrics (e.g. empty foreground HD95) will produce `NaN` and be reported as `N/A` instead of defaulting to `0.0` (which reads as a perfect surface match).

## Proposed Changes

### Component: Model Training & Evaluation (`hassl/training/trainer.py`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
- **Imports (V-1, V-2)**:
  - Add `from typing import Dict, List, Optional, Tuple, Any`.
  - Add `import numpy as np` at top level (and remove redundant inner import).
- **Determinism (V-10)**:
  - Call `monai.utils.set_determinism(seed=self.config.seed)` during `__init__`.
- **Per-Sample Provenance Weighting (V-8)**:
  - Modify `train_one_epoch_uamt` and `train_one_epoch_cps` to compute loss per sample using unreduced criterion: `(per_sample_loss * sample_weights).sum() / sample_weights.sum()`.
- **Mean Teacher Asymmetry (N-5)**:
  - Apply `get_weak_augmentation` to compute `inputs_u_teacher` for the teacher model.
  - Apply `get_strong_augmentation` to compute `inputs_u_student` for the student model `net_A`.
  - Remove noisy cross-perturbation in CPS mode.
- **Physical Volume R² (V-5)**:
  - Extract per-sample physical voxel dimensions ($dx, dy, dz$) from image metadata (or `config.spacing`).
  - Calculate `pred_vol_mm3 = voxel_count * (dx * dy * dz)` and `gt_vol_mm3 = voxel_count * (dx * dy * dz)` per sample before computing R².
- **Physical HD95 Metric (V-6, V-7)**:
  - Pass physical spacing to `HausdorffDistanceMetric`.
  - If HD95 calculation returns `NaN` or fails (e.g., empty foreground target), set `val_hd95 = float('nan')` instead of `0.0`.
  - Update progress string printing to show `N/A` for `NaN` metrics.

---

### Component: Data Pipeline (`hassl/data/data_engine.py`)

#### [MODIFY] [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)
- **Imports & Caching (V-3)**:
  - Add `CacheDataset` to `from monai.data import ...`.
  - Catch `(MemoryError, RuntimeError, Exception)` when initializing `CacheDataset` and log warnings instead of bare `except Exception: pass`.
- **Patient ID Grouping Validation (V-11)**:
  - Use `patient_id_regex` from config if specified.
  - Add explicit assertions after patient grouping: fail loudly if patient count is 1 or equals number of total volumes when $N_{\text{volumes}} > 5$.
- **Unlabeled Transformations (N-5)**:
  - Set `unlabeled_transforms` to base transforms (unaugmented) so that `trainer.py` can apply weak augmentation to teacher and strong augmentation to student.

---

### Component: Web Application Server (`hassl/app/server.py`)

#### [MODIFY] [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)
- **Accept Endpoint Relocation (V-4 / A-1)**:
  - Remove `shutil.copy2(preseg_path, dest)` which copied to `data/labels/`.
  - Keep ONLY `shutil.copy2(preseg_path, dest_approved)` writing to `data/pseudo_approved/`.
  - Update volume metadata in memory to point to `dest_approved` with status `"pseudo_approved"`.

---

### Component: Configuration & Tracking (`hassl/config.py`, `hassl/tracking.py`, `hassl/data/nrrd_utils.py`)

#### [MODIFY] [config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py)
- Add `pseudo_label_weight: float = 0.5` field to `HASSLConfig`.
- Add `patient_id_regex: Optional[str] = None` field to `HASSLConfig`.

#### [MODIFY] [tracking.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/tracking.py)
- Replace deprecated `image.ptp()` with `np.ptp(image)` (V-13, NumPy 2.x fix).

#### [MODIFY] [nrrd_utils.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/nrrd_utils.py)
- Ensure numpy array passed to SimpleITK `GetImageFromArray` has 3D shape `(Z, Y, X)` (squeezing channel dimension if 4D) and handle RAS spatial metadata correctly (V-9).

---

## Verification Plan

### Automated Tests
- **Import Smoke Test**: Verify Python import check on all modules in `hassl`:
  ```powershell
  python -c "import pkgutil, importlib, hassl; [importlib.import_module(m.name) for m in pkgutil.walk_packages(hassl.__path__, prefix='hassl.')]"
  ```
- **NumPy 2.x Compatibility**: Verify `np.ptp` usage in `tracking.py`.
- **Config Dataclass Fields**: Verify `pseudo_label_weight` and `patient_id_regex` are serialized properly.

### Manual / Structural Verification
- Check that `server.py` accept endpoint writes strictly to `data/pseudo_approved/` and not `data/labels/`.
- Verify `CacheDataset` import and non-silent error handling in `data_engine.py`.
- Verify physical volume calculations and HD95 `NaN` handling in `trainer.py`.
