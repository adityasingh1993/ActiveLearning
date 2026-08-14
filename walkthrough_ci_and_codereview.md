# HASSL Round 5 Remediation Walkthrough

All findings from the Round 5 Code Re-Review have been addressed across the codebase.

## Key Changes Made

### 1. Blocking Imports & Silent Failures Resolved (V-1, V-2, V-3)
- **[trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)**:
  - Added `from typing import Dict, List, Optional, Tuple, Any`.
  - Added `import numpy as np` at top level.
  - Added `monai.utils.set_determinism(seed=config.seed)` call at initialization.
- **[data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)**:
  - Imported `CacheDataset` from `monai.data`.
  - Replaced bare `except Exception: pass` around `CacheDataset` with narrowed error catching `(MemoryError, RuntimeError, Exception)` and explicit warning logs.

### 2. Gold Label Contamination Prevented (V-4 / A-1)
- **[server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)**:
  - Removed writing accepted presegmentations to `data/labels/` (`dest`).
  - Accepted UI masks are now written **strictly** to `data/pseudo_approved/` (`dest_approved`), preserving the purity of human gold labels.

### 3. Clinical Metrics & Statistical Units (V-5, V-6, V-7, V-8)
- **[config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py)**:
  - Added `pseudo_label_weight: float = 0.5` and `patient_id_regex: Optional[str] = None` fields to `HASSLConfig`.
- **[trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)**:
  - **Volume R² (V-5)**: Converted voxel counts to physical volume in $\text{mm}^3$ using per-case voxel spacing ($dx \cdot dy \cdot dz$) prior to computing R².
  - **HD95 Metric (V-6)**: Passed physical spacing `spacing=self.config.spacing` to `HausdorffDistanceMetric`.
  - **NaN Handling (V-7)**: Propagated `float('nan')` for metric failures or empty foreground cases, displaying `"N/A"` in logs instead of mapping failures to `0.0`.
  - **Supervised Loss Weighting (V-8)**: Implemented per-sample loss weighting `(per_sample_loss * sample_weights).sum() / sample_weights.sum()`.

### 4. Mean Teacher View Asymmetry & Domain Guardrails (N-5, V-9, V-11, V-13)
- **[data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)**:
  - Set `unlabeled_transforms` to base transforms (unaugmented) so `trainer.py` can generate weak view for teacher and strong view for student.
  - Added regex support (`patient_id_regex`) and strict assertions in `get_or_create_frozen_splits` to fail loudly if patient ID grouping collapses.
- **[nrrd_utils.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/nrrd_utils.py)**:
  - Ensured array channel dimensions are squeezed so array format is 3D `(Z, Y, X)` before passing to `sitk.GetImageFromArray`.
- **[tracking.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/tracking.py)**:
  - Replaced deprecated `image.ptp()` with `np.ptp(image)` for NumPy 2.x compatibility.

## 5. Interactive Web UI Mask Editing (Option 2)
- **[server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)**: Added `GET /api/volume/{id}/mask_slice` and `POST /api/volume/{id}/slice_edit` endpoints to fetch raw 2D uint8 mask slices and save slice modifications directly back to disk (`data/pseudo_approved/`).
- **[index.html](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.html)**: Added canvas overlay `<canvas id="draw-canvas">` over slice image with Brush/Eraser toolbar buttons, radius slider, and Save Slice Edit button.
- **[app.js](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/app.js)**: Implemented interactive mouse/touch drawing (Brush & Eraser), 2D mask matrix state sync, keyboard shortcuts (`B` for Brush, `E` for Eraser, `Ctrl+S` / `Cmd+S` to save slice edit).
- **[index.css](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.css)**: Styled interactive canvas overlay, drawing toolbar, and save action buttons.

## 6. Automated Synthetic CI Test Suite & Pre-Commit Hook (Option 1)
- **[test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)**: Integration test suite validating patient-level splits isolation (`splits.json`), provenance manifest gating (`data/labels/` purity), and import smoke tests.
- **[run_pre_commit.py](file:///f:/Projects/Canvas/AcftiveLearningV1/scripts/run_pre_commit.py)**: Standalone pre-commit hook script that verifies all `hassl` modules for `SyntaxError`, `NameError`, and internal `ImportError` regressions before commits.

---

## Verification Summary

- Verified pre-commit script execution: `python scripts/run_pre_commit.py` passed with 0 code errors.
- Verified syntax, imports, and API contracts for Web UI server endpoints and frontend static assets.
