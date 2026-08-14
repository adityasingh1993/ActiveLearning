# Implementation Plan - HASSL Round 6 Remediation

Fix all critical defects identified in Round 6 Re-Review, wire missing augmentation tools, align Web UI axis loading, update manifest provenance for human corrections, and improve static analysis checks.

## User Review Required

> [!IMPORTANT]
> - **Fix Accept Endpoint Crash (V6-1)**: Fix `NameError: name 'dest' is not defined` in `server.py:409` by returning `{dest_approved}`.
> - **Persist Hand Corrections to Manifest & Training (V6-2 & V6-3)**: Update `slice_edit` endpoint to update `pool_manifest.json` with provenance `"human_corrected"`. In `trainer.py` and `data_engine.py`, assign `sample_weights = 1.0` to `"human_corrected"` masks (matching human gold quality).
> - **Fix Axis Mismatch / Data Corruption (A-3)**: Update `_load_mask` in `server.py` to use SimpleITK (`sitk.ReadImage` + `sitk.GetArrayFromImage`) first, ensuring both image and mask arrays are loaded in ITK `(Z, Y, X)` order. Add `assert mask.shape == image.shape` check at load time.
> - **Execute Weak/Strong Augmentation (N-5)**: Apply `self.weak_aug` to teacher input and `self.strong_aug` to student input in `train_one_epoch_uamt`.
> - **Derive Voxel Volume & Spacing from Affine Matrix (V-5 & V-6)**: Extract physical voxel volume (`|det(affine[:3,:3])|`) and spacing from MetaTensor `affine` for accurate post-resize $R^2$ and HD95 metrics.

## Proposed Changes

### Component: Web Application Server (`hassl/app/server.py`)

#### [MODIFY] [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)
- **Fix Accept Endpoint (V6-1)**:
  - Replace `{dest}` with `{dest_approved}` in `accept_volume` response message.
- **Update Manifest on Slice Edit (V6-2 & V6-3)**:
  - In `edit_mask_slice`, update `pool_manifest.json` setting `manifest["provenance"][vol_id] = "human_corrected"`, and updating `labeled_ids` / `unlabeled_ids` lists.
- **Fix Mask Axis Transposition (A-3)**:
  - In `_load_mask`, call `SimpleITK` first before `nrrd.read`, guaranteeing both image and mask return `(Z, Y, X)` arrays.
  - In `get_volume_info`, `get_slice`, and `edit_mask_slice`, assert `mask.shape == image.shape` to prevent transposed mask drawing.

---

### Component: Training & Dataset Engine (`hassl/data/data_engine.py`, `hassl/training/trainer.py`)

#### [MODIFY] [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)
- Support `"human_corrected"` in `build_labeled_dataset`: admit masks with `provenance` `"human"` or `"pseudo_approved"` or `"human_corrected"`.

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
- **Provenance Weighting (V6-3)**:
  - Set `sample_weights = 1.0` for `provenance in ["human", "human_corrected"]` and `pseudo_label_weight` ($0.5$) for `"pseudo_approved"`.
- **Mean Teacher Asymmetry (N-5)**:
  - Execute `self.weak_aug({"image": inputs_u[b]})` for teacher view `inputs_u_teacher` and `self.strong_aug({"image": inputs_u[b]})` for student view `inputs_u_student`.
- **Affine-derived Volume & Spacing (V-5 & V-6)**:
  - Compute `voxel_vol = float(torch.abs(torch.det(affine[:3, :3])).item())` and `spacing = tuple(torch.linalg.norm(affine[:3, i]).item() for i in range(3))` from `inputs.meta['affine']`.
  - Scale voxel sums by `voxel_vol` before computing volume $R^2$.

---

### Component: Verification & Pre-Commit (`scripts/run_pre_commit.py`, `tests/test_pipeline_ci.py`)

#### [MODIFY] [run_pre_commit.py](file:///f:/Projects/Canvas/AcftiveLearningV1/scripts/run_pre_commit.py)
- Add `ruff check hassl --select F821,F401` static analysis check to flag undefined names inside function bodies and unused imports.
- Narrow external module error skipping to exact `ModuleNotFoundError` for missing 3rd party packages.

#### [MODIFY] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
- Add geometry preservation test calling `write_mask_with_spatial_geometry` and asserting `GetSize()`, `GetSpacing()`, `GetDirection()`.
- Add negative test cases for `get_or_create_frozen_splits` validating patient grouping failure detection.

---

## Verification Plan

### Automated Tests
- Run `python scripts/run_pre_commit.py` to verify `ruff check` (F821, F401) and import smoke checks.
- Run `pytest tests/test_pipeline_ci.py` to verify patient grouping assertions, manifest provenance, and spatial geometry writing.

### Manual Verification
- Test `accept_volume` endpoint: verify HTTP 200 success response without `NameError`.
- Test `edit_mask_slice`: verify slice edits write to `data/pseudo_approved/` AND update `pool_manifest.json` with `"human_corrected"`.
