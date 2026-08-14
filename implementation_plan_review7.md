# Implementation Plan - HASSL Round 7 Remediation

Address the spatial alignment defect in Mean Teacher (V7-1), fix silent augmentation fallbacks (V7-2), add defensive shape guards to `edit_mask_slice` (V7-4), move determinism initialization to pipeline entry (V6-7), clean exception handlers (V7-6), and add a behavioral alignment CI test.

## User Review Required

> [!IMPORTANT]
> - **Spatial Alignment for Mean Teacher Views (V7-1)**: Separate spatial/geometric augmentations (`RandAffined`, `RandFlipd`, `RandRotated`) from intensity/photometric augmentations (`RandMultiplicativeSpeckleNoised`, `RandGaussianSmoothd`, `RandScaleIntensityd`, `RandAdjustContrastd`). Apply a *shared* spatial transform to both teacher and student views so voxel-wise consistency loss is evaluated in identical spatial coordinate frames.
> - **Augmentation Fallback Logging (V7-2)**: Add explicit warning logging to the `try/except` block in `train_one_epoch_uamt` when augmentation fails, preventing silent fallbacks.
> - **Edit Mask Slice Guard (V7-4)**: Add shape alignment guard `_verify_and_align_shape(image, mask)` to `edit_mask_slice` in `server.py`.
> - **Pipeline Entry Determinism (V6-7)**: Call `monai.utils.set_determinism(seed)` at pipeline entry in `hassl/pipeline.py` before `build_dataloaders` is invoked.
> - **Behavioral Alignment Test**: Add `test_teacher_student_views_are_spatially_aligned` in `tests/test_pipeline_ci.py` using an asymmetric 3D phantom to verify $>0.95$ geometric Dice overlap between teacher and student views.

---

## Proposed Changes

### Component: Data Augmentations (`hassl/data/augmentations.py`)

#### [MODIFY] [augmentations.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/augmentations.py)
- Create `get_spatial_augmentation(keys)` containing geometric transforms (`RandAffined`, `RandRotated`, `RandFlipd`).
- Create `get_intensity_augmentation(keys)` containing photometric transforms (`RandMultiplicativeSpeckleNoised`, `RandGaussianSmoothd`, `RandScaleIntensityd`, `RandAdjustContrastd`).
- Keep `get_weak_augmentation` and `get_strong_augmentation` for backwards compatibility.

---

### Component: Training Engine (`hassl/training/trainer.py`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
- In `HASSLTrainer.__init__`, instantiate `self.spatial_aug` and `self.intensity_aug`.
- In `train_one_epoch_uamt`:
  - Apply `self.spatial_aug` to `inputs_u[b]` to create a shared geometric base `spatial_tensor`.
  - Set `inputs_u_teacher = spatial_tensor`.
  - Apply `self.intensity_aug` to `spatial_tensor` to produce `inputs_u_student`.
  - Add explicit warning log if augmentation fallback occurs in `except Exception`.

---

### Component: Web Application Server (`hassl/app/server.py`)

#### [MODIFY] [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)
- Extract helper `_verify_and_align_shape(image, mask)` and invoke it in `get_slice`, `get_mask_slice`, AND `edit_mask_slice` (V7-4 fix).

---

### Component: Pipeline & Data Engine (`hassl/pipeline.py`, `hassl/data/data_engine.py`)

#### [MODIFY] [pipeline.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/pipeline.py)
- Call `monai.utils.set_determinism(seed=config.seed)` at top of pipeline entry functions before `build_dataloaders` (V6-7 fix).

#### [MODIFY] [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)
- Replace `except (MemoryError, RuntimeError, Exception) as e:` with `except Exception as e:` (V7-6 fix).

---

### Component: Automated CI Test Suite (`tests/test_pipeline_ci.py`)

#### [MODIFY] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
- Add `test_teacher_student_views_are_spatially_aligned()`: Creates an off-center asymmetric phantom, generates teacher/student views via spatial + intensity augmentations, and asserts geometric mask alignment ($\text{Dice} > 0.95$).

---

## Verification Plan

### Automated Tests
- Run `python scripts/run_pre_commit.py` to verify static analysis (`ruff F821/F401`) and import smoke checks.
- Run `pytest tests/test_pipeline_ci.py` to run all CI tests including the new spatial alignment assertion test.

### Manual Verification
- Inspect `inputs_u_teacher` and `inputs_u_student` generation in `trainer.py` to confirm zero spatial misalignment between views.
