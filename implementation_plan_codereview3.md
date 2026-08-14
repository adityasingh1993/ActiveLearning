# Implementation Plan: Round 4 Code Review Remediation

Remediate all critical, major, and methodological defects identified in the Round 3 Code Review report.

## User Review Required

> [!IMPORTANT]
> 1. **P-1 Provenance Gate**: `build_labeled_dataset` will now strictly load `pool_manifest.json` and only ingest pseudo-labels marked `"pseudo_approved"`. Auto-promoted pseudo-labels will be written to `data/pseudo_unreviewed/` and human-approved ones to `data/pseudo_approved/`.
> 2. **P-1b Pseudo-Label Weighting**: Non-human labels will be weighted by `config.pseudo_label_weight` (default: 0.5) in the supervised loss computation.
> 3. **W-2b Greedy CoreSet Diversity**: `CoreSetStrategy` and `HybridStrategy` will preserve the greedy $k$-center selection order/ranks to guarantee intra-batch sample diversity.
> 4. **W-1b Spatial Geometry Inversion**: `write_mask_with_spatial_geometry` will resample preprocessed $128^3$ binary masks back into the native reference image's spatial dimensions and orientation before copying ITK physical metadata.
> 5. **N-5 Input Perturbation Asymmetry**: Unlabeled training stream will supply weakly augmented views to the EMA teacher and strongly augmented views to the student network.
> 6. **C-4 Patient-Level Holdouts**: `get_or_create_frozen_splits` will group volumes by patient ID prefix to eliminate data leakage across patient scans.

---

## Proposed Changes

### Component 1: Data Engine & Transforms (`hassl/data/`)

#### [MODIFY] [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)
- **Patient-level grouping (C-4)**: Update `get_or_create_frozen_splits` to extract patient prefixes (e.g., `case001` from `case001_slice1`) and perform patient-level splits for train/val/test.
- **Strict Provenance Gate (P-1)**:
  - Update `build_labeled_dataset` to load `pool_manifest.json`.
  - Only include pseudo-labels if `provenance[id] == "pseudo_approved"` and file exists in `data/pseudo_approved/`.
  - Include `"provenance"` in returned data dictionaries (`"human"` vs `"pseudo_approved"`).
- **Asymmetric Unlabeled Transforms (N-5)**:
  - Construct dual unlabeled dataloaders/transforms: weak augmentations for teacher inputs and strong augmentations for student inputs.

#### [MODIFY] [augmentations.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/augmentations.py)
- **Reproducible Transform (N-6)**: Subclass `monai.transforms.RandomizableTransform` in `RandMultiplicativeSpeckleNoised` using MONAI's `self.R` random generator.

#### [MODIFY] [nrrd_utils.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/nrrd_utils.py)
- **Spatial Geometry Inversion (W-1b)**:
  - Read `ref_img` with SimpleITK.
  - Resample $128^3$ `mask_arr` back to `ref_img.GetSize()` with nearest-neighbor interpolation in physical space.
  - Apply `CopyInformation(ref_img)` on the resampled mask.
  - Narrow `except Exception` to log explicit failure reasons.

---

### Component 2: Active Learning & Query Strategies (`hassl/active/`)

#### [MODIFY] [query_strategies.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/query_strategies.py)
- **Greedy CoreSet Diversity Fix (W-2b)**:
  - Update `CoreSetStrategy.query()` to dynamically update and return distance scores *after* the greedy selection loop.
  - Update `HybridStrategy.query()` to fuse rank-based greedy selection outputs so that the final queried batch retains intra-batch spatial diversity.

#### [MODIFY] [query_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/query_engine.py)
- Update output directories: write auto-promoted pseudo-labels to `data/pseudo_unreviewed/`.

---

### Component 3: Training & Web Application (`hassl/training/` & `hassl/app/`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
- **Weighted Supervised Loss (P-1b)**: Read `batch.get('provenance')` and multiply supervised loss by `0.5` for non-human labels.
- **Input Perturbation Asymmetry (N-5)**: Pass weakly augmented inputs to `teacher.forward_mc_dropout()` and strongly augmented inputs to `net_A`.

#### [MODIFY] [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)
- When a human accepts a volume via `/api/volume/{id}/accept`, move the mask from `data/pseudo_unreviewed/` to `data/pseudo_approved/` and update manifest provenance to `"pseudo_approved"`.

---

## Verification Plan

### Automated Tests
1. **Import & Syntax Verification**:
   ```powershell
   python -c "import ast; [ast.parse(open(f).read()) for f in ['hassl/data/data_engine.py', 'hassl/data/nrrd_utils.py', 'hassl/active/query_strategies.py', 'hassl/training/trainer.py']]"
   ```
2. **Provenance & CoreSet Unit Test**:
   - Run a unit test verifying that unreviewed pseudo-labels are ignored by `build_labeled_dataset()`.
   - Run a unit test asserting that `HybridStrategy.query()` selects diverse feature points.
3. **Geometry Match Test**:
   - Verify that SimpleITK resamples preprocessed masks back to native dimensions matching the reference image.

### End-to-End Synthetic Pipeline Execution
- Run `python -m hassl.pipeline --config config.yaml --phase all` over synthetic data to verify full pipeline execution.
