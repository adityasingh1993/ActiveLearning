# Structural & Server Fixes Walkthrough (M-2, V6-8, V6-9, A-2, A-4, A-5)

All 6 structural, metrics, and server debt items have been implemented and verified.

---

## 1. Summary of Changes

### M-2 & Configurable Preprocessing Pipeline
- Added `preprocessing_mode: str = "resize"` (`"resize"` | `"patch"`) to `HASSLConfig`.
- Added `patch_size: Tuple[int, int, int] = (96, 96, 96)` and `pos_neg_ratio: float = 1.0`.
- In `hassl/data/data_engine.py`:
  - `"resize"` mode: `Spacingd → Resized(spatial_size)` across all splits.
  - `"patch"` mode: `Spacingd` + `RandCropByPosNegLabeld` (labeled train) / `RandSpatialCropd` (unlabeled train). Validation dataset retains `Resized` for whole-volume `SlidingWindowInferer` execution without OOM.

### `Invertd` & Native Scan Volume Calculation
- Stored validation transform `self.val_transform` in `HASSLTrainer.__init__`.
- In `hassl/training/trainer.py` `validate()`:
  - Executed MONAI `Invertd(keys=["pred"], transform=self.val_transform)` per validation volume.
  - Warped the binary prediction tensor back through `Resized⁻¹ → Spacingd⁻¹` to the original scanner resolution.
  - Extracted `original_affine` from metadata to determine exact native voxel volume (`|det(original_affine[:3, :3])|`).
  - Physical foreground volume (`pv_mm3`) is calculated directly in native scanner space.

### V6-8: Metric Coercion Removal
- In `hassl/training/trainer.py` `validate()`:
  - Removed `0.0` coercions on `val_dice`, `val_precision`, and `val_recall`.
  - Replaced with `float('nan')` when metric tensors contain NaN or lack predictions.
  - Updated console output in `train()` to print `"N/A"` for any NaN metric.

### V6-9: RLE-Encoded Mask Slice Endpoint
- Added `_rle_encode(mask_slice)` helper in `hassl/app/server.py` to compress 2D slice masks to `[[val, count], ...]` run-length sequences.
- Updated `/api/volume/{vol_id}/mask_slice` endpoint to return `{"mask_rle": ..., "encoding": "rle", "shape": ...}`.
- Added `rle_decode(rle, shape)` in `hassl/app/static/app.js` to unpack RLE streams back into 2D mask arrays for canvas rendering.

### A-2: Pseudo-Mask Visibility in UI
- Updated `scan_volumes()` in `hassl/app/server.py` to check `data/pseudo/` as candidate pre-segmentation paths alongside `config.preseg_dir`.
- Auto-promoted active learning masks in `data/pseudo/` surface directly in the web UI under `status = "preseg"`.

### A-4: Bounded LRU Cache for Server Volumes
- Added thread-safe `LRUCache(maxsize)` backed by `collections.OrderedDict` in `hassl/app/server.py`.
- Added `server_cache_max_volumes: int = 20` setting in `HASSLConfig`.
- Replaced unbounded dicts `_state["cached_images"]` and `_state["cached_presegs"]` with `LRUCache` instances initialized during server startup.

### A-5: Persistent `/reject` Endpoint
- Updated `reject_volume()` in `hassl/app/server.py`:
  - Updates `pool_manifest.json` under `config.log_dir` with `provenance[vol_id] = "rejected"` and appends `vol_id` to `rejected_ids`.
  - Moves rejected pre-segmentation files to `data/rejected/` using `shutil.move` for audit trail.

---

## 2. Verification Results

### Automated CI Test Suite
Executed `python -m pytest tests/test_pipeline_ci.py -v --tb=short`:

```
================ 15 passed, 1 skipped in 8.17s ================
```

- **`test_preprocessing_mode_resize_has_resized_transform`**: PASSED
- **`test_preprocessing_mode_patch_train_labeled_has_rand_crop`**: PASSED
- **`test_preprocessing_mode_patch_val_still_has_resized`**: PASSED
- **`test_preprocessing_mode_patch_unlabeled_has_rand_spatial_crop`**: PASSED
- **`test_rle_encode_decode_round_trip`**: PASSED
- **`test_rle_encode_all_zeros`**: PASSED
- **`test_lru_cache_evicts_oldest`**: PASSED
- **`test_teacher_student_views_are_spatially_aligned`**: PASSED
- **`test_ssl_contrastive_loss_nonzero_at_batch_size_1`**: PASSED
- **`test_import_smoke_check`**: PASSED
