# Implementation Plan - M-2, V6-8, V6-9, A-2, A-4, A-5

---

## M-2 — Configurable preprocessing: `Spacingd+Resized` vs. `RandCropByPosNegLabeld`

**New config field**: `preprocessing_mode: str = "resize"` (`"resize"` | `"patch"`)

In `hassl/data/data_engine.py`, `get_base_transforms()` branches on this field:

| Subset | `"resize"` mode (default) | `"patch"` mode |
|---|---|---|
| Training labeled | `Spacingd → Resized(spatial_size)` + augmentation | `Spacingd` + `RandCropByPosNegLabeld(patch_size)` + augmentation |
| Training unlabeled | `Spacingd → Resized(spatial_size)` | `Spacingd` + `RandSpatialCropd(patch_size)` |
| Validation | `Spacingd → Resized(spatial_size)` | `Spacingd → Resized(spatial_size)` *(always — for OOM-safe whole-volume inference)* |

**New config fields added to `hassl/config.py`**:
```python
preprocessing_mode: str = "resize"                  # "resize" or "patch"
patch_size: Tuple[int, int, int] = (96, 96, 96)     # Crop size when mode == "patch"
pos_neg_ratio: float = 1.0                           # Positive/neg sample ratio for RandCropByPosNegLabeld
```

---

## Invertd — Warp 128³ prediction back to original scan space for volume

**Problem**: Volume is currently counted on the **128³ resized** prediction tensor and multiplied by the post-resize voxel volume. Because `Resized` isotropically stretches the volume to fit 128³ regardless of original aspect ratio, the per-voxel volume of the resized grid is wrong relative to the original anatomy.

**Correct path** (applies only to validation, where `Resized` is always used):

```
LoadImaged stores → meta["original_affine"]  (original scanner affine)
Spacingd                                      (resamples to config.spacing)
Resized(spatial_size)                         (stretches to 128³)
↓ inference ↓
preds_binary [B, 1, 128, 128, 128]
↓ Invertd ↓
Resized⁻¹    → back to Spacingd resolution
Spacingd⁻¹   → back to original scanner resolution
↓
count voxels × |det(original_affine[:3,:3])|
```

**MONAI mechanism**:
- `Spacingd`, `Resized`, and `Orientationd` all implement `InvertibleTransform` — they record their parameters during forward and can reverse them.
- `Invertd(keys=["pred"], transform=val_transform)` applies the inverse of the entire preprocessing compose to the prediction only.
- Original affine is accessed via `inputs.meta["original_affine"][b]` (written by `LoadImaged(image_only=False)`).

**Implementation in `trainer.py` `validate()`**:
1. Build `self.val_transform = get_base_transforms(config, keys=["image", "label"], is_training=False)` once in `__init__`.
2. In `validate()`, after thresholding to `preds_binary`:
   - Build a `{"pred": preds_binary[b]}` dict and call `self.val_transform.inverse({"image": inputs[b], "pred": preds_binary[b]})` via MONAI's `Invertd`.
   - Extract `orig_affine = inputs.meta["original_affine"][b]`.
   - Compute `orig_voxel_vol = float(torch.abs(torch.det(orig_affine[:3, :3])).item())`.
   - Count `pv_mm3 = inverted_pred["pred"].sum() * orig_voxel_vol`.
3. HD95 and Dice metrics still run **on the 128³** space — that is correct because they are dimensionless overlap metrics that don't depend on physical scale. Only the physical volume measurement needs to go back to original space.

> [!NOTE]
> `Invertd` runs per-sample in a loop inside the existing `for b in range(inputs.size(0))` loop. It is applied to the prediction only, not the image (which is never used after validation finishes). No new memory allocation for images.

---

## V6-8 — Propagate `NaN` instead of `0.0` for Dice, precision, recall

In `validate()` (`trainer.py`):
- Line 502: `val_dice = 0.0 if ...isnan...` → `float('nan') if ...isnan...`
- Lines 505–506: `val_prec`/`val_rec` → `float('nan')` on NaN, consistent with `val_hd95`.
- In `train()` print statement: extend the `"N/A"` guard to dice/prec/rec:
  ```python
  dice_str  = f"{val_dice:.4f}"  if not np.isnan(val_dice)  else "N/A"
  prec_str  = f"{val_prec:.4f}"  if not np.isnan(val_prec)  else "N/A"
  rec_str   = f"{val_rec:.4f}"   if not np.isnan(val_rec)   else "N/A"
  ```

---

## V6-9 — Replace `mask_slice.tolist()` with RLE encoding

In `server.py`, the `/api/volume/{vol_id}/mask-slice` response currently ships a full 2D int array (~250 KB for 128²). Replace with run-length encoding:

```python
def _rle_encode(mask: np.ndarray) -> list:
    """Encode binary 2D mask to [[value, count], ...] RLE pairs (row-major order)."""
    flat = mask.ravel().tolist()
    rle = []
    cur_val = flat[0]; count = 0
    for v in flat:
        if v == cur_val:
            count += 1
        else:
            rle.append([cur_val, count])
            cur_val = v; count = 1
    rle.append([cur_val, count])
    return rle
```

Response changes from:
```json
{"mask": [0,0,1,1,...250k ints]}
```
to:
```json
{"mask_rle": [[0,5120],[1,48],[0,4096],...], "encoding": "rle", "shape": [128,128]}
```

In **`app.js`** (line 203), the mask-slice fetch handler `state.mask2D = data.mask` becomes:
```js
if (data.encoding === 'rle') {
    state.mask2D = rle_decode(data.mask_rle, data.shape);
} else {
    state.mask2D = data.mask;
}
```
with a `rle_decode(rle, shape)` helper that reconstructs a `shape[0] × shape[1]` 2D array.

---

## A-2 — Surface `data/pseudo/` in the UI

In `scan_volumes()`, extend preseg candidate list:
```python
str(data_dir / "pseudo" / f"{vol_id}.seg.nrrd"),
str(data_dir / "pseudo" / f"{vol_id}{label_suffix}"),
```
Volumes with masks found in `pseudo/` appear as `status = "preseg"` in the UI, entering the standard review → accept/reject workflow.

---

## A-4 — Bounded LRU cache for image/mask volumes

Replace `_state["cached_images"]` and `_state["cached_presegs"]` plain dicts with `LRUCache`:
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize: int = 20):
        self._cache = OrderedDict()
        self.maxsize = maxsize
    def get(self, key, default=None):
        if key not in self._cache: return default
        self._cache.move_to_end(key)
        return self._cache[key]
    def __contains__(self, key): return key in self._cache
    def __setitem__(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
    def __getitem__(self, key): return self.get(key)
    def pop(self, key, *args): return self._cache.pop(key, *args)
```
`maxsize` driven by new config field `server_cache_max_volumes: int = 20`.

---

## A-5 — Persist `/reject` to disk

In `reject_volume()`:
1. Write to `pool_manifest.json`:
   - `manifest["provenance"][vol_id] = "rejected"`
   - Append `vol_id` to `manifest.setdefault("rejected_ids", [])`
   - Remove from `unlabeled_ids` / `pseudo_ids` if present
2. Move preseg file to `data/rejected/{vol_id}{label_suffix}` using `shutil.move` as audit trail.

---

## Files Changed

| File | Changes |
|---|---|
| [config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py) | `preprocessing_mode`, `patch_size`, `pos_neg_ratio`, `server_cache_max_volumes` |
| [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py) | Configurable transform strategy; `RandCropByPosNegLabeld` / `RandSpatialCropd` in patch mode |
| [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py) | `Invertd` for original-space volume calculation; NaN propagation for Dice/prec/rec |
| [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py) | RLE mask encoding (V6-9), pseudo dir scan (A-2), LRU cache (A-4), persist reject (A-5) |
| [app.js](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/app.js) | RLE decode on mask-slice fetch; `rle_decode()` helper |

## Verification Plan
- `python scripts/run_pre_commit.py` — zero static analysis errors.
- `pytest tests/test_pipeline_ci.py` — all existing tests pass.
- New test `test_preprocessing_mode_configurable` — asserts patch mode produces `RandCropByPosNegLabeld` in training transform chain and `Resized` in val transform chain.
