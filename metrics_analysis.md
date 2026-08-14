# Why Metrics Are Not Improving — Root Cause Analysis

## Summary

Several concrete issues work together to suppress Dice and Recall. Listed by severity.

---

## 🔴 Issue 1 — Unsupervised Loss Hijacking Training (Most Likely)

**File:** [`trainer.py` L398](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py#L398)

```python
loss = loss_sup + unsup_weight * loss_unsup
```

The unsupervised (MC Dropout teacher pseudo-label) loss is added **from epoch 0**, ramping from 0→1 over `consistency_rampup_epochs=30`. But the teacher at epoch 0 is just an EMA copy of a random-initialized student. Its pseudo-labels are **garbage** (random noise), and the student is now being pushed to match random predictions.

**DiceCELoss on pseudo-labels** computes cross-entropy too. At epoch 0, if the teacher predicts ~50% foreground randomly, the CE term may be *larger* than the supervised signal on 5 labeled volumes.

**Fix:** Raise `consistency_rampup_epochs` to at least 50–80. Or don't start the unsupervised loss until supervised Dice on labeled data exceeds, e.g., 0.3.

---

## 🔴 Issue 2 — `UncertaintyMaskedLoss` Collapses When DiceCELoss Returns a Scalar

**File:** [`losses.py` L52–54](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/losses.py#L52)

```python
if base_loss_per_voxel.ndim == 0:
    # Base loss aggregated to scalar (e.g. DiceCELoss reduction='mean')
    return base_loss_per_voxel * mask.mean()
```

`DiceCELoss` with `reduction='mean'` **always returns a scalar** (ndim=0). So the mask is reduced to `mask.mean()` — a single float. This means:

- The mask is **not applied voxel-wise** at all — it is just a global scaling factor
- If mean uncertainty > 0.75 quantile threshold (which is always 25% of voxels), `mask.mean()` ≈ 0.75, so the loss is attenuated by 25% regardless of spatial distribution
- **High-uncertainty regions are not actually excluded** from the loss

**Fix:** Either switch `DiceCELoss` to `reduction='none'` so the mask can be applied spatially, or accept that UncertaintyMaskedLoss is effectively a constant-factor scaler (not harmful, but doesn't help).

---

## 🔴 Issue 3 — DynUNet Deep Supervision: Main Head Not Extracted During Inference

**File:** [`trainer.py` L560–564](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py#L560)

```python
preds = inferer(inputs, self.net_A)
if isinstance(preds, (list, tuple)):
    preds = preds[0]
elif preds.ndim == 6:
    preds = preds[:, 0]
```

DynUNet with `deep_supervision=True` returns a `[B, num_heads, C, D, H, W]` 6D tensor. The code handles this. ✅

But during the **per-epoch foreground fraction diagnostic** at L1080–1083:

```python
_p = self.net_A(bd['image'].to(self.device))
if isinstance(_p, (list, tuple)): _p = _p[0]
elif _p.ndim == 6: _p = _p[:, 0]
_pb = (torch.sigmoid(_p) > 0.5).float()
```

This is correct ✅. However the loss computation in `train_one_epoch_uamt` at L350–354:

```python
preds_l = self.net_A(inputs_l)
for b in range(inputs_l.size(0)):
    p_b = preds_l[b:b+1] if torch.is_tensor(preds_l) else [p[b:b+1] for p in preds_l]
    l_b = compute_multiscale_loss(self.criterion, p_b, targets_l[b:b+1])
```

When `preds_l` is a 6D tensor `[B, heads, C, D, H, W]`, `preds_l[b:b+1]` gives shape `[1, heads, C, D, H, W]` — NOT a list. This goes to `compute_multiscale_loss`, which checks `torch.is_tensor(preds) and preds.ndim == 6` ✅. That handles it correctly.

---

## 🟠 Issue 4 — EMA Decay Too High for Small Datasets

**Config:** `ema_decay: float = 0.999`

With `ema_decay=0.999`, the teacher updates by only `0.1%` per step. With `batch_size=1` and e.g. 10 labeled volumes, one epoch = 10 steps. After 30 epochs = 300 steps. Teacher lag = ~300 steps * 0.001 = 0.3 effective student states. For small medical datasets, this means the **teacher lags far behind the student** for the first 50–100 epochs and provides useless pseudo-labels.

**Fix:** Reduce to `ema_decay=0.99` or use a warmup schedule (0.9 → 0.999 over 50 epochs).

---

## 🟠 Issue 5 — Binary Loss: `include_background=True` Causes Imbalance Dominance

**File:** [`losses.py` L126](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/losses.py#L126)

```python
self.dice_ce = DiceCELoss(
    include_background=False if num_classes > 1 else True,
    ...
)
```

For binary segmentation (`num_classes=1`), `include_background=True` means the Dice loss is computed over the background class AND the foreground class. For ultrasound segmentation where foreground (e.g. bladder) may be only 2–5% of volume, the background Dice is trivially high (0.97+), which **masks poor foreground Dice** in the loss signal.

**Fix:** For binary, set `include_background=False`. This forces the Dice loss to only optimize foreground overlap.

```python
self.dice_ce = DiceCELoss(
    include_background=False,  # Always exclude background for foreground-focus
    sigmoid=sigmoid, softmax=softmax,
    reduction=monai_reduction
)
```

> [!CAUTION]
> The cross-entropy term in DiceCELoss will still include background implicitly (it's a standard CE across all voxels). Only the Dice component is gated by `include_background`. This is the correct behavior.

---

## 🟠 Issue 6 — `lcc_min_size_voxels=100` May Be Zeroing Out All Predictions

**File:** [`trainer.py` L128–130](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py#L128)

```python
if largest_size >= min_size_voxels:
    out[b, c] = (labeled_arr == max_lab).astype(arr.dtype)
# else: all components are too small — output stays zero
```

If the model is still learning and producing small scattered predictions (< 100 voxels), **LCC zeros out the entire prediction**. This makes `val_dice_lcc` 0.0 and `val_recall_lcc` 0.0, which looks catastrophic in the logs.

The base metrics (`val_dice`, `val_recall` — no LCC) should still be non-zero. If both are bad, LCC isn't the primary cause but inflates perceived badness.

**Fix:** Log both `val_dice` (no LCC) and `val_dice_lcc` separately and focus on the non-LCC metric for early training assessment. LCC is a post-processing heuristic, not a training signal.

---

## 🟡 Issue 7 — Class Imbalance: No `pos_weight` in CE Term

The `DiceCELoss` uses standard cross-entropy without `pos_weight`. For 3D ultrasound where foreground is 2–5% of voxels, the CE gradient pushes heavily toward predicting background everywhere (mode collapse to all-zero prediction).

DiceLoss alone partially compensates but CE dominates in early epochs.

**Fix:** Add `pos_weight` based on observed foreground fraction:

```python
# If fg_fraction ≈ 0.03, pos_weight = (1 - 0.03) / 0.03 ≈ 32
pos_weight = torch.tensor([(1 - fg_frac) / (fg_frac + 1e-6)], device=device)
nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Or use `DiceFocalLoss` from MONAI which combines Focal + Dice and handles imbalance natively.

---

## 🟡 Issue 8 — Strong Augmentation May Be Too Aggressive

**File:** [`data_engine.py` L347–348](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py#L347)

```python
if is_training and apply_strong_aug:
    transforms.append(get_strong_augmentation(keys=keys))
```

Strong augmentation is applied to the **labeled training data** (as well as unlabeled). If it includes random elastic deformations, intensity shifts, or rotations, a small labeled set (5–10 volumes) may produce very distorted training samples, making it hard for the model to learn anatomy before the pseudo-label signal kicks in.

**Recommendation:** Check what `get_strong_augmentation` actually does. Consider using lighter augmentation on labeled data and reserving strong aug for the unlabeled stream only.

---

## Recommended Priority Fixes

| Priority | Issue | Fix |
|---|---|---|
| 🔴 1 | `include_background=True` causing imbalanced loss | Change to `False` in `CombinedSegLoss` |
| 🔴 2 | Unsupervised loss active too early with garbage teacher | Increase `consistency_rampup_epochs` to 80 |
| 🟠 3 | EMA decay too high for small dataset | Lower `ema_decay` to `0.99` |
| 🟠 4 | `UncertaintyMaskedLoss` ineffective on scalar loss | Accept or switch DiceCELoss to per-voxel |
| 🟡 5 | No pos_weight for class imbalance | Add `pos_weight` or switch to `DiceFocalLoss` |

