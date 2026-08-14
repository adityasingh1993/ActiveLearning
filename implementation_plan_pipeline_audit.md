# Fix: Metrics Not Improving — 7-Issue Training Pipeline Repair

## Problem

Dice and Recall are stagnating or falling during training despite the pipeline running without errors. 7 concrete bugs and misconfigurations are responsible.

---

## Proposed Changes

---

### Component 1 — `losses.py`

#### [MODIFY] [losses.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/losses.py)

**Fix 1a — `include_background=False` for binary Dice (Critical)**

`include_background=True` on ultrasound (foreground ≈ 2–5% of voxels) means the loss is dominated by trivially-high background Dice (0.97+). The gradient signal for foreground is drowned out. Change to `False` always.

**Fix 1b — `to_onehot_y=True` for multiclass DiceCELoss (Critical)**

Without `to_onehot_y=True`, multiclass targets (integer class indices) are fed directly as activation values to DiceLoss. Class index `2` is treated as a probability of 2.0. The resulting overlap computation is mathematically invalid.

**Fix 1c — Separate `reduction='none'` criterion for UncertaintyMaskedLoss (Critical)**

`CombinedSegLoss` currently uses `reduction='mean'` everywhere, returning a scalar. `UncertaintyMaskedLoss` then does `scalar_loss * mask.mean()` — this is just a global scaling factor, not voxel-wise masking. Noisy pseudo-labels contribute fully to the gradient. 

Solution: store a `dice_ce_unreduced` variant with `reduction='none'` and use it inside `UncertaintyMaskedLoss.forward()` so voxel-level mask exclusion actually works.

---

### Component 2 — `trainer.py`

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)

**Fix 2a — CPS mode: apply `_make_unlabeled_views` to unlabeled data (Critical)**

In `train_one_epoch_cps`, `inputs_u` is fed raw (no augmentation) to both `net_A` and `net_B`. CPS relies on perturbation-induced consistency — without augmentation, both networks see the identical image and quickly agree, collapsing pseudo-label diversity.

**Fix 2b — Respect `val_split` config in `get_or_create_frozen_splits` (High)**

The 15%-hardcoded split gives 1 validation patient on a 10-patient dataset, producing erratic Dice and causing early stopping to trigger prematurely.

---

### Component 3 — `config.py`

#### [MODIFY] [config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py)

**Fix 3a — Raise `consistency_rampup_epochs` default to 80 (High)**

At epoch 30 the teacher is barely trained yet unsupervised loss reaches full weight. Pseudo-labels from a poorly-trained teacher actively harm the student.

**Fix 3b — Lower `ema_decay` default to 0.99 (High)**

With batch_size=1 and ~10 labeled volumes = 10 steps/epoch, `ema_decay=0.999` means the teacher changes by only 0.001 per step. The teacher needs ~700 steps (70 epochs) to reflect current student. Lower to 0.99 so it converges after ~10 epochs.

---

## Verification Plan

### Automated
- Confirm `DiceCELoss(include_background=False, to_onehot_y=True)` no longer computes background Dice
- Confirm `UncertaintyMaskedLoss` now applies per-voxel mask when base loss has `reduction='none'`

### Manual
- Run 5 training epochs and verify:
  - `supervised_loss` is decreasing
  - `val_pred_fg_fraction` > 0.001 (model predicts foreground, not all-zero collapse)
  - `val_dice` on labeled set increasing from epoch 1 onward
  - `val_dice_lcc` increases once model produces predictions > `lcc_min_size_voxels`
