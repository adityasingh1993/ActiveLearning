# Implementation Plan - Fix Always-Zero Contrastive Loss

Fix the issue where `ssl_loss_contrastive` evaluates to `0.0` on every iteration when `batch_size=1`.

## Root Cause Analysis

In `hassl/ssl/ssl_pretrainer.py`:
1. `b1` and `b2` bottleneck features were pooled with global average pooling `F.adaptive_avg_pool3d(..., 1)`, producing single 1D feature vectors `(B, D_embed)`.
2. When `batch_size=1` ($B=1$), `feat1.size(0) == 1`.
3. `_infonce_loss` contained an early return check:
   ```python
   if feat1.ndim == 2 and feat1.size(0) == 1:
       return torch.tensor(0.0, device=feat1.device, requires_grad=True)
   ```
   This returned a dummy zero tensor on every single iteration when `batch_size=1`, disabling contrastive pre-training entirely.

## User Review Required

> [!IMPORTANT]
> - **Spatial Sub-Patch InfoNCE Contrastive Pooling**: Replace global 1D average pooling with $2 \times 2 \times 2$ spatial grid pooling (`F.adaptive_avg_pool3d(b, (2, 2, 2))`), yielding $8$ spatial patch representations per volume (`[B * 8, C_in]`).
> - Projection head `self.proj_head` projects each sub-patch embedding to `ssl_embedding_dim` (`[B * 8, ssl_embedding_dim]`).
> - InfoNCE loss matches identical spatial patches as positive pairs and different spatial patches within the volume / across batch volumes as negative pairs.
> - This guarantees non-zero InfoNCE contrastive loss and valid gradients for any batch size ($B \ge 1$).

---

## Proposed Changes

### Component: SSL Pretrainer (`hassl/ssl/ssl_pretrainer.py`)

#### [MODIFY] [ssl_pretrainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/ssl/ssl_pretrainer.py)
- In `SSLPretrainer.__init__`, update dry-run initialization for `self.proj_head` using sub-patch features `[8, C_in]`.
- In `_infonce_loss`: Remove early return of `0.0` when `feat1.size(0) == 1`.
- In `train()` step:
  - Pool `b1` and `b2` bottleneck feature maps to $2 \times 2 \times 2$ spatial grid (`F.adaptive_avg_pool3d(b, (2, 2, 2))`) and reshape to `[B * 8, C_in]`.
  - Pass `p1` and `p2` through `self.proj_head` to compute `feat1` and `feat2` (`[B * 8, ssl_embedding_dim]`).
  - Calculate `loss_cont = self._infonce_loss(feat1, feat2, temperature)`.

---

### Component: Automated CI Test Suite (`tests/test_pipeline_ci.py`)

#### [MODIFY] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
- Add `test_ssl_contrastive_loss_nonzero_at_batch_size_1()`:
  - Construct dummy `SSLPretrainer` with `batch_size=1`.
  - Perform 1 forward-backward step on synthetic volume `[1, 1, 32, 32, 32]`.
  - Assert `loss_cont > 0.0` and `loss_cont.grad_fn is not None`.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_pipeline_ci.py` to verify `test_ssl_contrastive_loss_nonzero_at_batch_size_1` passes.
- Run `python scripts/run_pre_commit.py` to ensure zero static analysis or import regressions.

### Manual Verification
- Inspect loss output in SSL pre-training step: confirm `Cont > 0.0` (e.g. `Cont: 1.8421`).
