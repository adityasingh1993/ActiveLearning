# TTA-Based Aleatoric Uncertainty Gate for Pseudo-Label Auto-Promotion

## Goal

Currently `auto_promote_pseudo_labels` filters candidates using only a single-pass
**foreground confidence score** (≥ 0.85). This misses cases where the model is
"confident" but the underlying data is inherently ambiguous (e.g. blurry boundaries,
probe angle noise in ultrasound).

We will add a **dual quality gate** modelled on MONAILabel's approach:

| Gate | What it measures | Threshold |
| :--- | :--- | :--- |
| **MC Dropout variance** (Epistemic) | Model parameter uncertainty — does the model agree with itself across dropout masks? | `mc_var < ε_epistemic` |
| **TTA variance** (Aleatoric) | Data-inherent noise — does the prediction change when the image is augmented? | `tta_var < ε_aleatoric` |
| **Foreground confidence** (Existing) | Is the model actually predicting a foreground organ? | `conf ≥ 0.85` |

Only volumes that pass **all three** gates are promoted.

---

## Proposed Changes

### `hassl/active/query_strategies.py`

#### [MODIFY] [query_strategies.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/query_strategies.py)

Add a new `TTAUncertaintyScorer` class after `BALDStrategy`:

```python
class TTAUncertaintyScorer:
    """Aleatoric Uncertainty via Test-Time Augmentation (TTA).

    Runs N augmented forward passes (random flips + intensity jitter)
    and measures the voxel-wise variance of the probability maps.
    High variance = data-inherent ambiguity (aleatoric uncertainty).
    """

    def __init__(self, model, num_passes: int = 8):
        self.model = model
        self.T = num_passes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random spatial + intensity augmentation for TTA."""
        import random
        aug = x.clone()
        # Random horizontal flip along D, H, W
        for dim in [2, 3, 4]:
            if random.random() > 0.5:
                aug = torch.flip(aug, dims=[dim])
        # Small intensity jitter
        aug = aug + torch.randn_like(aug) * 0.02
        return aug

    def score(self, x: torch.Tensor) -> np.ndarray:
        """Return mean voxel-wise variance across TTA passes per volume [B]."""
        self.model.eval()
        with torch.no_grad():
            preds = []
            for _ in range(self.T):
                aug_x = self._augment(x.to(self.device))
                out = self.model(aug_x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]
                prob = torch.sigmoid(out) if out.shape[1] == 1 else torch.softmax(out, dim=1)
                preds.append(prob)

            preds = torch.stack(preds, dim=0)          # [T, B, C, D, H, W]
            variance = preds.var(dim=0)                # [B, C, D, H, W]
            volume_variance = variance.mean(dim=(1, 2, 3, 4))   # [B]
            return volume_variance.cpu().numpy()
```

---

### `hassl/active/query_engine.py`

#### [MODIFY] [query_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/query_engine.py)

Update `auto_promote_pseudo_labels` signature and internals:

```python
def auto_promote_pseudo_labels(
    self,
    model,
    dataloader,
    k: int = 10,
    confidence_threshold: float = 0.85,
    mc_passes: int = 5,
    tta_passes: int = 8,
    epistemic_threshold: float = 0.05,
    aleatoric_threshold: float = 0.02,
):
```

**Logic per batch per volume**:
1. Single-pass sigmoid → foreground confidence score (existing).
2. `BALDStrategy.score(x)` → epistemic variance across `mc_passes` MC Dropout passes.
3. `TTAUncertaintyScorer.score(x)` → aleatoric variance across `tta_passes` TTA passes.
4. Only add to `candidates` if **all three gates pass**.
5. Store `mc_var` and `tta_var` in candidate dict for logging.

Log `mean_mc_var` and `mean_tta_var` alongside `mean_pseudo_confidence` to tracker.

---

## Verification Plan

### Automated Tests
- `python scripts/run_pre_commit.py` (ruff + pytest 47 tests)

### Manual Verification
- Confirm `auto_promote_pseudo_labels(model, loader, mc_passes=5, tta_passes=8)` runs without error on synthetic data.
- Check that low-confidence or high-uncertainty volumes are correctly excluded from `promoted_ids`.
