# Implementation Plan — Round 10 Review Remediations

Remediate state lifecycle management across active learning rounds, implement missing features (warmup, gradient clipping), and fix the synthetic dataset end-to-end integration tests.

---

## Proposed Changes

### Component: Training & Lifecycle (`hassl/training/trainer.py` & `hassl/ssl/ssl_pretrainer.py`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)

1. **`EarlyStopping` Sticky Flag & Reset (V10-1)**:
   - In `EarlyStopping.__call__`:
     ```python
     if improved:
         self.best_score = val_score
         self.counter = 0
         self.early_stop = False  # Clear sticky flag on improvement
         return False
     ```
   - Add `reset()` method:
     ```python
     def reset(self):
         self.counter = 0
         self.best_score = None
         self.early_stop = False
     ```

2. **Checkpoint Loading vs. New AL Round Lifecycle (V10-1 & V10-2)**:
   - Update `load_checkpoint(self, path: str, weights_only: bool = False)`:
     - When `weights_only=True` (starting a new AL round or querying models):
       - Load network parameters (`net_A`, `teacher`, `net_B`) and `best_dice`.
       - Skip loading `optimizer`, `scheduler`, and `early_stopper` state dictionaries.
       - Re-create/reset scheduler (`self._build_scheduler(...)`) so `T_max` is fresh and initial learning rate is `train_lr` (not `1e-6`).
       - Reset `self.early_stopper.reset()` so round starts with `early_stop = False` and `counter = 0`.
   - Update `resume(self, path: str, weights_only: bool = False)` to forward `weights_only`.

3. **Gradient Clipping (V10-5)**:
   - In `train_one_epoch_uamt` and `train_one_epoch_cps`:
     ```python
     self.scaler.unscale_(self.optimizer)
     torch.nn.utils.clip_grad_norm_(self.net_A.parameters(), max_norm=12.0)
     self.scaler.step(self.optimizer)
     self.scaler.update()
     ```

4. **LR Scheduler Warmup (`lr_warmup_epochs`) (V10-4)**:
   - In `_build_scheduler(self, optimizer)`:
     - Read `warmup_epochs = getattr(self.config, 'lr_warmup_epochs', 0)`.
     - If `warmup_epochs > 0` and `scheduler_type == 'cosine'`:
       - Use `LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)` for warmup.
       - Use `CosineAnnealingLR(optimizer, T_max=max(1, train_epochs - warmup_epochs), eta_min=min_lr)` for main schedule.
       - Combine via `SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])`.

5. **Clean `if`/`else` for Schedulers (V10-6)**:
   - Replace inline conditional expressions with explicit `if`/`else` blocks.

---

### Component: Data Engine & Synthetic Data (`hassl/data/data_engine.py` & `hassl/utils/synthetic_data.py`)

#### [MODIFY] [data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py)
- Line 56: Change `if n_total_vols > 5:` to `if n_total_vols >= 5:` (V9-3 fix).

#### [MODIFY] [synthetic_data.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/utils/synthetic_data.py)
- Add optional `image_size: Optional[Tuple[int, int, int]] = None` parameter to `generate_synthetic_dataset` and `generate_single_volume` (V9-2 fix).
- Change synthetic volume IDs from `vol_000` to `US000_v1`, `US001_v1`, etc. so patient ID extraction (`split('_')[0]`) produces distinct patient IDs (`US000`, `US001`) (V9-3 fix).

---

### Component: CI Tests & Scripts (`tests/test_pipeline_ci.py` & `scripts/run_pre_commit.py`)

#### [MODIFY] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
- Fix import: `generate_synthetic_dataset` instead of `generate_synthetic_ultrasound_dataset` (V9-1 fix).
- Enable `test_full_pipeline_synthetic_end_to_end` with `image_size=(32, 32, 32)`.
- Assert on state dict contents and artifact files (`ckpt_file.stat().st_size > 0`, `pool_manifest.json` exists) (V9-4 fix).
- Add new unit tests for `EarlyStopping.reset()`, `load_checkpoint(weights_only=True)`, and LR scheduler warmup.

#### [MODIFY] [run_pre_commit.py](file:///f:/Projects/Canvas/AcftiveLearningV1/scripts/run_pre_commit.py)
- Add `pytest tests/ -q` execution step so failing tests block commits.

---

## Verification Plan

### Automated Tests
1. `python scripts/run_pre_commit.py` — runs ruff checks, import checks, and `pytest tests/ -q`.
2. `pytest tests/test_pipeline_ci.py -v` — all tests pass (including `test_full_pipeline_synthetic_end_to_end`).
