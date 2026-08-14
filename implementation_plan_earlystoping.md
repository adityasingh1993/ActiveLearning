# Implementation Plan - Early Stopping & Learning Rate Scheduling

Implement Early Stopping and Learning Rate Scheduling across HASSL semi-supervised training (`HASSLTrainer`) and self-supervised pre-training (`SSLPretrainer`), with full YAML-configurable fields in `HASSLConfig`.

## User Review Required

> [!IMPORTANT]
> - **Configuration Additions (`hassl/config.py`)**:
>   - `lr_scheduler`: `"cosine"`, `"plateau"`, or `"none"` (default: `"cosine"`)
>   - `min_lr`: Minimum learning rate floor (default: `1e-6`)
>   - `lr_warmup_epochs`: Warmup epochs for linear LR warmup (default: `5`)
>   - `use_early_stopping`: Enable early stopping for training (default: `True`)
>   - `early_stopping_patience`: Patience in epochs before stopping (default: `30`)
>   - `early_stopping_min_delta`: Minimum improvement threshold (default: `1e-4`)
>   - `ssl_use_early_stopping`: Enable early stopping for SSL pretraining (default: `True`)
>   - `ssl_early_stopping_patience`: SSL patience in epochs (default: `20`)
> - **Learning Rate Scheduling**: Log real-time learning rate to experiment trackers (`wandb`/`mlflow`).
> - **Early Stopping**: Stop training when validation performance plateaus to save compute and prevent overfitting.
> - **Checkpoint Persistence**: Save/restore `scheduler` and `early_stopper` state dictionaries in checkpoints (`roundX_latest.pth`).

---

## Proposed Changes

### Component: Configuration (`hassl/config.py`)

#### [MODIFY] [config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py)
Add LR scheduling and early stopping fields under SSL Pre-training and Semi-Supervised Training sections in `HASSLConfig`.

---

### Component: Early Stopping & Trainer (`hassl/training/trainer.py`)

#### [MODIFY] [trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py)
1. Implement `EarlyStopping` class tracking validation metrics (`mode='max'` for Dice, `'min'` for loss).
2. In `HASSLTrainer.__init__`:
   - Initialize LR scheduler(s) (`self.scheduler` or `self.scheduler_A` & `self.scheduler_B`).
   - Initialize `self.early_stopper = EarlyStopping(...)` if `config.use_early_stopping` is True.
3. In `HASSLTrainer.train()`:
   - Step scheduler at each epoch (`self.scheduler.step()` or `self.scheduler.step(val_dice)`).
   - Log current learning rate to tracker (`'learning_rate': current_lr`).
   - Check `self.early_stopper(val_dice)`. If triggered, log early stopping notification and break epoch loop.
4. Update `save_checkpoint` and `load_checkpoint` to save/restore `scheduler` and `early_stopper` states.

---

### Component: SSL Pre-training (`hassl/ssl/ssl_pretrainer.py`)

#### [MODIFY] [ssl_pretrainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/ssl/ssl_pretrainer.py)
1. In `SSLPretrainer.__init__`:
   - Initialize `self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.ssl_epochs, eta_min=config.min_lr)`.
   - Initialize `self.early_stopper = EarlyStopping(patience=config.ssl_early_stopping_patience, min_delta=1e-4, mode='min')`.
2. In `SSLPretrainer.train()`:
   - Step `self.scheduler.step()` at each epoch.
   - Check `self.early_stopper(avg_loss)`. Break loop if early stopping triggers.

---

### Component: Automated CI Test Suite (`tests/test_pipeline_ci.py`)

#### [MODIFY] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
Add `test_early_stopping_and_lr_scheduler()`:
- Construct `HASSLTrainer` with `train_epochs=50`, `early_stopping_patience=3`, `lr_scheduler="cosine"`.
- Verify LR decay and early stopping triggering when validation Dice plateaus.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_pipeline_ci.py` to verify `test_early_stopping_and_lr_scheduler` passes.
- Run `python scripts/run_pre_commit.py` to ensure `ruff F821/F401` static analysis passes cleanly.

### Manual Verification
- Run synthetic cohort train test: confirm learning rate decay and early stopping logs appear as expected.
