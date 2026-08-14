# HASSL: Step-by-Step Execution Guide (CLI & Browser)

This guide provides step-by-step instructions to run the HASSL (Hybrid Active Semi-Supervised Learning) pipeline via **CLI**, **3D Slicer**, and **Web Browser UI**.

---

## 1. Prerequisites & Environment Setup

### Step 1.1: Clone & Navigate to Project Directory
```bash
cd f:\Projects\Canvas\AcftiveLearningV1
```

### Step 1.2: Install Required Dependencies
Ensure you have Python 3.9+ and PyTorch installed, then run:
```bash
pip install -r requirements.txt
```

*This installs MONAI, PyTorch, SimpleITK, pynrrd, FastAPI, uvicorn, scikit-learn, and pytest.*

---

## 2. Dataset Setup

### Option 2A: Use Your Own Ultrasound Data
Organize your dataset inside the `data/` directory:
```
f:\Projects\Canvas\AcftiveLearningV1\data\
├── images/    <-- Place all ~300 image volumes (.mha files) here
└── labels/    <-- Place your initial label volumes (.seg.nrrd files) here
```

> [!WARNING]
> **Important:** If you change your dataset or update the `val_split` configuration, you MUST delete `data/splits.json` before your first run to allow the pipeline to regenerate the new splits.

### Option 2B: Generate Synthetic Ultrasound Data (For Quick Testing)
If you don't have real data ready yet, run the synthetic generator script to create 20 mock 3D ultrasound volumes with 5 labels:
```bash
python -m hassl.utils.synthetic_data --output ./data --num-volumes 20 --num-labeled 5
```

---

## 3. Configuration Setup

HASSL comes with two ready-to-use YAML configuration files:

- **`config.yaml`** (Default: Prototype mode for **8GB GPU** VRAM)
- **`config_full.yaml`** (Full mode for **24GB GPU** VRAM)

To inspect or edit parameters (e.g. image resolution spacing, batch size, or active learning query size), open `config.yaml` in any text editor.

---

## 4. Execution Workflows (Choose One Mode)

---

### MODE A: Fully Automated Self-Training (Zero Manual Labor)

Use this mode if you want the pipeline to train, automatically promote high-confidence pseudo-labels, and retrain **100% hands-free**.

#### Step 1: Run the Automated Loop
```bash
python -m hassl.pipeline --config config.yaml --phase auto-loop
```

#### What happens automatically:
1. Performs Self-Supervised Pre-training (SSL) on all 300 volumes.
2. Trains the initial model on your labeled volumes.
3. Evaluates pseudo-labels for unlabeled volumes.
4. Auto-promotes the top most confident pseudo-labels into `data/labels/`.
5. Retrains over multiple rounds until complete.

---

### MODE B: 3D Slicer Active Learning Loop (Desktop Workflow)

Use this mode if you want to inspect and correct AI pre-segmentations inside **3D Slicer**.

#### Step 1: Run Initial SSL & Semi-Supervised Training
```bash
python -m hassl.pipeline --config config.yaml --phase all
```

#### Step 2: Open 3D Slicer and Review Predictions
1. Launch **3D Slicer**.
2. Load the queried ultrasound volume: `data/images/volume_XXX.mha`.
3. Load the AI pre-segmentation mask: `data/al_preseg/volume_XXX.seg.nrrd`.
4. Correct any boundary errors using Slicer's Segment Editor (~3–5 mins per scan).
5. Save the corrected segmentation file to: `data/labels/volume_XXX.seg.nrrd`.

#### Step 3: Trigger Active Learning Retrain Round
Once you finish annotating the batch in Slicer, run:
```bash
python -m hassl.pipeline --config config.yaml --phase al-round --round 1
```

---

### MODE C: Browser-Based Web UI (No 3D Slicer Required)

Use this mode if you want to review and accept AI predictions **in your web browser** without installing desktop software.

#### Step 1: Start the Web UI Server
```bash
python -m hassl.pipeline --config config.yaml --phase serve
```
*Output will confirm: `Open: http://localhost:8000`*

#### Step 2: Open Your Browser
Open **`http://localhost:8000`** in Chrome, Edge, or Firefox.

#### Step 3: Annotate in the Browser
1. **Select a Volume**: Click any volume with the status **`preseg`** in the left sidebar queue.
2. **Scroll Slices**: Use your mouse wheel or `Arrow Up/Down` keys to scroll through 2D slices.
3. **Switch Views**: Click **Axial**, **Coronal**, or **Sagittal** tabs (or press keys `1`, `2`, `3`).
4. **Adjust Overlay**: Toggle the mask overlay on/off or adjust the opacity slider.
5. **Accept Label**: Click **"✅ Accept Label"** (or press key `A`). The server automatically saves the mask to `data/labels/` and updates the queue!
6. **Trigger Retrain**: Click **"🔄 Retrain Model"** in the top header once you finish a batch.

---

## 5. Phase-by-Phase CLI Reference

If you want to run specific phases individually:

```bash
# Phase 1: Run Self-Supervised Pre-training (SSL) on all volumes
python -m hassl.pipeline --config config.yaml --phase pretrain

# Phase 2: Run Semi-Supervised Training on current labeled set
python -m hassl.pipeline --config config.yaml --phase train

# Phase 3: Run Active Learning Query (find top informative scans)
python -m hassl.pipeline --config config.yaml --phase query --round 1

# Phase 4: Export AI Pre-segmentation masks for all unlabeled scans
python -m hassl.pipeline --config config.yaml --phase export-preseg

# Phase 5: Switch to 24GB GPU mode
python -m hassl.pipeline --config config_full.yaml --phase train
```


---

## 6. Monitoring Training (W&B / MLflow Curves)

Every epoch logs the following curves. Open your W&B project and group them as described below.

### 6.1 Core Validation Curves

| Curve | Expected behaviour | Action if not improving |
|:---|:---|:---|
| `val_dice` | Rises from ~0 toward 0.6–0.9 | Check `teacher_pseudo_conf` and `val_pred_fg_fraction` |
| `val_dice_lcc` | Slightly below or equal to `val_dice` after epoch ~30 | If 0.0 all the time, lower `lcc_min_size_voxels` |
| `val_recall` | Should improve alongside Dice | If recall is low but precision is high → threshold too high |
| `val_hd95` | Should decrease (smaller = better) | Noisy in early epochs; trust after epoch 50 |

### 6.2 Teacher vs Student Gap

Three new curves are logged every epoch (prototype / UA-MT mode only):

| Curve | What it measures | Healthy range |
|:---|:---|:---|
| `val_dice_teacher` | EMA teacher Dice on validation set | ≥ `val_dice` after warmup |
| `teacher_pseudo_conf` | Mean teacher sigmoid confidence on **unlabeled** voxels | Rises from ~0.5 → 0.85+ over first 80 epochs |
| `teacher_pseudo_fg_frac` | Fraction of unlabeled voxels the teacher labels as foreground | Should roughly match your real FG fraction |

Console output shows both side by side each epoch:
```
Epoch  15/200 | Loss: 0.7430 | Dice(S): 0.1820 | Dice(T): 0.2140 | Prec: 0.4210 | Rec: 0.2930 | ...
```

#### Reading the teacher / student gap

```
val_dice_teacher  >  val_dice    ✅ Healthy — EMA smoothing generalises better than raw student.
val_dice_teacher  ≈  val_dice    ✅ Fine — teacher closely tracks student.
val_dice_teacher  <  val_dice    ⚠️  EMA decay too high — teacher is lagging the student.
                                      Fix: reduce ema_decay (try 0.95 for very small datasets).
Both stuck near 0.0              ❌  Neither is learning foreground.
                                      Check teacher_pseudo_fg_frac — if near 0, teacher still
                                      predicts all background. Unsupervised loss may be dominating
                                      too early. Increase consistency_rampup_epochs.
teacher_pseudo_conf stays ~0.5   ❌  Teacher is still random. Too few supervised steps before
                                      unsupervised loss kicks in. Increase consistency_rampup_epochs.
teacher_pseudo_fg_frac >> real   ⚠️  Teacher over-predicts foreground (false positives).
                                      Lower pseudo_confidence_threshold or raise lambda_unsup.
```

### 6.3 Training Diagnostics

| Curve | What it measures |
|:---|:---|
| `consistency_rampup_weight` | Unsupervised loss weight: 0.0 at epoch 0 → 1.0 at `consistency_rampup_epochs` (resets to 0.0 at start of each AL round so student adapts to newly labeled scans first) |
| `uncertainty_mean` | Mean MC Dropout epistemic uncertainty per epoch (should decrease) |
| `supervised_loss` | Loss on labeled data (should decrease steadily) |
| `unsupervised_loss` | Consistency loss on unlabeled data (noisy in early epochs) |
| `train_fg_fraction` | Fraction of foreground voxels in labeled batch |
| `val_pred_fg_fraction` | Fraction of foreground in model predictions (watch for near-zero collapse) |

> **Note on Rampup Resets Across AL Rounds**:
> At the start of every new Active Learning round (Round 0 → Round 1 → Round 2), `consistency_rampup_weight` automatically resets to `0.0` at epoch 0. This gives the student model ~20 epochs to absorb newly annotated hard scans before the EMA teacher's consistency loss reaches full strength. If a training round is interrupted mid-round (e.g. at epoch 45), resuming resumes rampup at `45 / 80 = 0.5625` without resetting.

### 6.4 Recommended W&B Panel Layout

Create three panel groups in W&B:

**Panel 1 — Learning Progress**
- `val_dice` + `val_dice_teacher` (same plot) — teacher vs student
- `val_dice_lcc`
- `val_hd95`

**Panel 2 — Teacher Quality**
- `teacher_pseudo_conf`
- `teacher_pseudo_fg_frac`
- `consistency_rampup_weight`

**Panel 3 — Loss Dynamics**
- `supervised_loss`
- `unsupervised_loss`
- `uncertainty_mean`
- `learning_rate`

---

## 7. Troubleshooting

- **`validate Invertd failed for sample 0`**:
  This is now a warning only — metrics are not affected, only the native-space visualization preview. Usually caused by empty `applied_operations` on MetaTensor (e.g., using `CacheDataset` for validation — don't).

- **`val_dice_lcc = 0.0` in early epochs**:
  This is normal. LCC post-processing zeros predictions smaller than `lcc_min_size_voxels` (default 100 voxels). Watch `val_dice` (no LCC) instead for early training progress.

- **`model producing near-ZERO predictions`**:
  Printed when `val_pred_fg_fraction < 1e-4`. Usually caused by class imbalance (background dominates). Check that `include_background=False` in losses (already fixed in code).

- **`val_dice_teacher < val_dice` throughout training**:
  EMA decay is too high for your dataset size. Reduce `ema_decay` from `0.99` toward `0.95`. With `batch_size=1` and fewer than 5 labeled volumes, even `0.99` may be too slow.

- **`teacher_pseudo_conf` stuck at 0.5**:
  The teacher is still outputting near-uniform probabilities. This means the supervised signal has not had enough time to train the student before the unsupervised loss kicks in. Increase `consistency_rampup_epochs` (try 120–150 for very small datasets).

- **`teacher_pseudo_fg_frac` is near 0.0 after epoch 30**:
  The teacher is predicting all background on unlabeled data. The pseudo-labels are pure noise — the consistency loss is actively hurting training. Increase `consistency_rampup_epochs` and verify the supervised loss is decreasing.

---

## 8. Running Unit Tests

To verify that all modules, data loaders, neural network architectures, and active learning strategies are functioning cleanly on your CPU:

```bash
pytest tests/
```

Expected output:
```
============================= test session results ==============================
collected 24 items

tests/test_config.py ........                                            [ 33%]
tests/test_data_engine.py .....                                         [ 54%]
tests/test_nrrd_utils.py ..                                             [ 62%]
tests/test_trainer.py .....                                            [ 83%]
tests/test_losses.py ....                                               [100%]

============================== 24 passed in 3.42s ==============================
```
