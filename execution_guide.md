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
└── labels/    <-- Place your 50 initial label volumes (.seg.nrrd files) here
```

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
2. Trains the initial model on your 50 labeled volumes.
3. Evaluates pseudo-labels for unlabeled volumes.
4. Auto-promotes the top 10 most confident pseudo-labels into `data/labels/`.
5. Retrains over multiple rounds (50 → 60 → 70 → 80 volumes) until complete.

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

# Phase 3: Run Active Learning Query (find top 10 informative scans)
python -m hassl.pipeline --config config.yaml --phase query --round 1

# Phase 4: Export AI Pre-segmentation masks for all unlabeled scans
python -m hassl.pipeline --config config.yaml --phase export-preseg

# Phase 5: Switch to 24GB GPU mode
python -m hassl.pipeline --config config_full.yaml --phase train
```

---

## 6. Running Unit Tests

To verify that all modules, data loaders, neural network architectures, and active learning strategies are functioning cleanly on your CPU:

```bash
pytest tests/
```

Expected output:
```
============================= test session starts =============================
collected 24 items

tests/test_config.py ........                                            [ 33%]
tests/test_data_engine.py .....                                         [ 54%]
tests/test_nrrd_utils.py ..                                             [ 62%]
tests/test_trainer.py .....                                            [ 83%]
tests/test_losses.py ....                                               [100%]

============================== 24 passed in 3.42s =============================
```
