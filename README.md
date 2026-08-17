# HASSL: Hybrid Active Semi-Supervised Learning

> **Note:** This branch includes the QC-gated auto-labeling development workflow. The frozen 47-label segmentation baseline remains unchanged while QC and active-learning stages are evaluated on top of it.

## Auto-label / QC development workflow

After the frozen 47-case supervised CV baseline has been completed, the current development sequence is:

```text
47-case OOF segmentation predictions
        ↓
build_oof_qc_dataset.py
        ↓
train_qc_oof.py
        ↓
analyze_qc_auto_label_policy.py
        ↓
train_final_supervised_47.py
        ↓
run_auto_label_pool.py
```

### 1. Build the leak-safe OOF QC dataset

```bash
python scripts/build_oof_qc_dataset.py \
  --config config_resize128.yaml
```

The QC inputs contain only deployment-available features (probabilities, confidence/entropy, student-teacher disagreement, predicted volume, connected components, geometry, boundary confidence, and image/predicted-region intensity statistics). Ground-truth-derived Dice/precision/recall/RVE/HD95 remain `target_*` columns only.

### 2. Train/evaluate fold-aware QC

```bash
python scripts/train_qc_oof.py
```

Evaluation leaves one original segmentation fold out at a time, so the QC model never trains on the row it is scored on.

### 3. Calibrate the development three-way policy

```bash
python scripts/analyze_qc_auto_label_policy.py
```

The development policy produces:

- `HIGH_CONFIDENCE_PSEUDO_LABEL` / development AUTO_ACCEPT-equivalent candidates,
- `REVIEW`,
- `ACTIVE_LEARN_PRIORITY`.

These thresholds are development calibration only. A future locked set is required before production auto-accept.

### 4. Train one final segmentation model on all 47 labels

```bash
python scripts/train_final_supervised_47.py \
  --config config_resize128.yaml
```

The fixed training duration defaults to the median best epoch across the five frozen CV folds. The deployment source is the fixed-epoch `final_checkpoint.pth`, not the training-set-selected `best_checkpoint.pth`.

### 5. Run the first unlabeled-pool auto-label/QC pass

Start with a small smoke test:

```bash
python scripts/run_auto_label_pool.py \
  --config config_resize128.yaml \
  --limit 3
```

If the native-geometry checks pass, run the full pool:

```bash
python scripts/run_auto_label_pool.py \
  --config config_resize128.yaml \
  --overwrite
```

If unlabeled images are stored separately from `config.data_dir`, pass:

```bash
python scripts/run_auto_label_pool.py \
  --config config_resize128.yaml \
  --input-dir /path/to/unlabeled/images \
  --overwrite
```

The script uses the final student + EMA teacher 50/50 ensemble, extracts the same QC feature schema used during development, predicts Dice and failure probability, assigns the three-way bucket, and writes native-space `.seg.nrrd` predictions under:

```text
experiments/auto_label_pool_v1/
├── auto_label_manifest.csv
├── unlabeled_qc_features.csv
├── run_metadata.json
├── high_confidence_pseudo_label/
├── review/
└── active_learning_priority/
```

Native-space outputs are guarded strictly: the inverted mask must match the source voxel grid before export, and the saved segmentation size/spacing/origin/direction must match the reference image afterward.

---

The remainder of the original project documentation is available in the repository history and dedicated design/implementation documents.
