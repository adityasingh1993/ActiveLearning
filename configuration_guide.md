# HASSL Configuration Guide

This guide explains how to configure the **HASSL (Hybrid Active Semi-Supervised Learning)** pipeline for 3D medical image segmentation.

---

## 🎛️ Choosing the Right Configuration File

HASSL comes with two pre-configured master configuration files:

| File | Compute Tier | Recommended Hardware | Core Architecture | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **[`config.yaml`](file:///f:/Projects/Canvas/AcftiveLearningV1/config.yaml)** | **`prototype`** | **8GB VRAM** (RTX 3070/4060, T4) | Single DynUNet + UA-Mean Teacher | Fast prototyping, local desktop dev, quick AL iterations |
| **[`config_full.yaml`](file:///f:/Projects/Canvas/AcftiveLearningV1/config_full.yaml)** | **`full`** | **24GB VRAM** (RTX 3090/4090, A10G, V100) | Dual-Network CPS (DynUNet + SwinUNETR) | Production benchmarks, final paper metrics, high-res 3D |

---

## 📋 Complete Parameter Reference

### Data

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | `"./data"` | Root folder containing labeled, unlabeled, and pseudo-labeled dataset splits. |
| `image_suffix` | `str` | `".mha"` | File extension for 3D input image volumes. |
| `label_suffix` | `str` | `".seg.nrrd"` | File extension for 3D ground truth binary mask files. |
| `num_classes` | `int` | `1` | `1` for binary segmentation (sigmoid activation), `>1` for multi-organ (softmax). |
| `spacing` | `tuple` | `(1.0, 1.0, 1.0)` | Voxel physical resolution in mm per voxel (`Spacingd` target). |
| `label_names` | `dict` | `null` | Multi-class label map {0:'bg',1:'bladder',3:'urethra'}. None = binary mode. |
| `lcc_min_size_voxels` | `int` | `100` | LCC satellite filter: discard CCs smaller than this at spatial_size resolution. |
| `spatial_size` | `tuple` | `(128, 128, 128)` | Whole-volume resampled dimensions. |
| `preprocessing_mode` | `str` | `"resize"` | `"resize"` (Spacingd + Resized whole-volume) or `"patch"` (Spacingd + RandCrop). |
| `patch_size` | `tuple` | `(96, 96, 96)` | Sub-volume crop size when `preprocessing_mode == "patch"`. |
| `pos_neg_ratio` | `float` | `1.0` | Positive/negative sample crop ratio for `RandCropByPosNegLabeld`. |
| `val_split` | `int` | `5` | Number of gold labeled volumes held out for fixed validation. |
| `cache_dir` | `str` | `"./cache"` | PersistentDataset cache. |
| `use_cache_dataset` | `bool` | `true` | Caches preprocessed tensors in RAM/CacheDataset. |
| `patient_id_regex` | `str` | `null` | Custom regex for extracting patient ID prefix from volume filename. |
| `server_cache_max_volumes` | `int` | `20` | Max volumes held in server LRU image/mask cache. |

### Compute

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `compute_mode` | `str` | `"prototype"` | `"prototype"` (8GB) or `"full"` (24GB). |
| `device` | `str` | `"cuda"` | `"cuda"` or `"cpu"`. |
| `batch_size` | `int` | `1` | `1` for 8GB, `2` for 24GB. |
| `num_workers` | `int` | `2` | DataLoader workers. |
| `use_amp` | `bool` | `true` | Automatic Mixed Precision. |
| `use_gradient_checkpointing` | `bool` | `true` | Memory saving. |
| `seed` | `int` | `42` | Random seed. |

### Network

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `unet_backbone` | `str` | `"dynunet"` | `"unet"` or `"dynunet"`. |
| `unet_channels` | `tuple` | `(16, 32, 64, 128, 256)` | UNet channels. |
| `unet_strides` | `tuple` | `(2, 2, 2, 2)` | UNet strides. |
| `unet_num_res_units` | `int` | `2` | Only for UNet. |
| `dynunet_deep_supervision` | `bool` | `true` | Only for DynUNet. |
| `dropout` | `float` | `0.2` | MC Dropout rate. |
| `loss_type` | `str` | `"generalized_dice_focal"` | `"generalized_dice_focal"` or `"dice_ce"`. |
| `loss_lambda_gdl` | `float` | `1.0` | Weight for Generalized Dice Loss (volume-normalized). |
| `loss_lambda_focal` | `float` | `0.25` | Weight for Focal Loss (0.25 prevents 99% background mode collapse). |
| `loss_focal_gamma` | `float` | `2.0` | Focusing parameter for Focal Loss. |
| `swinunetr_feature_size` | `int` | `48` | Full mode additional network parameter. |

### SSL Pre-training (Phase 2)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ssl_epochs` | `int` | `100` | Pre-training epochs. |
| `ssl_lr` | `float` | `1e-4` | Learning rate. |
| `ssl_weight_decay` | `float` | `1e-5` | Weight decay. |
| `ssl_mask_ratio` | `float` | `0.3` | Fraction of volume to mask. |
| `ssl_mask_cube_size` | `int` | `16` | Size of each masked sub-cube. |
| `ssl_contrastive_temp` | `float` | `0.07` | InfoNCE temperature. |
| `ssl_embedding_dim` | `int` | `128` | Projection head output dim. |
| `ssl_use_early_stopping` | `bool` | `true` | Use early stopping. |
| `ssl_early_stopping_patience` | `int` | `20` | Early stopping patience. |

### Semi-Supervised Training (Phase 3)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `train_epochs` | `int` | `200` | Per AL round. |
| `train_lr` | `float` | `1e-4` | Learning rate. |
| `train_weight_decay` | `float` | `1e-5` | Weight decay. |
| `lr_scheduler` | `str` | `"cosine"` | `"cosine"`, `"plateau"`, or `"none"`. |
| `min_lr` | `float` | `1e-6` | Minimum learning rate. |
| `lr_warmup_epochs` | `int` | `5` | Warmup epochs. |
| `use_early_stopping` | `bool` | `true` | Early stopping. |
| `early_stopping_patience` | `int` | `30` | Patience. |
| `early_stopping_min_delta` | `float` | `1e-4` | Minimum delta. |
| `lambda_unsup` | `float` | `1.0` | Max weight for unsupervised loss. |
| `ema_decay` | `float` | `0.99` | EMA teacher momentum. Lower than 0.999 for small datasets. |
| `flexmatch_threshold` | `float` | `0.95` | Initial pseudo-label confidence threshold. |
| `mc_dropout_passes` | `int` | `5` | Passes for 8GB (`10` for 24GB). |
| `consistency_rampup_epochs` | `int` | `80` | Linear rampup for unsupervised weight. |
| `pseudo_label_weight` | `float` | `0.5` | Weight multiplier for pseudo-labeled approved samples. |
| `save_every_n_epochs` | `int` | `20` | Checkpoint frequency. |
| `log_image_every_n_epochs` | `int` | `10` | Log sample predictions. |

### Active Learning (Phase 4)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `al_query_size` | `int` | `10` | Volumes to query per round. |
| `al_rounds` | `int` | `3` | Total AL rounds (3-4 is optimal). |
| `al_strategy` | `str` | `"hybrid"` | `"bald"`, `"coreset"`, `"disagreement"`, `"hybrid"`. |
| `al_hybrid_weights` | `tuple` | `(0.4, 0.3, 0.3)` | BALD, CoreSet, Disagreement. |

### Pseudo-Label Promotion Quality Gates

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pseudo_confidence_threshold` | `float` | `0.85` | Min foreground certainty to promote. |
| `pseudo_mc_passes` | `int` | `5` | MC Dropout passes for epistemic gate. |
| `pseudo_mc_var_threshold` | `float` | `0.05` | Max allowed MC Dropout variance (epistemic). |
| `pseudo_tta_passes` | `int` | `8` | TTA passes for aleatoric gate. |
| `pseudo_tta_var_threshold` | `float` | `0.02` | Max allowed TTA variance (aleatoric). |
| `pseudo_tta_flip` | `bool` | `true` | Random flips during TTA. |
| `pseudo_tta_intensity_std` | `float` | `0.02` | Gaussian intensity jitter std during TTA. |

### Experiment Tracking (Phase 5)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tracker` | `str` | `"wandb"` | `"wandb"`, `"mlflow"`, or `"none"`. |
| `project_name` | `str` | `"hassl-ultrasound"` | Project name. |
| `experiment_name` | `str` | `"bladder-prototype"` | Experiment name. |
| `run_name` | `str` | `null` | Auto-generated if None. |
| `mlflow_tracking_uri` | `str` | `null` | MLFlow tracking URI. |

### Directories

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `checkpoint_dir` | `str` | `"./experiments/checkpoints"` | Saved models. |
| `log_dir` | `str` | `"./experiments/logs"` | Metrics and logs. |
| `preseg_dir` | `str` | `"./data/al_preseg"` | AL predictions. |
| `embedding_dir` | `str` | `"./experiments/embeddings"` | Pre-training embeddings. |

---

## 🚧 Common Pitfalls

- **`splits.json` Caching**: The `splits.json` file must be deleted if you change `val_split` or update your dataset. Otherwise, it will continue using the outdated dataset split.
- **`ema_decay` Tuning**: The default `ema_decay=0.99` is correct for small datasets (<100 labeled volumes). If you have a large dataset (>100 labeled volumes), use `ema_decay=0.999`.
- **`consistency_rampup_epochs`**: Default is `80`.
  - *Why rampup is needed*: At epoch 0, the teacher is randomly initialized and its pseudo-labels on unlabeled data are pure noise. Ramping up `unsup_weight` linearly from 0.0 to 1.0 prevents noisy unsupervised gradients from overwhelming the supervised signal in early training.
  - *Per-Round Reset*: **Rampup automatically resets to 0.0 at the start of every Active Learning (AL) round.** When a new round starts, newly annotated hard scans are added to the labeled pool. The student must first adapt to these new ground-truth labels during epochs 0–20 before the EMA teacher's updated predictions exert full consistency pressure.
  - *Mid-Round Resumes*: If an in-progress round is interrupted and resumed (e.g. at epoch 45), rampup resumes at `45 / 80 = 0.5625` rather than resetting to 0.
- **`preprocessing_mode` Selection**:
  - `resize`: The whole volume is resized to 128³ (or `spatial_size`). This is fast and good for small datasets.
  - `patch`: Uses native resolution crops. This is better for large structures but requires more epochs to train effectively.

---

## 🎯 Recommended Presets by Scenario

### Scenario A: Fast Desktop Prototyping (8GB GPU)
* **Config file**: `config.yaml`
* **Command**: `python -m hassl run-all --config config.yaml`
* **Key settings**: `compute_mode: "prototype"`, `batch_size: 1`, `mc_dropout_passes: 5`.

### Scenario B: 600+ Unlabeled Volumes Auto-Annotation
* **Config file**: `config.yaml` or `config_full.yaml`
* **Command**: `python -m hassl auto-train --config config.yaml`
* **Key settings**:
  ```yaml
  al_rounds: 4
  al_query_size: 100                    # Auto-promote 100 top volumes per round
  pseudo_confidence_threshold: 0.85      # Strict 85% confidence gate
  pseudo_label_weight: 0.5              # Safe 0.5 loss weighting for pseudo labels
  ```

### Scenario C: High-Resolution Anisotropic Anatomies (Patch Mode)
* **Config file**: Modify `preprocessing_mode` in `config.yaml`
* **Key settings**:
  ```yaml
  preprocessing_mode: "patch"
  patch_size: [96, 96, 96]
  pos_neg_ratio: 1.0
  ```
