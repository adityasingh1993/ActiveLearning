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

### 1. Data Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | `"./data"` | Root folder containing labeled, unlabeled, and pseudo-labeled dataset splits. |
| `image_suffix` | `str` | `".mha"` | File extension for 3D input image volumes. |
| `label_suffix` | `str` | `".seg.nrrd"` | File extension for 3D ground truth binary mask files. |
| `num_classes` | `int` | `1` | `1` for binary segmentation (sigmoid activation), `>1` for multi-organ (softmax). |
| `spacing` | `tuple` | `[0.100, 0.0386, 0.0982]` | Voxel physical resolution in mm per voxel (`Spacingd` target). |
| `spatial_size` | `tuple` | `[128, 128, 128]` | Whole-volume resampled dimensions. |
| `preprocessing_mode` | `str` | `"resize"` | `"resize"` (Spacingd + Resized whole-volume) or `"patch"` (Spacingd + RandCrop). |
| `patch_size` | `tuple` | `[96, 96, 96]` | Sub-volume crop size when `preprocessing_mode == "patch"`. |
| `pos_neg_ratio` | `float` | `1.0` | Positive/negative sample crop ratio for `RandCropByPosNegLabeld`. |
| `val_split` | `int` | `5` | Number of gold labeled volumes held out for fixed validation. |
| `use_cache_dataset` | `bool` | `true` | Caches preprocessed tensors in RAM for 5x-10x faster epoch speeds. |
| `patient_id_regex` | `str` | `null` | Regex string for extracting patient ID prefix to prevent data leakage across splits. |

---

### 2. Compute Parameters

| Parameter | Type | Prototype (8GB) | Full (24GB) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `compute_mode` | `str` | `"prototype"` | `"full"` | Switches pipeline architecture between 8GB single net & 24GB dual net. |
| `batch_size` | `int` | `1` | `2` | Batch size per GPU. |
| `use_amp` | `bool` | `true` | `true` | Automatic Mixed Precision (FP16) for speed and VRAM savings. |
| `use_gradient_checkpointing` | `bool` | `true` | `true` | Trades minimal compute for reduced activation VRAM overhead. |

---

### 3. Active Learning & Dual-Uncertainty Gates

| Parameter | Type | Default | Recommended Usage |
| :--- | :--- | :--- | :--- |
| `al_rounds` | `int` | `3` | Total Active Learning self-training cycles (3–4 is optimal). |
| `al_query_size` | `int` | `10` | Number of volumes auto-promoted or queried for human review per round. |
| `al_strategy` | `str` | `"hybrid"` | `"bald"` (Epistemic), `"coreset"` (Diversity), `"disagreement"` (Ensemble), or `"hybrid"`. |
| `pseudo_confidence_threshold` | `float` | `0.85` | **Gate 1**: Min mean foreground certainty (85%) to auto-promote pseudo labels. |
| `pseudo_mc_passes` | `int` | `5` (8GB) / `10` (24GB) | **Gate 2**: MC Dropout passes for parameter uncertainty estimation. |
| `pseudo_mc_var_threshold` | `float` | `0.05` | **Gate 2**: Max allowed epistemic variance. Volumes above this are rejected. |
| `pseudo_tta_passes` | `int` | `8` | **Gate 3**: Test-Time Augmentation passes for data noise estimation. |
| `pseudo_tta_var_threshold` | `float` | `0.02` | **Gate 3**: Max allowed aleatoric variance. Volumes above this are rejected. |
| `pseudo_tta_flip` | `bool` | `true` | Enables spatial axis un-flipping during TTA uncertainty evaluation. |
| `pseudo_tta_intensity_std` | `float` | `0.02` | Gaussian noise std added during TTA passes. |

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

---

## 🚀 Summary
All 53 parameters are supported across both configuration files and the `HASSLConfig` Python class. You can edit `config.yaml` directly or pass custom YAML overrides via `--config my_custom_config.yaml`.
