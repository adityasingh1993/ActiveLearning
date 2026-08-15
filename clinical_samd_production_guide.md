# HASSL Clinical & Regulatory (FDA SaMD) Production Guide

---

## 1. Overview of Upgrades

This document details the architectural, clinical, and regulatory production enhancements introduced in branch **`feat/clinical-samd-enhancements`**. These upgrades transform the HASSL pipeline into a high-precision, clinically robust, and regulatory-auditable **Software as a Medical Device (SaMD)** system.

---

## 2. Detailed Technical & Clinical Upgrades

### 2.1 Dual-CNN CPS Backbone Option (`net_b_backbone: "res_dynunet"`)
- **Clinical Motivation**: In 3D B-mode ultrasound, acoustic attenuation, reverberation, and low tissue-to-tissue contrast make organ boundaries blurry. Vision Transformers (`SwinUNETR`) use patch embeddings that can introduce spatial artifacts at low-contrast boundaries.
- **Architectural Solution**: Added support for **`net_b_backbone: "res_dynunet"`** (DynUNet with residual units) in 24GB Cross-Pseudo Supervision (CPS) mode.
- **Benefits**: Retains the boundary extraction superiority of deep 3D convolutions while maintaining representation divergence between Net A and Net B.

### 2.2 Soft Centerline Topology Loss (`clDice`)
- **Clinical Motivation**: Ultrasound signal dropouts from bowel gas or calcifications can cause physical disconnections or internal gaps in the detrusor muscle wall.
- **Architectural Solution**: Implemented **`SoftCLDiceLoss`** in `hassl/training/losses.py`. Soft skeletonization via 3D max-pooling/min-pooling computes centerline overlap:
  $$\text{clDice} = \frac{2 \cdot \text{Precision}_{\text{skel}} \cdot \text{Sensitivity}_{\text{skel}}}{\text{Precision}_{\text{skel}} + \text{Sensitivity}_{\text{skel}}}$$
- **Benefits**: Enforces closed, watertight 3D organ topology and prevents topological fragmentation.

### 2.3 3D Binary Hole Filling in LCC Post-Processing (`lcc_fill_holes`)
- **Clinical Motivation**: Neural networks can predict tiny false-negative interior holes inside solid organs (bladder/prostate), degrading Recall and Dice.
- **Architectural Solution**: Integrated 3D binary hole filling (`scipy.ndimage.binary_fill_holes`) in `apply_keep_largest_cc`.
- **Benefits**: Fills interior false-negative gaps with $1.0$, boosting **Recall** and **Dice** without creating exterior boundary false positives.

### 2.4 Student-Teacher Ensemble Validation
- **Clinical Motivation**: Single-model predictions can exhibit frame-to-frame boundary noise.
- **Architectural Solution**: In prototype (UA-Mean Teacher) mode, validation computes ensemble probabilities:
  $$P_{\text{ensemble}} = \frac{1}{2} \left( \sigma(P_{\text{student}}) + \sigma(P_{\text{teacher}}) \right)$$
- **Benefits**: Cancels out single-model prediction spikes, boosting **Precision** and overall **Dice**.

### 2.5 ROI-Localized Active Learning Queries
- **Clinical Motivation**: Calculating BALD uncertainty across the entire pelvic volume allows distant bowel gas entropy to distort active learning candidate selection.
- **Architectural Solution**: Restrict uncertainty scoring strictly within the organ bounding box (`mean_preds > 0.05`).
- **Benefits**: Eliminates wasted queries on un-annotatable pelvic background noise.

### 2.6 Cryptographic Data Lineage Ledger (FDA SaMD / GMLP)
- **Regulatory Motivation**: FDA Software as a Medical Device (SaMD) and Good Machine Learning Practice (GMLP) guidelines require immutable audit trails for continuously updating AI algorithms.
- **Architectural Solution**: Manifest updates in `QueryEngine._save_manifest` compute SHA-256 transaction hashes (`latest_hash`).
- **Benefits**: Guarantees tamper-evident data lineage tracking across all active learning iterations.

---

## 3. Configuration Summary

```yaml
# In config.yaml
include_boundary: true            # Distance transform boundary loss for sharp organ surfaces
boundary_weight: 0.5              # Weight multiplier for Boundary Loss
include_cldice: false             # Soft Centerline Dice (clDice) loss for 3D topology preservation
cldice_weight: 0.3                # Weight multiplier for clDice Loss
prediction_threshold: 0.5         # Probability decision threshold for binary segmentation
lcc_fill_holes: true              # Apply 3D binary hole filling inside LCC organ predictions
enable_crypto_ledger: true        # SHA-256 cryptographic transaction hashing for data lineage
net_b_backbone: "res_dynunet"     # Secondary network backbone ("res_dynunet" Dual-CNN or "swinunetr")
```
