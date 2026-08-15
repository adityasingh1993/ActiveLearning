# QC Gate & MedSAM Integration Guide

## 1. Overview

This guide explains the **Quality Control (QC) Gate** and **MedSAM Foundation Model Adapter** integrated into the HASSL active learning pipeline on branch `feat/qc-gate-medsam`.

Together, these systems create a **robust, clinically defensible, and FDA SaMD-auditable** segmentation pipeline by:

1. Automatically evaluating every prediction for clinical reliability before downstream use.
2. Routing failed predictions to a **human review queue** (exported to 3D Slicer).
3. Providing zero-shot cold-start bootstrap and cross-validation via MedSAM foundation models.

---

## 2. QC Gate Architecture

### 2.1 Module: `hassl/utils/qc_gate.py`

The `QCGate` runs 4 composable checkers (+ optional MedSAM) on every inference output.

```
QCGate
├── UncertaintyChecker    — MC Dropout entropy over foreground ROI
├── VolumeChecker         — Anatomical prior voxel count plausibility  
├── TopologyChecker       — Connected component count limit
├── ConvexityChecker      — Surface convexity ratio (shape regularity)
└── MedSAMCrossChecker    — Optional Dice agreement with MedSAM reference
```

### 2.2 QCReport Output

Every evaluation produces a `QCReport`:
```python
@dataclass
class QCReport:
    volume_id:      str          # Volume identifier
    passed:         bool         # Gate decision
    score:          float        # Composite quality score [0, 1]
    flags:          List[str]    # Which checkers failed
    checker_scores: Dict[str, float]  # Per-checker breakdown
    sha256_hash:    str          # FDA SaMD audit hash
```

### 2.3 Checker Details

| Checker | Signal | Clinical Rationale |
|:---|:---|:---|
| **Uncertainty** | MC Dropout entropy over foreground | Acoustic dropout / reverberation creates uncertain boundaries |
| **Volume** | Foreground voxel count vs. anatomical range | Physiologically implausible volumes = segmentation failure |
| **Topology** | Max connected component count | Bowel gas shadowing creates satellite label fragments |
| **Convexity** | mask volume / convex hull volume | Non-convex shapes from artifact boundaries |
| **MedSAM Dice** | HASSL Dice vs. MedSAM reference mask | Cross-model validation on high-uncertainty samples |

### 2.4 Composite Score

The composite score uses the **geometric mean** of all checker scores:
$$S_{\text{composite}} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \ln(s_i)\right)$$

This penalises any single weak link — a volume with a perfect volume score but near-zero convexity will still receive a low composite score.

### 2.5 SHA-256 Audit Hash

Every `QCReport` contains a `sha256_hash = SHA-256(volume_id + score + flags)`. This hash is:
- Stored in `human_review_queue.json` for each failed entry
- Logged to the experiment tracker
- Compliant with FDA GMLP immutable audit trail requirements

---

## 3. Human Review Queue

### 3.1 Module: `hassl/active/review_queue.py`

Persistent JSON-backed queue for QC-failed predictions.

```python
queue = ReviewQueue(queue_path="./experiments/logs/human_review_queue.json")

# Add QC-failed volume
queue.add_failed(
    volume_id="patient_042",
    qc_flags=["uncertainty", "topology"],
    qc_score=0.18,
    sha256_hash=report.sha256_hash,
    round_num=2,
    preseg_path="./preseg/patient_042_preseg.nii.gz",
)

# Clinician corrects the mask in 3D Slicer, then:
queue.mark_resolved("patient_042", corrected_label_path="./labels/patient_042.nii.gz")

# Check what's pending
pending = queue.get_pending()
queue.print_summary()
```

### 3.2 Status Lifecycle

```
PENDING → IN_REVIEW → RESOLVED
           ↑
     (clinician loads
      mask in Slicer)
```

The queue file also contains a `ledger_hash` (SHA-256 of all entries) for tamper detection.

---

## 4. MedSAM Integration

### 4.1 Module: `hassl/utils/medsam_adapter.py`

Thin adapter supporting both **SAM-Med3D** and **MedSAM2**.

#### Enabling MedSAM (opt-in)
```yaml
# config.yaml
medsam_enabled: true
medsam_variant: "sam_med3d"      # or "medsam2" for 24GB mode
medsam_checkpoint_dir: "./checkpoints/medsam"
medsam_qc_dice_threshold: 0.50
medsam_bootstrap_rounds: 1
```

Weights (~2–5GB) are downloaded automatically from HuggingFace on first use.

#### 4.2 Three Integration Points

| Mode | When | What It Does |
|:---|:---|:---|
| **Cold-start bootstrap** | Round 0..N (`medsam_bootstrap_rounds`) | Generates coarse pseudo-labels for unlabeled pool when < 5 labeled volumes |
| **QC cross-check** | Every `run_query` phase | Computes Dice between HASSL and MedSAM mask; low Dice → human review |
| **New cohort initialiser** | On-demand | Provides ROI seed mask from bounding-box prompt for unseen patient cohorts |

#### 4.3 Bounding Box Prompt Generation

```python
from hassl.utils.medsam_adapter import MedSAMAdapter

adapter = MedSAMAdapter(variant="sam_med3d")

# Auto-generate 3D bbox from prior prediction
bbox_min, bbox_max = MedSAMAdapter.generate_bbox_prompt(
    rough_mask=prior_pred,
    margin=10
)

# Run 3D inference
medsam_mask = adapter.infer_volume(volume_np, bbox_min, bbox_max)

# QC cross-check
dice = adapter.get_dice_vs(hassl_mask, volume_np)
```

#### 4.4 Variant Selection Guide

| Variant | VRAM | Best For |
|:---|:---|:---|
| `sam_med3d` | ~6GB | Default; 3D volumetric attention; best for full 3D ultrasound volumes |
| `medsam2` | ~10GB | 24GB mode; streaming memory propagation across slices; best for cine/4D |

---

## 5. Pipeline Flow Diagram

```
run_query()
    │
    ├── Active Learning Query (BALD + CoreSet + Disagreement)
    │
    ├── export_presegmentation() → ./preseg/
    │
    └── QC Gate Loop (for each queried volume):
            │
            ├── [MedSAM cross-check] (if medsam_enabled)
            │
            ├── QCGate.evaluate_tensor()
            │       ├── UncertaintyChecker
            │       ├── VolumeChecker
            │       ├── TopologyChecker
            │       ├── ConvexityChecker
            │       └── MedSAMCrossChecker
            │
            ├── PASS → pseudo-label pool eligible
            └── FAIL → ReviewQueue.add_failed()
                           → human_review_queue.json
                           → SHA-256 audit log
```

---

## 6. Tuning the QC Gate

Loosen the gate if too many valid predictions are being flagged:

```yaml
qc_uncertainty_threshold: 0.50    # Increase to be more lenient on entropy
qc_volume_min_voxels: 200          # Reduce for small/paediatric bladders
qc_max_cc_count: 5                 # Increase for multi-focal pathology
qc_surface_convexity_min: 0.55     # Reduce for pathologically-shaped organs
```

Tighten for high-confidence pseudo-label promotion:

```yaml
qc_uncertainty_threshold: 0.20    # Stricter entropy gate
qc_surface_convexity_min: 0.80    # Only very smooth surfaces pass
medsam_qc_dice_threshold: 0.65   # Higher Dice agreement required
```

---

## 7. File Reference

| File | Purpose |
|:---|:---|
| [`hassl/utils/qc_gate.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/utils/qc_gate.py) | QCGate, QCReport, all checkers |
| [`hassl/utils/medsam_adapter.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/utils/medsam_adapter.py) | MedSAM lazy-loader, prompt generation, Dice cross-check |
| [`hassl/active/review_queue.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/review_queue.py) | Persistent human review queue |
| [`hassl/pipeline.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/pipeline.py) | QC Gate wired into `run_query()` |
| [`tests/test_qc_gate.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_qc_gate.py) | QCGate unit tests |
| [`tests/test_medsam_adapter.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_medsam_adapter.py) | MedSAMAdapter unit tests |
| [`tests/test_review_queue.py`](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_review_queue.py) | ReviewQueue unit tests |
| [`config.yaml`](file:///f:/Projects/Canvas/AcftiveLearningV1/config.yaml) | Default QC gate + MedSAM config |
| [`config_full.yaml`](file:///f:/Projects/Canvas/AcftiveLearningV1/config_full.yaml) | 24GB mode config |
