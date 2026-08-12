# HASSL: Architecture, Design Decisions & HLD/LLD Specification

---

## 1. Executive Summary

**HASSL (Hybrid Active Semi-Supervised Learning)** is a medical image segmentation framework specifically designed for 3D B-Mode Ultrasound images.

When provided with a dataset of **300 3D volumes where only 50 are labeled**, traditional supervised learning overfits, and basic uncertainty-based active learning selects redundant or noisy volumes. 

HASSL resolves this by integrating **Self-Supervised Pre-training (SSL)**, **Semi-Supervised Learning (SSL)**, and **Hybrid Active Learning (AL)** into a unified, dual-tier system.

---

## 2. Key Architectural Decisions & Rationale

### Decision 1: Hybrid Active Semi-Supervised Learning (HASSL) over Pure Active Learning
* **What we decided**: Combine SSL pre-training + Semi-supervised training + Active learning queries.
* **Why**: Active learning alone requires retrain cycles on very few labels (50 scans). By using SSL pre-training (masked inpainting + contrastive learning) on all 300 scans first, the model starts with rich anatomical representations. Semi-supervised training (UA-Mean Teacher / Cross-Pseudo Supervision) enforces consistency on the 250 unlabeled scans during every training step, maximizing accuracy before active learning even begins.

### Decision 2: Dual-Compute Tier System (8GB Prototype ↔ 24GB Production)
* **What we decided**: A single unified codebase configured via `config.yaml` (`compute_mode: "prototype"` or `"full"`).
* **Why**: The initial prototyping environment is constrained to an **8GB VRAM GPU**, but production training will utilize a **24GB VRAM GPU**. 
  * **Prototype Mode (8GB)**: Uses a single UNet/DynUNet + Exponential Moving Average (EMA) Teacher, AMP mixed precision, and 128³ volume resizing with batch size 1 (~6.2 GB VRAM).
  * **Full Mode (24GB)**: Scales up to dual-network Cross-Pseudo Supervision (Net A: DynUNet, Net B: SwinUNETR transformer), batch size 2, and 10 MC Dropout passes (~18.5 GB VRAM).

### Decision 3: Multi-Backbone Selection (UNet vs DynUNet vs SwinUNETR)
* **What we decided**: Configurable backbone selection via `unet_backbone: "dynunet"` or `"unet"`.
* **Why**: Standard UNet has fixed kernel sizes ($3\times3\times3$) and equal strides across all layers. **DynUNet** (MONAI's nnU-Net implementation) automatically adapts kernel sizes, filters, and deep supervision heads based on input spatial resolution, providing superior Dice scores on anisotropic ultrasound grids.

### Decision 4: Hybrid Active Learning Query Strategy
* **What we decided**: Combine **BALD** (epistemic uncertainty), **CoreSet** (representation diversity), and **Disagreement** (model discrepancy) into a weighted score: $0.4 \cdot \text{BALD} + 0.3 \cdot \text{CoreSet} + 0.3 \cdot \text{Disagreement}$.
* **Why**: Pure uncertainty sampling selects outlier volumes with heavy speckle noise or acoustic shadows. CoreSet ensures selected volumes span the entire feature embedding space, preventing redundant annotations.

### Decision 5: Three Flexible Operating Modes (Automated, Slicer, Web UI)
* **What we decided**: Build support for Option A (`--phase auto-loop`), Option B (`--phase al-round`), and Option C (`--phase serve`).
* **Why**: 
  * **Option A (Automated)**: 100% hands-free pseudo-label promotion for fast baseline experiments without human bottleneck.
  * **Option B (3D Slicer)**: Standard medical workflow exporting `.seg.nrrd` files for radiologist/researcher review in Slicer.
  * **Option C (Web UI)**: Light browser-based slice viewer (FastAPI + JS canvas) for review without installing desktop software.

### Decision 6: `.seg.nrrd` Parsing via `pynrrd`
* **What we decided**: Use `pynrrd` with custom header extraction for 3D Slicer segment files.
* **Why**: 3D Slicer `.seg.nrrd` files store detached headers and custom segment metadata tags (`Segment0_Name`, `Segment0_Color`, `Segment0_LabelValue`). Loading via raw NIfTI/ITK loaders ignores these segment mappings, causing binary label mismatches.

---

## 3. High-Level Design (HLD)

### 3.1 System Context Diagram

```
+-----------------------------------------------------------------------------------+
|                                   USER INTERFACE                                  |
|   CLI (`pipeline.py`)    |    FastAPI Web UI (`localhost:8000`)  |   3D Slicer App |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                              HASSL PIPELINE ORCHESTRATOR                          |
|                                     (`pipeline.py`)                               |
+-----------------------------------------------------------------------------------+
        │                                 │                                 │
        ▼                                 ▼                                 ▼
+---------------+               +-------------------+             +------------------+
|  DATA ENGINE  |               |  TRAINING ENGINE  |             |  ACTIVE LEARNING |
| (`data_engine`|               |   (`trainer.py`)  |             | (`query_engine`) |
+---------------+               +-------------------+             +------------------+
        │                                 │                                 │
        │ Load .mha / .seg.nrrd           │ UA-MT / CPS Loss                │ Hybrid Scoring
        ▼                                 ▼                                 ▼
+---------------+               +-------------------+             +------------------+
| MONAI Cache / |               |  Network Backbones|             | Feature Embeddings|
| PersistentDS  |               | (DynUNet/SwinUNETR|             | (t-SNE / UMAP)   |
+---------------+               +-------------------+             +------------------+
                                          │
                                          ▼
                                +-------------------+
                                |EXPERIMENT TRACKER |
                                | (WandB / MLflow)  |
                                +-------------------+
```

### 3.2 Subsystem Responsibilities

1. **Config Subsystem ([config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py))**: Central source of truth. Validates memory constraints, batch sizes, learning rates, spatial dimensions, and compute modes.
2. **Data Subsystem ([data_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/data/data_engine.py))**: Handles dictionary-based MONAI pipelines: `LoadImaged` → `EnsureChannelFirstd` → `Orientationd(RAS)` → `Spacingd` → `Resized(128³)` → `ScaleIntensityRangePercentilesd`.
3. **SSL Subsystem ([ssl_pretrainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/ssl/ssl_pretrainer.py))**: Pre-trains encoders on 300 unlabeled volumes using 3D sub-cube masking (30%), InfoNCE contrastive pairs, and 3D rotation classification.
4. **Semi-Supervised Subsystem ([trainer.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/training/trainer.py))**: Implements consistency regularization. Uses MC Dropout variance for voxel-wise uncertainty masking.
5. **Active Query Subsystem ([query_engine.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/active/query_engine.py))**: Manages dataset manifest `pool_manifest.json`, ranks unlabeled volumes via Hybrid scoring, and handles pseudo-label promotion / pre-seg export.
6. **Web App Subsystem ([server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py))**: FastAPI backend streaming 2D slices along axial, coronal, and sagittal planes with semi-transparent mask overlays.

---

## 4. Low-Level Design (LLD)

### 4.1 Dataclass & Configuration Schema ([config.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/config.py))

```python
@dataclass
class HASSLConfig:
    data_dir: str = "./data"
    image_suffix: str = ".mha"
    label_suffix: str = ".seg.nrrd"
    num_classes: int = 1
    spacing: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    spatial_size: Tuple[int, int, int] = (128, 128, 128)
    val_split: int = 5
    cache_dir: str = "./cache"
    
    compute_mode: str = "prototype"  # "prototype" (8GB) or "full" (24GB)
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 2
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    
    unet_backbone: str = "dynunet"  # "unet" or "dynunet"
    dropout: float = 0.2
    
    train_epochs: int = 200
    train_lr: float = 1e-4
    lambda_unsup: float = 1.0
    ema_decay: float = 0.999
    flexmatch_threshold: float = 0.95
    mc_dropout_passes: int = 5
    
    al_query_size: int = 10
    al_rounds: int = 3
    al_strategy: str = "hybrid"
    al_hybrid_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    
    tracker: str = "wandb"  # "wandb", "mlflow", or "none"
```

### 4.2 Class Interactions & Signatures

```
+------------------------------------------------------------------------------------+
|                                    HASSLTrainer                                    |
+------------------------------------------------------------------------------------+
| - net_A: nn.Module (UNet / DynUNet)                                                |
| - net_B: Optional[nn.Module] (SwinUNETR in Full mode)                              |
| - teacher: EMATeacher                                                              |
| - scaler: GradScaler                                                               |
| - flex_match: FlexMatchThreshold                                                   |
+------------------------------------------------------------------------------------+
| + train_one_epoch_uamt(epoch: int) -> Tuple[float, float, float, float]            |
| + train_one_epoch_cps(epoch: int) -> Tuple[float, float, float, float]             |
| + validate() -> float (Dice metric)                                                |
| + save_checkpoint(path: str)                                                       |
| + load_checkpoint(path: str)                                                       |
+------------------------------------------------------------------------------------+
                                       │
                                       │ Instantiates
                                       ▼
+------------------------------------------------------------------------------------+
|                                    QueryEngine                                     |
+------------------------------------------------------------------------------------+
| - manifest_path: str ("pool_manifest.json")                                        |
| - state: Dict (labeled_ids, unlabeled_ids, rounds)                                 |
+------------------------------------------------------------------------------------+
| + initialize_pool() -> Tuple[List[str], List[str]]                                 |
| + run_query(strategy, loader, k: int) -> Tuple[List[str], List[float]]             |
| + auto_promote_pseudo_labels(model, loader, k: int) -> List[str]                   |
| + export_presegmentation(model, loader, volume_ids, output_dir)                   |
+------------------------------------------------------------------------------------+
```

### 4.3 Tensor Transformations & Spatial Dimensions

```
Raw Input Volume (.mha)
Shape: (D_raw, H_raw, W_raw) e.g., (180, 256, 256)
                       │
                       ▼ LoadImaged + EnsureChannelFirstd
Shape: (1, D_raw, H_raw, W_raw)
                       │
                       ▼ Orientationd(RAS) + Spacingd(0.1, 0.1, 0.1)
Shape: (1, D_space, H_space, W_space)
                       │
                       ▼ Resized(128, 128, 128)
Shape: (1, 128, 128, 128)
                       │
                       ▼ ScaleIntensityRangePercentilesd(1, 99, 0, 1)
Tensor: Float32 in [0.0, 1.0]
                       │
                       ▼ Batched Dataloader (Batch Size = B)
Tensor: (B, 1, 128, 128, 128)
                       │
                       ▼ Network Forward Pass (UNet / DynUNet)
Output Logits: (B, num_classes, 128, 128, 128)
                       │
                       ▼ Sigmoid (Binary) / Softmax (Multi-class)
Predicted Mask: (B, 1, 128, 128, 128) uint8
```

---

## 5. Summary Table: High-Level vs Low-Level Specs

| Dimension | High-Level Design (HLD) | Low-Level Design (LLD) |
|:---|:---|:---|
| **Goal** | Efficient 3D ultrasound segmentation with 50 labels | Custom MONAI/PyTorch classes, loss functions, API routes |
| **Compute** | 8GB Prototype / 24GB Full mode switching | `GradScaler`, `autocast()`, batch size 1 vs 2, checkpointing |
| **Backbone** | Adaptable 3D convolutional network | `DynUNet` (deep supervision, 5 filters: 16→256), `SwinUNETR` |
| **Semi-Supervision**| Consistency regularization + uncertainty masking | `EMATeacher` ($\alpha=0.999$), `UncertaintyMaskedLoss`, `FlexMatch` |
| **Active Learning** | Hybrid selection (Uncertainty + Diversity + Discrepancy) | `BALDStrategy` ($T=5$), `CoreSetStrategy` (128-dimGAP), `HybridStrategy` |
| **Interfaces** | Automated Loop, 3D Slicer, Web Dashboard | CLI `pipeline.py`, `pynrrd` parser, FastAPI streaming `server.py` |
