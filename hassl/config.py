"""
HASSL Configuration Module.

Centralized YAML-driven configuration for all HASSL pipeline components.
Supports two compute tiers (prototype/full) and configurable network backbones.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml


@dataclass
class HASSLConfig:
    """Master configuration for the HASSL pipeline.

    This config drives all pipeline phases: data loading, SSL pre-training,
    semi-supervised training, active learning, and experiment tracking.

    Two compute tiers:
        - "prototype" (8GB VRAM): Single UNet/DynUNet + UA-Mean Teacher
        - "full" (24GB VRAM): Dual-network CPS (UNet/DynUNet + SwinUNETR)
    """

    # ─── Data ───────────────────────────────────────────────────────────
    data_dir: str = "./data"
    image_suffix: str = ".mha"
    label_suffix: str = ".seg.nrrd"
    num_classes: int = 1  # 1 = binary (sigmoid), >1 = multi-class (softmax)
    # Lowered 1.0 -> 0.1mm: measured native spacing was ~0.04-0.13mm, so resampling to 1.0mm
    # collapsed labeled structures to 14-156 foreground voxels post-preprocessing (confirmed
    # root cause of Dice stuck near 0). 0.1mm is the finest value the OOM guard in
    # data_engine.get_base_transforms allows before it skips Spacingd entirely.
    spacing: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # Spacingd pixdim (mm per voxel)
    label_names: Optional[Dict[int, str]] = None  # Multi-class label map {0:'bg',1:'bladder',3:'urethra'}
                                                   # None = binary mode (AsDiscreted threshold=0.5)
                                                   # dict = NormalizeLabelsInDatasetd remaps non-contiguous values
    lcc_min_size_voxels: int = 100  # LCC satellite filter: discard CCs smaller than this at spatial_size resolution
                                    # 100 is safe for medium targets (bladder/prostate ~1000-10000 voxels at 128^3)
                                    # Set to 0 to disable
    spatial_size: Tuple[int, int, int] = (128, 128, 128)  # Resize target (used in "resize" mode and for val/inference)
    # Default changed "resize" -> "patch": whole-volume resize can shrink a small/sparse
    # foreground structure below what GDL+Focal can recover from, producing all-background
    # collapse (val_pred_fg_fraction ~ 0) even with tuned loss weights. Patch mode guarantees
    # foreground-centered crops via RandCropByPosNegLabeld. See config.yaml for full rationale.
    preprocessing_mode: str = "patch"  # "resize" (Spacingd+Resized) or "patch" (Spacingd+RandCropByPosNegLabeld)
    # Lowered 96 -> 32: observed real volumes as small as (38,21,22) voxels post-Spacingd, so a
    # 96^3 patch was mostly wasted zero-padding (get_base_transforms now pads-before-crop so it
    # no longer hard-crashes, but a patch much bigger than the volume still isn't useful).
    patch_size: Tuple[int, int, int] = (32, 32, 32)  # Training crop size when preprocessing_mode == "patch"
    pos_neg_ratio: float = 2.0  # Positive/negative sample ratio for RandCropByPosNegLabeld (patch mode only).
                                 # Raised from 1.0 to bias sampling further toward foreground-containing crops.
    val_split: int = 5  # Number of labeled volumes held out for validation
    cache_dir: str = "./cache"  # PersistentDataset cache
    use_cache_dataset: bool = True  # Cache preprocessed tensors in RAM/CacheDataset
    patient_id_regex: Optional[str] = None  # Custom regex for extracting patient ID prefix from volume filename
    server_cache_max_volumes: int = 20  # Max volumes held in server LRU image/mask cache

    # ─── Compute ────────────────────────────────────────────────────────
    compute_mode: str = "prototype"  # "prototype" (8GB) or "full" (24GB)
    device: str = "cuda"  # "cuda" or "cpu"
    batch_size: int = 1  # 1 for 8GB, 2 for 24GB
    num_workers: int = 2
    use_amp: bool = True  # Mixed precision training
    use_gradient_checkpointing: bool = True
    seed: int = 42

    # ─── Network ────────────────────────────────────────────────────────
    unet_backbone: str = "dynunet"  # "unet" or "dynunet"
    unet_channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    unet_strides: Tuple[int, ...] = (2, 2, 2, 2)
    unet_num_res_units: int = 2  # Only for UNet
    dynunet_deep_supervision: bool = True  # Only for DynUNet
    dropout: float = 0.2  # For MC Dropout during uncertainty estimation
    loss_type: str = "generalized_dice_focal"  # "generalized_dice_focal" or "dice_ce"
    loss_lambda_gdl: float = 1.0  # Weight for Generalized Dice Loss
    loss_lambda_focal: float = 0.25  # Weight for Focal Loss (0.25 prevents 99% bg mode collapse)
    loss_focal_gamma: float = 2.0  # Gamma focusing parameter for Focal Loss
    include_boundary: bool = True  # Distance transform boundary loss for sharp edge contours
    boundary_weight: float = 0.5  # Weight multiplier for Boundary Loss
    prediction_threshold: float = 0.5  # Probability decision threshold for binary segmentation
    lcc_fill_holes: bool = True  # Apply 3D binary hole filling inside LCC organ masks

    # Full mode additional network
    swinunetr_feature_size: int = 48

    # ─── SSL Pre-training (Phase 2) ─────────────────────────────────────
    ssl_epochs: int = 100
    ssl_lr: float = 1e-4
    ssl_weight_decay: float = 1e-5
    ssl_mask_ratio: float = 0.3  # Fraction of volume to mask
    ssl_mask_cube_size: int = 16  # Size of each masked sub-cube
    ssl_contrastive_temp: float = 0.07  # InfoNCE temperature
    ssl_embedding_dim: int = 128  # Projection head output dim
    ssl_use_early_stopping: bool = True
    ssl_early_stopping_patience: int = 20

    # ─── Semi-Supervised Training (Phase 3) ──────────────────────────────
    train_epochs: int = 200  # Per AL round
    train_lr: float = 1e-4
    train_weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # "cosine", "plateau", or "none"
    min_lr: float = 1e-6
    lr_warmup_epochs: int = 5
    use_early_stopping: bool = True
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-4
    lambda_unsup: float = 1.0  # Max weight for unsupervised loss
    ema_decay: float = 0.99   # EMA teacher momentum. Lower than 0.999 for small datasets:
                               # with batch_size=1, ~10 labeled vols → 10 steps/epoch.
                               # At 0.999 the teacher needs ~700 steps (70 epochs) to reflect
                               # the student; at 0.99 it catches up in ~10 epochs.
                               # Use 0.999 only for large datasets (>100 labeled volumes).
    flexmatch_threshold: float = 0.95  # Initial pseudo-label confidence threshold
    mc_dropout_passes: int = 5  # 5 for 8GB, 10 for 24GB
    consistency_rampup_epochs: int = 80  # Linear rampup for unsupervised weight.
                                          # Keep low until teacher is trained: at epoch 30 the EMA
                                          # teacher on a small dataset is still producing noisy labels.
                                          # 80 epochs allows ~800 steps (batch_size=1, ~10 vols) for
                                          # the teacher to converge before full unsupervised weight.
    pseudo_label_weight: float = 0.5  # Weight multiplier for pseudo-labeled approved samples
    save_every_n_epochs: int = 20  # Checkpoint frequency
    log_image_every_n_epochs: int = 10  # Log sample predictions

    # ─── Active Learning (Phase 4) ───────────────────────────────────────
    al_query_size: int = 10  # Volumes to query per round
    al_rounds: int = 3  # Total AL rounds (3-4 is optimal)
    al_strategy: str = "hybrid"  # "bald", "coreset", "disagreement", "hybrid"
    al_hybrid_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)  # BALD, CoreSet, Disagreement

    # ─── Pseudo-Label Promotion Quality Gates ────────────────────────────
    pseudo_confidence_threshold: float = 0.85   # Min foreground certainty to promote
    pseudo_mc_passes: int = 5                   # MC Dropout passes for epistemic gate
    pseudo_mc_var_threshold: float = 0.05       # Max allowed MC Dropout variance (epistemic)
    pseudo_tta_passes: int = 8                  # TTA passes for aleatoric gate
    pseudo_tta_var_threshold: float = 0.02      # Max allowed TTA variance (aleatoric)
    pseudo_tta_flip: bool = True                # Random flips during TTA
    pseudo_tta_intensity_std: float = 0.02      # Gaussian intensity jitter std during TTA

    # ─── Experiment Tracking (Phase 5) ───────────────────────────────────
    tracker: str = "wandb"  # "wandb", "mlflow", or "none"
    project_name: str = "hassl-ultrasound"
    experiment_name: str = "bladder-prototype"
    run_name: Optional[str] = None  # Auto-generated if None
    mlflow_tracking_uri: Optional[str] = None  # e.g., "http://localhost:5000" or "file:./mlruns"

    # ─── Directories (auto-resolved) ────────────────────────────────────
    checkpoint_dir: str = "./experiments/checkpoints"
    log_dir: str = "./experiments/logs"
    preseg_dir: str = "./data/al_preseg"
    embedding_dir: str = "./experiments/embeddings"

    def __post_init__(self):
        """Validate and adjust config based on compute mode."""
        if self.compute_mode == "prototype":
            # Enforce 8GB-friendly settings
            if self.batch_size > 1:
                print(f"[HASSL Config] Overriding batch_size={self.batch_size} → 1 for prototype mode")
                self.batch_size = 1
            if self.mc_dropout_passes > 5:
                print(f"[HASSL Config] Overriding mc_dropout_passes={self.mc_dropout_passes} → 5 for prototype mode")
                self.mc_dropout_passes = 5
        elif self.compute_mode == "full":
            # Use higher settings for 24GB
            if self.batch_size < 2:
                self.batch_size = 2
            if self.mc_dropout_passes < 10:
                self.mc_dropout_passes = 10

        # Create directories
        for dir_attr in ["checkpoint_dir", "log_dir", "preseg_dir", "embedding_dir", "cache_dir"]:
            Path(getattr(self, dir_attr)).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "HASSLConfig":
        """Load config from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            HASSLConfig instance with values from YAML overriding defaults.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Convert nested tuples from YAML lists
        for key in ["spacing", "spatial_size", "patch_size", "unet_channels", "unet_strides", "al_hybrid_weights"]:
            if key in yaml_config and isinstance(yaml_config[key], list):
                yaml_config[key] = tuple(yaml_config[key])

        return cls(**yaml_config)

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to a YAML file.

        Args:
            yaml_path: Output path for the YAML file.
        """
        config_dict = asdict(self)
        # Convert tuples to lists for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to dictionary for experiment tracking."""
        return asdict(self)

    @property
    def is_binary(self) -> bool:
        """Whether this is a binary segmentation task."""
        return self.num_classes == 1

    @property
    def output_channels(self) -> int:
        """Number of output channels for the network.

        Binary: 1 channel (sigmoid), Multi-class: N channels (softmax).
        """
        return 1 if self.is_binary else self.num_classes

    @property
    def activation(self) -> str:
        """Activation function for the final layer."""
        return "sigmoid" if self.is_binary else "softmax"

    def get_full_mode_config(self) -> "HASSLConfig":
        """Return a copy of this config upgraded to full (24GB) mode.

        Useful for switching from prototype to full without editing YAML.
        """
        import copy
        full_config = copy.deepcopy(self)
        full_config.compute_mode = "full"
        full_config.batch_size = 2
        full_config.mc_dropout_passes = 10
        return full_config
