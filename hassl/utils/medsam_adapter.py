"""
HASSL MedSAM Adapter Module.

Thin, framework-agnostic adapter for MedSAM foundation models:
    - SAM-Med3D : Full 3D volumetric transformer (3D spatial embeddings)
    - MedSAM2   : Memory-propagation model for video/slice coherence

Integration points in HASSL:
    1. Cold-start bootstrap: Generate coarse pseudo-labels for unlabeled pool
       on rounds 0..N (when labeled data < 5 volumes).
    2. QC cross-check: Validate HASSL predictions against MedSAM reference.
    3. Pre-segmentation initialiser: Provide ROI seed masks for new cohorts.

Design:
    - Lazy loading: model weights are NOT downloaded at import time.
    - Opt-in: controlled by `medsam_enabled: false` in config (default off).
    - Swap-friendly: MedSAM2 and SAM-Med3D share the same MedSAMAdapter API.
    - Graceful fallback: if weights are unavailable, returns a dummy all-zero mask
      and logs a warning rather than crashing the pipeline.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MEDSAM_HUGGINGFACE = {
    "sam_med3d": "blueyo0/SAM-Med3D",
    "medsam2": "bowang-lab/MedSAM2-Tiny",
}

_MEDSAM_CKPT_FILENAMES = {
    "sam_med3d": "sam_med3d_turbo.pth",
    "medsam2": "MedSAM2_tiny.pth",
}


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary Dice coefficient."""
    inter = float((a * b).sum())
    denom = float(a.sum() + b.sum()) + 1e-8
    return 2.0 * inter / denom


class MedSAMAdapter:
    """Lightweight adapter for MedSAM-family foundation models.

    Supports SAM-Med3D and MedSAM2. Model weights are lazy-loaded on first
    inference call from HuggingFace Hub or a local checkpoint directory.

    Args:
        variant: "sam_med3d" (default) or "medsam2".
        checkpoint_dir: Local directory to cache downloaded weights.
        device: Torch device string (default "cuda" if available else "cpu").
        fp16: Whether to use half-precision inference (reduces VRAM, default False).
    """

    def __init__(
        self,
        variant: str = "sam_med3d",
        checkpoint_dir: str = "./checkpoints/medsam",
        device: Optional[str] = None,
        fp16: bool = False,
    ):
        if variant not in _MEDSAM_HUGGINGFACE:
            raise ValueError(
                f"Unknown MedSAM variant '{variant}'. "
                f"Choose from: {list(_MEDSAM_HUGGINGFACE)}"
            )
        self.variant = variant
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fp16 = fp16
        self._model = None  # Lazy-loaded on first use
        self._device = device

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    @classmethod
    def from_config(cls, config) -> "MedSAMAdapter":
        """Construct adapter from a HASSLConfig instance."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = getattr(config, "compute_mode", "standard") != "full"
        return cls(
            variant=getattr(config, "medsam_variant", "sam_med3d"),
            checkpoint_dir=getattr(config, "medsam_checkpoint_dir", "./checkpoints/medsam"),
            device=device,
            fp16=fp16,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Weight Management
    # ──────────────────────────────────────────────────────────────────────

    def _get_checkpoint_path(self) -> Path:
        """Return local checkpoint path, downloading from HuggingFace if needed."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = _MEDSAM_CKPT_FILENAMES[self.variant]
        ckpt_path = self.checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            logger.info(
                "MedSAMAdapter: checkpoint not found locally. Downloading '%s' from HuggingFace Hub...",
                _MEDSAM_HUGGINGFACE[self.variant],
            )
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
                hf_hub_download(
                    repo_id=_MEDSAM_HUGGINGFACE[self.variant],
                    filename=ckpt_name,
                    local_dir=str(self.checkpoint_dir),
                )
            except Exception as e:
                logger.warning(
                    "MedSAMAdapter: HuggingFace download failed (%s). "
                    "Returning dummy mask. Place checkpoint manually at: %s",
                    e,
                    ckpt_path,
                )
        return ckpt_path

    def _load_model(self) -> None:
        """Lazy-load MedSAM model weights."""
        if self._model is not None:
            return

        ckpt_path = self._get_checkpoint_path()
        if not ckpt_path.exists():
            logger.warning(
                "MedSAMAdapter: checkpoint file missing at %s. "
                "Adapter will return dummy masks until weights are available.",
                ckpt_path,
            )
            self._model = None
            return

        try:
            import torch
            logger.info("MedSAMAdapter: loading %s from %s ...", self.variant, ckpt_path)
            state_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)

            if self.variant == "sam_med3d":
                # SAM-Med3D image encoder + mask decoder
                from segment_anything import sam_model_registry  # type: ignore
                self._model = sam_model_registry["vit_b"](checkpoint=None)
                self._model.load_state_dict(state_dict, strict=False)
            elif self.variant == "medsam2":
                # MedSAM2: build_sam2 from sam2 package
                from sam2.build_sam import build_sam2  # type: ignore
                self._model = build_sam2(
                    "sam2_hiera_t.yaml",
                    str(ckpt_path),
                    device=self.device,
                )

            self._model = self._model.to(self.device)
            if self.fp16:
                self._model = self._model.half()
            self._model.eval()
            logger.info("MedSAMAdapter: %s loaded successfully.", self.variant)

        except Exception as e:
            logger.warning(
                "MedSAMAdapter: failed to load model (%s). Returning dummy masks.", e
            )
            self._model = None

    # ──────────────────────────────────────────────────────────────────────
    # Prompt Generation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_bbox_prompt(
        rough_mask: Optional[np.ndarray] = None,
        volume_shape: Optional[Tuple[int, int, int]] = None,
        margin: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a 3D bounding-box prompt from a rough organ mask or full volume.

        Args:
            rough_mask: Optional binary rough mask [D, H, W] (e.g., from prior prediction).
                        If None, a central-third bounding box is used.
            volume_shape: [D, H, W] shape required when rough_mask is None.
            margin: Voxel margin to add around detected bounding box.

        Returns:
            (bbox_min, bbox_max): Two [3] int arrays representing the 3D bounding box corners.
        """
        if rough_mask is not None and rough_mask.any():
            nz = np.argwhere(rough_mask)
            bbox_min = np.maximum(nz.min(axis=0) - margin, 0)
            bbox_max = np.minimum(nz.max(axis=0) + margin, np.array(rough_mask.shape) - 1)
        else:
            assert volume_shape is not None, "volume_shape required when rough_mask is None"
            d, h, w = volume_shape
            bbox_min = np.array([d // 4, h // 4, w // 4])
            bbox_max = np.array([3 * d // 4, 3 * h // 4, 3 * w // 4])
        return bbox_min.astype(int), bbox_max.astype(int)

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def infer_volume(
        self,
        volume_np: np.ndarray,
        bbox_min: Optional[np.ndarray] = None,
        bbox_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run prompted 3D segmentation inference on a single volume.

        Args:
            volume_np: Normalised input volume [D, H, W] in [0, 1].
            bbox_min: 3D bounding box minimum corner [3]. Auto-generated if None.
            bbox_max: 3D bounding box maximum corner [3]. Auto-generated if None.

        Returns:
            Binary segmentation mask [D, H, W] in {0, 1}, dtype float32.
        """
        self._load_model()

        # Graceful fallback: return zeros if model unavailable
        if self._model is None:
            logger.warning(
                "MedSAMAdapter: model not loaded; returning dummy zero mask."
            )
            return np.zeros(volume_np.shape, dtype=np.float32)

        # Auto-generate bounding box from image centre if not provided
        if bbox_min is None or bbox_max is None:
            bbox_min, bbox_max = self.generate_bbox_prompt(volume_shape=volume_np.shape)

        try:
            import torch
            with torch.no_grad():
                # Normalise volume to [-1, 1] if needed
                v = volume_np.astype(np.float32)
                v = (v - v.mean()) / (v.std() + 1e-8)

                # Build simple slice-by-slice prompt for MedSAM-family
                # (full 3D attention is model-dependent; we use a best-effort 2D loop)
                D, H, W = v.shape
                pred_vol = np.zeros((D, H, W), dtype=np.float32)

                d_min, h_min, w_min = bbox_min
                d_max, h_max, w_max = bbox_max

                # Build bounding box prompt tensor for each slice in depth
                for d in range(int(d_min), int(d_max) + 1):
                    slice_img = v[d]  # [H, W]
                    # MedSAM standard 2D prompt format: [x1, y1, x2, y2] in pixel coords
                    box_prompt = np.array([w_min, h_min, w_max, h_max], dtype=np.float32)

                    img_t = torch.from_numpy(
                        np.stack([slice_img] * 3, axis=0)[None]  # [1, 3, H, W]
                    ).to(self.device)
                    if self.fp16:
                        img_t = img_t.half()

                    box_t = torch.from_numpy(box_prompt[None, None]).to(self.device)  # [1, 1, 4]

                    # SAM-style forward pass
                    img_embed = self._model.image_encoder(img_t)
                    sparse_emb, dense_emb = self._model.prompt_encoder(
                        points=None, boxes=box_t, masks=None
                    )
                    low_res_masks, _ = self._model.mask_decoder(
                        image_embeddings=img_embed,
                        image_pe=self._model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    # Upsample to [H, W]
                    mask_prob = torch.nn.functional.interpolate(
                        low_res_masks.float(), size=(H, W), mode="bilinear", align_corners=False
                    )
                    pred_vol[d] = (torch.sigmoid(mask_prob[0, 0]) > 0.5).float().cpu().numpy()

                return pred_vol.astype(np.float32)

        except Exception as e:
            logger.warning(
                "MedSAMAdapter.infer_volume: inference failed (%s). Returning zero mask.", e
            )
            return np.zeros(volume_np.shape, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # QC Cross-Check Helper
    # ──────────────────────────────────────────────────────────────────────

    def get_dice_vs(
        self,
        hassl_mask: np.ndarray,
        volume_np: np.ndarray,
        bbox_min: Optional[np.ndarray] = None,
        bbox_max: Optional[np.ndarray] = None,
    ) -> float:
        """Cross-validate HASSL prediction against MedSAM reference.

        Args:
            hassl_mask: HASSL binary mask [D, H, W].
            volume_np: Input volume [D, H, W] for MedSAM inference.
            bbox_min: Optional bounding box minimum corner.
            bbox_max: Optional bounding box maximum corner.

        Returns:
            Dice coefficient between HASSL and MedSAM masks in [0, 1].
        """
        medsam_mask = self.infer_volume(volume_np, bbox_min, bbox_max)
        return _dice(hassl_mask, medsam_mask)

    @staticmethod
    def compute_sha256(volume_id: str, mask: np.ndarray) -> str:
        """Compute SHA-256 hash of (volume_id, mask bytes) for auditability."""
        h = hashlib.sha256()
        h.update(volume_id.encode("utf-8"))
        h.update(mask.tobytes())
        return h.hexdigest()
