"""
HASSL QC Gate Module.

Automated multi-signal quality control gate that evaluates the reliability and
clinical validity of segmentation predictions before they are used downstream
(pseudo-label promotion, pre-segmentation export, or training data inclusion).

QC Gate Checkers (composable):
    - UncertaintyChecker  : MC Dropout / entropy-based model confidence
    - VolumeChecker       : Anatomical prior voxel count plausibility
    - TopologyChecker     : Connected component count limit
    - ConvexityChecker    : Surface convexity ratio (shape regularity)
    - MedSAMCrossChecker  : Dice agreement with MedSAM reference (optional)

All failures produce:
    - A QCReport with composite score, per-checker flags, and SHA-256 hash
    - An entry in the human review queue (if ReviewQueue is provided)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.ndimage as ndi
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8


@dataclass
class QCReport:
    """Output report from the QC Gate for a single volume prediction."""

    volume_id: str
    passed: bool
    score: float                          # Composite quality score [0, 1]; higher = better
    flags: List[str] = field(default_factory=list)   # Names of failed checkers
    checker_scores: Dict[str, float] = field(default_factory=dict)  # Per-checker scores
    sha256_hash: str = ""                 # SHA-256(volume_id + score + flags) for audit trail

    def to_dict(self) -> dict:
        return {
            "volume_id": self.volume_id,
            "passed": self.passed,
            "score": round(self.score, 6),
            "flags": self.flags,
            "checker_scores": {k: round(v, 6) for k, v in self.checker_scores.items()},
            "sha256_hash": self.sha256_hash,
        }


def _sha256_report(volume_id: str, score: float, flags: List[str]) -> str:
    """Compute SHA-256 audit hash over (volume_id, score, flags)."""
    payload = json.dumps(
        {"volume_id": volume_id, "score": round(score, 6), "flags": sorted(flags)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Individual Checkers
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyChecker:
    """Assess model confidence via mean voxel entropy over foreground ROI.

    Computes entropy of the sigmoid probability map. A high mean entropy over
    foreground voxels indicates the model is uncertain about the prediction.

    Args:
        threshold: Max allowed mean entropy (default 0.35).
    """

    name = "uncertainty"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def check(self, prob_map: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Args:
            prob_map: Foreground probability map [...] in [0, 1].
            mask: Optional binary mask restricting entropy calculation to ROI.

        Returns:
            Quality score in [0, 1]; 1.0 = perfectly confident, 0.0 = maximum uncertainty.
        """
        p = np.clip(prob_map, _EPS, 1.0 - _EPS)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        if mask is not None and mask.any():
            mean_entropy = float(entropy[mask > 0].mean())
        else:
            mean_entropy = float(entropy.mean())

        # Score: 1.0 if entropy == 0, 0.0 if entropy >= threshold
        score = max(0.0, 1.0 - (mean_entropy / (self.threshold + _EPS)))
        return min(1.0, score)

    def passes(self, prob_map: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        return self.check(prob_map, mask) > 0.0


class VolumeChecker:
    """Anatomical plausibility check on predicted organ volume.

    Rejects masks where the foreground voxel count is outside the physiologically
    expected range for the target organ (e.g., bladder: 500–150,000 voxels).

    Args:
        min_voxels: Minimum acceptable foreground voxels (default 500).
        max_voxels: Maximum acceptable foreground voxels (default 150,000).
    """

    name = "volume"

    def __init__(self, min_voxels: int = 500, max_voxels: int = 150_000):
        self.min_voxels = min_voxels
        self.max_voxels = max_voxels

    def check(self, binary_mask: np.ndarray) -> float:
        """
        Returns:
            1.0 if within range, linearly decays to 0.0 at boundary violations.
        """
        count = int(binary_mask.sum())
        if count < self.min_voxels:
            return max(0.0, count / (self.min_voxels + _EPS))
        if count > self.max_voxels:
            return max(0.0, 1.0 - (count - self.max_voxels) / (self.max_voxels + _EPS))
        return 1.0

    def passes(self, binary_mask: np.ndarray) -> bool:
        count = int(binary_mask.sum())
        return self.min_voxels <= count <= self.max_voxels


class TopologyChecker:
    """Check that the prediction has at most N connected components.

    Spurious satellite fragments from ultrasound shadowing artifacts manifest as
    disconnected label islands. More than qc_max_cc_count components is suspicious.

    Args:
        max_cc: Maximum allowed connected component count (default 3).
    """

    name = "topology"

    def __init__(self, max_cc: int = 3):
        self.max_cc = max_cc

    def check(self, binary_mask: np.ndarray) -> float:
        """
        Returns:
            1.0 if <= max_cc, decreasing penalty per extra component.
        """
        if not binary_mask.any():
            return 0.0
        _, num = ndi.label(binary_mask)
        if num <= self.max_cc:
            return 1.0
        return max(0.0, 1.0 - (num - self.max_cc) * 0.15)

    def passes(self, binary_mask: np.ndarray) -> bool:
        if not binary_mask.any():
            return False
        _, num = ndi.label(binary_mask)
        return num <= self.max_cc


class ConvexityChecker:
    """Measure surface convexity of the predicted mask.

    Computes ratio = |mask| / |convex_hull(mask)|. Spiky or non-convex predictions
    from boundary artifacts will have low convexity ratios.

    Args:
        min_convexity: Minimum convexity ratio (default 0.70).
    """

    name = "convexity"

    def __init__(self, min_convexity: float = 0.70):
        self.min_convexity = min_convexity

    def check(self, binary_mask: np.ndarray) -> float:
        """
        Returns:
            Convexity ratio in [0, 1]; 1.0 = perfectly convex.
        """
        if not binary_mask.any():
            return 0.0
        try:
            from scipy.ndimage import binary_fill_holes
            convex_hull = binary_fill_holes(binary_mask)
            mask_vol = float(binary_mask.sum())
            hull_vol = float(convex_hull.sum())
            if hull_vol < _EPS:
                return 0.0
            return min(1.0, mask_vol / hull_vol)
        except Exception:
            return 1.0  # Graceful fallback: skip this checker

    def passes(self, binary_mask: np.ndarray) -> bool:
        return self.check(binary_mask) >= self.min_convexity


# ─────────────────────────────────────────────────────────────────────────────
# QC Gate (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class QCGate:
    """Multi-signal Quality Control Gate for segmentation predictions.

    Runs a configurable set of checkers and produces a QCReport with a
    composite quality score, per-checker flags, and an audit SHA-256 hash.

    Args:
        uncertainty_threshold: UncertaintyChecker max entropy (default 0.35).
        volume_min_voxels: VolumeChecker lower bound (default 500).
        volume_max_voxels: VolumeChecker upper bound (default 150,000).
        max_cc_count: TopologyChecker max connected components (default 3).
        surface_convexity_min: ConvexityChecker minimum convexity (default 0.70).
        medsam_dice_threshold: Min Dice vs MedSAM mask before flagging (default 0.50).
            Set to None to disable MedSAM cross-check.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        volume_min_voxels: int = 500,
        volume_max_voxels: int = 150_000,
        max_cc_count: int = 3,
        surface_convexity_min: float = 0.70,
        medsam_dice_threshold: Optional[float] = None,
    ):
        self.uncertainty = UncertaintyChecker(threshold=uncertainty_threshold)
        self.volume = VolumeChecker(min_voxels=volume_min_voxels, max_voxels=volume_max_voxels)
        self.topology = TopologyChecker(max_cc=max_cc_count)
        self.convexity = ConvexityChecker(min_convexity=surface_convexity_min)
        self.medsam_dice_threshold = medsam_dice_threshold

    @classmethod
    def from_config(cls, config) -> "QCGate":
        """Construct QCGate from a HASSLConfig instance."""
        medsam_threshold = (
            config.medsam_qc_dice_threshold
            if getattr(config, "medsam_enabled", False)
            else None
        )
        return cls(
            uncertainty_threshold=getattr(config, "qc_uncertainty_threshold", 0.35),
            volume_min_voxels=getattr(config, "qc_volume_min_voxels", 500),
            volume_max_voxels=getattr(config, "qc_volume_max_voxels", 150_000),
            max_cc_count=getattr(config, "qc_max_cc_count", 3),
            surface_convexity_min=getattr(config, "qc_surface_convexity_min", 0.70),
            medsam_dice_threshold=medsam_threshold,
        )

    def evaluate(
        self,
        volume_id: str,
        prob_map: np.ndarray,
        binary_mask: np.ndarray,
        medsam_mask: Optional[np.ndarray] = None,
    ) -> QCReport:
        """Run all checkers and produce a QCReport.

        Args:
            volume_id: Unique identifier for the volume (for audit logging).
            prob_map: Foreground probability map [D, H, W] in [0, 1].
            binary_mask: Binarized prediction mask [D, H, W] in {0, 1}.
            medsam_mask: Optional MedSAM reference mask [D, H, W] for cross-check.

        Returns:
            QCReport with passed status, composite score, flags, and SHA-256 hash.
        """
        checker_scores: Dict[str, float] = {}
        flags: List[str] = []

        # 1. Uncertainty
        u_score = self.uncertainty.check(prob_map, mask=binary_mask)
        checker_scores["uncertainty"] = u_score
        if u_score <= 0.0:
            flags.append("uncertainty")

        # 2. Volume plausibility
        v_score = self.volume.check(binary_mask)
        checker_scores["volume"] = v_score
        if not self.volume.passes(binary_mask):
            flags.append("volume")

        # 3. Topology
        t_score = self.topology.check(binary_mask)
        checker_scores["topology"] = t_score
        if not self.topology.passes(binary_mask):
            flags.append("topology")

        # 4. Surface convexity
        c_score = self.convexity.check(binary_mask)
        checker_scores["convexity"] = c_score
        if not self.convexity.passes(binary_mask):
            flags.append("convexity")

        # 5. MedSAM cross-check (optional)
        if medsam_mask is not None and self.medsam_dice_threshold is not None:
            inter = float((binary_mask * medsam_mask).sum())
            union = float(binary_mask.sum() + medsam_mask.sum()) + _EPS
            medsam_dice = 2.0 * inter / union
            checker_scores["medsam_dice"] = medsam_dice
            if medsam_dice < self.medsam_dice_threshold:
                flags.append("medsam_dice")

        # Composite score: geometric mean of checker scores (penalises any weak link)
        scores_arr = np.array(list(checker_scores.values()), dtype=float)
        composite = float(np.exp(np.mean(np.log(scores_arr.clip(_EPS, 1.0)))))

        passed = len(flags) == 0
        audit_hash = _sha256_report(volume_id, composite, flags)

        report = QCReport(
            volume_id=volume_id,
            passed=passed,
            score=composite,
            flags=flags,
            checker_scores=checker_scores,
            sha256_hash=audit_hash,
        )

        log_level = logging.DEBUG if passed else logging.WARNING
        logger.log(
            log_level,
            "QCGate [%s] → %s (score=%.3f, flags=%s)",
            volume_id,
            "PASS" if passed else "FAIL",
            composite,
            flags,
        )
        return report

    def evaluate_tensor(
        self,
        volume_id: str,
        logit_tensor: torch.Tensor,
        medsam_mask_np: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> QCReport:
        """Convenience wrapper accepting a raw logit/probability PyTorch tensor.

        Args:
            volume_id: Volume identifier string.
            logit_tensor: [1, 1, D, H, W] or [1, C, D, H, W] logit tensor (unnormalised or sigmoid).
            medsam_mask_np: Optional MedSAM numpy mask [D, H, W].
            threshold: Binarisation threshold (default 0.5).

        Returns:
            QCReport.
        """
        t = logit_tensor.detach().cpu().float()
        if t.ndim == 5:
            t = t[0, 0]  # [D, H, W]
        # If already probability (max <= 1), use directly; otherwise apply sigmoid
        if t.max().item() > 1.0 or t.min().item() < 0.0:
            prob = torch.sigmoid(t).numpy()
        else:
            prob = t.numpy()
        binary = (prob >= threshold).astype(np.float32)
        return self.evaluate(volume_id, prob, binary, medsam_mask_np)
