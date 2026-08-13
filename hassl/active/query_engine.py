import json
import os
import glob
import time
from pathlib import Path
import torch
import numpy as np

from hassl.data.nrrd_utils import write_mask_with_spatial_geometry


class QueryEngine:
    """Manages pool manifests, provenance tracking, pseudo-label ranking, and pre-segmentation exports."""

    def __init__(self, config=None, manifest_path=None, tracker=None):
        self.config = config
        self.tracker = tracker
        self.device = torch.device(
            getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        if manifest_path is None and config is not None:
            manifest_path = os.path.join(config.log_dir, "pool_manifest.json")
        elif manifest_path is None:
            manifest_path = "./experiments/logs/pool_manifest.json"

        self.manifest_path = manifest_path

        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'labeled_ids': [],
                'unlabeled_ids': [],
                'pseudo_ids': [],
                'provenance': {},  # {vol_id: "human" | "pseudo_unreviewed" | "pseudo_approved"}
                'rounds': []
            }

    def _save_manifest(self):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def initialize_pool(self, data_dir=None, image_suffix=None, label_suffix=None):
        if config := self.config:
            data_dir = data_dir or config.data_dir
            image_suffix = image_suffix or config.image_suffix
            label_suffix = label_suffix or config.label_suffix

        data_dir_path = Path(data_dir)
        all_images = sorted(glob.glob(str(data_dir_path / f"**/*{image_suffix}"), recursive=True))
        all_labels = sorted(glob.glob(str(data_dir_path / "labels" / f"*{label_suffix}")))

        labeled_ids = [os.path.basename(p).replace(label_suffix, '') for p in all_labels]
        all_ids = [os.path.basename(p).replace(image_suffix, '') for p in all_images]

        unlabeled_ids = [vid for vid in all_ids if vid not in labeled_ids]

        self.state['labeled_ids'] = labeled_ids
        self.state['unlabeled_ids'] = unlabeled_ids

        # Set human gold standard provenance (C-1 fix)
        for vid in labeled_ids:
            self.state['provenance'][vid] = "human"

        self._save_manifest()

        if self.tracker:
            self.tracker.log_metrics({
                'initial_labeled': len(labeled_ids),
                'initial_unlabeled': len(unlabeled_ids)
            }, step=0)

        return labeled_ids, unlabeled_ids

    def get_labeled_ids(self):
        return self.state.get('labeled_ids', [])

    def get_unlabeled_ids(self):
        return self.state.get('unlabeled_ids', [])

    def detect_new_labels(self, label_dir=None, label_suffix=None):
        """Detect newly added human labels in data/labels/ and update provenance (C-1 fix)."""
        if config := self.config:
            label_dir = label_dir or os.path.join(config.data_dir, 'labels')
            label_suffix = label_suffix or config.label_suffix

        all_labels = glob.glob(os.path.join(label_dir, f'*{label_suffix}'))
        current_labeled = set(self.state['labeled_ids'])
        new_labeled = []

        for path in all_labels:
            vid = os.path.basename(path).replace(label_suffix, '')
            if vid not in current_labeled:
                new_labeled.append(vid)
                if vid in self.state['unlabeled_ids']:
                    self.state['unlabeled_ids'].remove(vid)
                if vid in self.state.get('pseudo_ids', []):
                    self.state['pseudo_ids'].remove(vid)
                self.state['labeled_ids'].append(vid)
                self.state['provenance'][vid] = "human"

        if new_labeled:
            self._save_manifest()
            if self.tracker:
                self.tracker.log_metrics({'new_labels_added': len(new_labeled)}, step=0)

        return new_labeled

    def _invert_prediction(self, pred_tensor: torch.Tensor, image_tensor: torch.Tensor, batch_data: dict, index: int) -> np.ndarray:
        """Invert prediction back to original volume spatial orientation and shape (Orientationd + Resized inversion)."""
        if self.config:
            try:
                from hassl.data.data_engine import get_base_transforms
                from monai.transforms import Invertd

                val_transform = get_base_transforms(self.config, keys=["image"], is_training=False)
                inv_transform = Invertd(
                    keys=["pred"],
                    transform=val_transform,
                    orig_keys=["image"],
                    meta_keys=["pred_meta_dict"],
                    orig_meta_keys=["image_meta_dict"],
                    nearest_interp=True,
                    to_tensor=True,
                )

                sample = {
                    "image": image_tensor[index].detach().cpu(),
                    "pred": pred_tensor[index:index + 1].detach().cpu(),
                }
                if 'image_meta_dict' in batch_data:
                    sample["image_meta_dict"] = {
                        k: (v[index] if isinstance(v, (list, tuple)) else v)
                        for k, v in batch_data['image_meta_dict'].items()
                    }

                inv_out = inv_transform(sample)
                inv_pred = inv_out["pred"]
                if inv_pred.ndim == 4:
                    inv_pred = inv_pred[0]
                return (inv_pred > 0.5).numpy().astype(np.uint8)
            except Exception:
                pass

        return (pred_tensor[index, 0] > 0.5).cpu().numpy().astype(np.uint8)

    def auto_promote_pseudo_labels(
        self,
        model,
        dataloader,
        k: int = 10,
        confidence_threshold: float = 0.85,
        mc_passes: int = 5,
        mc_var_threshold: float = 0.05,
        tta_passes: int = 8,
        tta_var_threshold: float = 0.02,
        tta_flip: bool = True,
        tta_intensity_std: float = 0.02,
    ):
        """Rank and promote top-K high-quality pseudo-labels using three quality gates.

        Gate 1 — Foreground Confidence (existing):
            Mean probability of foreground-predicted voxels must be >= confidence_threshold.

        Gate 2 — Epistemic (MC Dropout) Variance:
            Voxel-wise prediction variance across mc_passes MC Dropout forward passes
            must be < mc_var_threshold. High variance = model is uncertain.

        Gate 3 — Aleatoric (TTA) Variance:
            Voxel-wise prediction variance across tta_passes augmented forward passes
            must be < tta_var_threshold. High variance = data is inherently ambiguous.

        Only volumes passing ALL THREE gates are promoted to data/pseudo_unreviewed/.
        All thresholds are configurable and can be set via HASSLConfig fields.
        """
        if model is None or dataloader is None:
            raise ValueError("model and dataloader are required for pseudo-label promotion.")

        # Pull thresholds from config if available (config values take priority)
        if self.config is not None:
            confidence_threshold = getattr(self.config, 'pseudo_confidence_threshold', confidence_threshold)
            mc_passes           = getattr(self.config, 'pseudo_mc_passes', mc_passes)
            mc_var_threshold    = getattr(self.config, 'pseudo_mc_var_threshold', mc_var_threshold)
            tta_passes          = getattr(self.config, 'pseudo_tta_passes', tta_passes)
            tta_var_threshold   = getattr(self.config, 'pseudo_tta_var_threshold', tta_var_threshold)
            tta_flip            = getattr(self.config, 'pseudo_tta_flip', tta_flip)
            tta_intensity_std   = getattr(self.config, 'pseudo_tta_intensity_std', tta_intensity_std)

        from hassl.active.query_strategies import BALDStrategy, TTAUncertaintyScorer
        mc_scorer  = BALDStrategy(model, num_passes=mc_passes)
        tta_scorer = TTAUncertaintyScorer(
            model,
            num_passes=tta_passes,
            flip=tta_flip,
            intensity_std=tta_intensity_std,
        )

        model.eval()
        candidates = []

        output_pseudo_dir = (
            os.path.join(self.config.data_dir, 'pseudo_unreviewed')
            if self.config else './data/pseudo_unreviewed'
        )
        os.makedirs(output_pseudo_dir, exist_ok=True)

        with torch.no_grad():
            for batch in dataloader:
                x = batch['image'].to(self.device)
                vids = batch.get('id', batch.get('volume_id', []))
                image_paths = batch.get('image_meta_dict', {}).get('filename_or_obj', [None] * len(vids))

                # ── Gate 1: single-pass foreground confidence ──────────────
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                probs = (
                    torch.sigmoid(out)
                    if getattr(self.config, 'num_classes', 1) == 1
                    else torch.softmax(out, dim=1)
                )

                # ── Gate 2: MC Dropout epistemic variance ──────────────────
                mc_vars = mc_scorer.score(x)   # [B]

                # ── Gate 3: TTA aleatoric variance ─────────────────────────
                tta_vars = tta_scorer.score(x)  # [B]

                for i, vid in enumerate(vids):
                    if vid not in self.state['unlabeled_ids']:
                        continue

                    p_vol = probs[i, 0]
                    fg_mask = p_vol > 0.5
                    fg_count = fg_mask.sum().item()

                    # Gate 1 — Foreground confidence
                    if fg_count > 10:
                        conf = float((p_vol[fg_mask] - 0.5).mean().item() * 2.0)
                    else:
                        conf = 0.0  # Empty/background prediction

                    mc_var  = float(mc_vars[i])
                    tta_var = float(tta_vars[i])

                    # All three gates must pass
                    if (
                        conf >= confidence_threshold
                        and mc_var < mc_var_threshold
                        and tta_var < tta_var_threshold
                    ):
                        inverted_pred = self._invert_prediction(probs, x, batch, i)
                        candidates.append({
                            'id': vid,
                            'confidence': conf,
                            'mc_var': mc_var,
                            'tta_var': tta_var,
                            'pred_vol': inverted_pred,
                            'ref_path': image_paths[i] if i < len(image_paths) else None,
                        })

        # Deduplicate candidates by volume ID, keeping the highest confidence entry per volume
        best_candidates: dict = {}
        for cand in candidates:
            vid = cand['id']
            if vid not in best_candidates or cand['confidence'] > best_candidates[vid]['confidence']:
                best_candidates[vid] = cand

        # Sort by confidence descending and take top K
        sorted_candidates = sorted(best_candidates.values(), key=lambda c: c['confidence'], reverse=True)
        top_k_candidates = sorted_candidates[:k]

        promoted_ids = []
        for cand in top_k_candidates:
            vid = cand['id']
            pred_vol = cand['pred_vol']
            ref_path = cand['ref_path']
            label_sfx = self.config.label_suffix if self.config else '.seg.nrrd'
            output_path = os.path.join(output_pseudo_dir, f'{vid}{label_sfx}')

            write_mask_with_spatial_geometry(output_path, pred_vol, reference_image_path=ref_path)

            if vid in self.state['unlabeled_ids']:
                self.state['unlabeled_ids'].remove(vid)
            if 'pseudo_ids' not in self.state:
                self.state['pseudo_ids'] = []
            if vid not in self.state['pseudo_ids']:
                self.state['pseudo_ids'].append(vid)
            self.state['provenance'][vid] = "pseudo_unreviewed"
            promoted_ids.append(vid)

        self._save_manifest()

        if self.tracker and top_k_candidates:
            self.tracker.log_metrics({
                'auto_promoted_pseudo_labels': len(promoted_ids),
                'mean_pseudo_confidence':
                    float(np.mean([c['confidence'] for c in top_k_candidates])),
                'mean_pseudo_mc_var':
                    float(np.mean([c['mc_var'] for c in top_k_candidates])),
                'mean_pseudo_tta_var':
                    float(np.mean([c['tta_var'] for c in top_k_candidates])),
            }, step=0)
        elif self.tracker:
            self.tracker.log_metrics({'auto_promoted_pseudo_labels': 0}, step=0)

        return promoted_ids

    def run_query(self, strategy=None, unlabeled_loader=None, round_num=1, embeddings=None, k=10):
        """Run active learning query with strategy."""
        if self.config:
            k = self.config.al_query_size

        if strategy is None or unlabeled_loader is None:
            top_k_ids = self.state['unlabeled_ids'][:k]
            scores = {vid: 1.0 for vid in top_k_ids}
        else:
            top_k_ids, scores = strategy.query(unlabeled_loader, self.state['labeled_ids'], k)

        round_info = {
            'round_num': round_num,
            'queried_ids': top_k_ids,
            'scores': {vid: float(scores.get(vid, 0.0)) for vid in top_k_ids},
            'timestamp': time.time()
        }

        self.state['rounds'].append(round_info)
        self._save_manifest()

        if self.tracker:
            self.tracker.log_metrics({
                'al_round': round_num,
                'queried_samples': len(top_k_ids)
            }, step=round_num)

        return top_k_ids, [scores.get(vid, 0.0) for vid in top_k_ids]

    def export_presegmentation(self, model, dataloader, volume_ids=None, output_dir=None):
        """Export AI pre-segmentation masks for human review with spatial geometry (C-3 & W-1 fix)."""
        if model is None or dataloader is None:
            raise RuntimeError("Cannot export pre-segmentation: trained model and dataloader are required (C-3 fix).")

        if self.config:
            output_dir = output_dir or self.config.preseg_dir
        output_dir = output_dir or './data/al_preseg'
        os.makedirs(output_dir, exist_ok=True)

        model.eval()
        exported_count = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch['image'].to(self.device)
                vids = batch.get('id', batch.get('volume_id', []))
                image_paths = batch.get('image_meta_dict', {}).get('filename_or_obj', [None] * len(vids))

                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                probs = torch.sigmoid(out) if getattr(self.config, 'num_classes', 1) == 1 else torch.softmax(out, dim=1)

                for i, vid in enumerate(vids):
                    if volume_ids is None or vid in volume_ids:
                        pred_vol = self._invert_prediction(probs, x, batch, i)
                        ref_path = image_paths[i] if i < len(image_paths) else None
                        output_path = os.path.join(output_dir, f'{vid}.seg.nrrd')

                        # W-1 fix: Write with SimpleITK to preserve spatial origin, spacing, and direction matrix
                        write_mask_with_spatial_geometry(output_path, pred_vol, reference_image_path=ref_path)
                        exported_count += 1

        if self.tracker:
            self.tracker.log_metrics({'exported_presegmentations': exported_count}, step=0)
