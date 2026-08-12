import json
import os
import glob
import time
from pathlib import Path
import torch
import numpy as np
import nrrd


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
                self.state['provenance'][vid] = "human"  # Gold standard provenance

        if new_labeled:
            self._save_manifest()
            if self.tracker:
                self.tracker.log_metrics({'new_labels_added': len(new_labeled)}, step=0)

        return new_labeled

    def auto_promote_pseudo_labels(self, model, dataloader, k=10, confidence_threshold=0.90):
        """Rank, filter, and promote top K high-confidence pseudo-labels to data/pseudo/ (C-1 & C-2 fix)."""
        if model is None or dataloader is None:
            raise ValueError("model and dataloader are required for pseudo-label promotion.")

        model.eval()
        candidates = []

        # C-1 fix: Write pseudo-labels to data/pseudo/ NEVER directly to gold data/labels/
        output_pseudo_dir = os.path.join(self.config.data_dir, 'pseudo') if self.config else './data/pseudo'
        os.makedirs(output_pseudo_dir, exist_ok=True)

        with torch.no_grad():
            for batch in dataloader:
                x = batch['image'].to(self.device)
                vids = batch.get('id', batch.get('volume_id', []))

                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                probs = torch.sigmoid(out) if getattr(self.config, 'num_classes', 1) == 1 else torch.softmax(out, dim=1)
                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)

                # Compute confidence per volume (distance of max probability from 0.5 margin)
                confidence_scores = (torch.abs(probs - 0.5) * 2.0).mean(dim=(1, 2, 3, 4)).cpu().numpy()

                for i, vid in enumerate(vids):
                    if vid in self.state['unlabeled_ids']:
                        conf = float(confidence_scores[i])
                        if conf >= confidence_threshold:
                            candidates.append({
                                'id': vid,
                                'confidence': conf,
                                'pred_vol': preds[i, 0],
                            })

        # C-2 fix: Sort candidates by confidence score descending
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        top_k_candidates = candidates[:k]

        promoted_ids = []
        for cand in top_k_candidates:
            vid = cand['id']
            pred_vol = cand['pred_vol']
            output_path = os.path.join(output_pseudo_dir, f'{vid}{self.config.label_suffix if self.config else ".seg.nrrd"}')
            nrrd.write(output_path, pred_vol)

            self.state['unlabeled_ids'].remove(vid)
            if 'pseudo_ids' not in self.state:
                self.state['pseudo_ids'] = []
            self.state['pseudo_ids'].append(vid)
            self.state['provenance'][vid] = "pseudo_unreviewed"  # Machine provenance
            promoted_ids.append(vid)

        self._save_manifest()
        if self.tracker:
            self.tracker.log_metrics({
                'auto_promoted_pseudo_labels': len(promoted_ids),
                'mean_pseudo_confidence': float(np.mean([c['confidence'] for c in top_k_candidates])) if top_k_candidates else 0.0,
            }, step=0)

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
        """Export AI pre-segmentation masks for human review (C-3 fix)."""
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

                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)

                for i, vid in enumerate(vids):
                    if volume_ids is None or vid in volume_ids:
                        pred_vol = preds[i, 0]
                        output_path = os.path.join(output_dir, f'{vid}.seg.nrrd')
                        nrrd.write(output_path, pred_vol)
                        exported_count += 1

        if self.tracker:
            self.tracker.log_metrics({'exported_presegmentations': exported_count}, step=0)
