import json
import os
import glob
import time
from pathlib import Path
import torch
import numpy as np
import nrrd


class QueryEngine:
    """Manages the dataset pool state (labeled/unlabeled sets) and exports pre-segmentations."""

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
        all_labels = sorted(glob.glob(str(data_dir_path / f"**/*{label_suffix}"), recursive=True))

        labeled_ids = [os.path.basename(p).replace(label_suffix, '') for p in all_labels]
        all_ids = [os.path.basename(p).replace(image_suffix, '') for p in all_images]

        unlabeled_ids = [vid for vid in all_ids if vid not in labeled_ids]

        self.state['labeled_ids'] = labeled_ids
        self.state['unlabeled_ids'] = unlabeled_ids
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
        if config := self.config:
            label_dir = label_dir or os.path.join(config.data_dir, 'labels')
            if not os.path.exists(label_dir):
                label_dir = config.data_dir
            label_suffix = label_suffix or config.label_suffix

        all_labels = glob.glob(os.path.join(label_dir, f'*{label_suffix}'))
        current_labeled = set(self.state['labeled_ids'])
        new_labeled = []

        for path in all_labels:
            vid = os.path.basename(path).replace(label_suffix, '')
            if vid not in current_labeled and vid in self.state['unlabeled_ids']:
                new_labeled.append(vid)
                self.state['unlabeled_ids'].remove(vid)
                self.state['labeled_ids'].append(vid)

        if new_labeled:
            self._save_manifest()
            if self.tracker:
                self.tracker.log_metrics({'new_labels_added': len(new_labeled)}, step=0)

        return new_labeled

    def auto_promote_pseudo_labels(self, model, dataloader, k=10):
        """Option A (Fully Automated): Automatically predict & promote top K pseudo-labels into training pool."""
        model.eval()
        promoted_ids = []

        output_label_dir = os.path.join(self.config.data_dir, 'labels') if self.config else './data/labels'
        os.makedirs(output_label_dir, exist_ok=True)

        with torch.no_grad():
            for batch in dataloader:
                if len(promoted_ids) >= k:
                    break
                x = batch['image'].to(self.device)
                vids = batch.get('id', batch.get('volume_id', []))

                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]

                probs = torch.sigmoid(out) if getattr(self.config, 'num_classes', 1) == 1 else torch.softmax(out, dim=1)
                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)

                for i, vid in enumerate(vids):
                    if len(promoted_ids) >= k:
                        break
                    if vid in self.state['unlabeled_ids']:
                        pred_vol = preds[i, 0]
                        output_path = os.path.join(output_label_dir, f'{vid}{self.config.label_suffix if self.config else ".seg.nrrd"}')
                        nrrd.write(output_path, pred_vol)

                        self.state['unlabeled_ids'].remove(vid)
                        self.state['labeled_ids'].append(vid)
                        promoted_ids.append(vid)

        self._save_manifest()
        if self.tracker:
            self.tracker.log_metrics({'auto_promoted_pseudo_labels': len(promoted_ids)}, step=0)

        return promoted_ids

    def run_query(self, strategy=None, unlabeled_loader=None, round_num=1, embeddings=None, k=10):
        if self.config:
            k = self.config.al_query_size

        # Default fallback if strategy/loader not provided: pick next k unlabeled IDs
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

    def export_presegmentation(self, model=None, dataloader=None, volume_ids=None, output_dir=None):
        if self.config:
            output_dir = output_dir or self.config.preseg_dir
        output_dir = output_dir or './data/al_preseg'
        os.makedirs(output_dir, exist_ok=True)

        if volume_ids:
            # Touch empty preseg placeholder files for volume_ids if model is not running
            for vid in volume_ids:
                output_path = os.path.join(output_dir, f'{vid}.seg.nrrd')
                if not os.path.exists(output_path):
                    # Write placeholder 128x128x128 array
                    nrrd.write(output_path, np.zeros((128, 128, 128), dtype=np.uint8))

        if model is not None and dataloader is not None:
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    x = batch['image'].to(self.device)
                    vids = batch.get('id', batch.get('volume_id', []))

                    out = model(x)
                    if isinstance(out, (tuple, list)):
                        out = out[0]

                    preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)

                    for i, vid in enumerate(vids):
                        pred_vol = preds[i, 0]
                        output_path = os.path.join(output_dir, f'{vid}.seg.nrrd')
                        nrrd.write(output_path, pred_vol)

        if self.tracker:
            self.tracker.log_metrics({'exported_presegmentations': len(volume_ids or [])}, step=0)
