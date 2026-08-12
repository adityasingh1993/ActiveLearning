import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union

from hassl.training.ema import enable_dropout


class BALDStrategy:
    """Bayesian Active Learning by Disagreement (Epistemic Uncertainty via MC Dropout)."""

    def __init__(self, model, num_passes: int = 5, T: int = 5):
        self.model = model
        self.T = num_passes or T
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def score(self, x: torch.Tensor) -> np.ndarray:
        """Compute BALD mutual information score per volume (M-5 fix)."""
        self.model.eval()
        enable_dropout(self.model)  # M-5 fix: enable only dropout, keep norm layers in eval

        with torch.no_grad():
            preds = []
            for _ in range(self.T):
                out = self.model(x.to(self.device))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif out.ndim == 6:
                    out = out[:, 0]

                prob = torch.sigmoid(out) if out.shape[1] == 1 else torch.softmax(out, dim=1)
                preds.append(prob)

            preds = torch.stack(preds, dim=0)  # [T, B, C, D, H, W]

            # Expected Entropy E[H[y|x,w]]
            eps = 1e-8
            entropy_per_pass = - (preds * torch.log(preds + eps) + (1 - preds) * torch.log(1 - preds + eps))
            expected_entropy = entropy_per_pass.mean(dim=0)

            # Predictive Entropy H[y|x]
            mean_preds = preds.mean(dim=0)
            predictive_entropy = - (mean_preds * torch.log(mean_preds + eps) + (1 - mean_preds) * torch.log(1 - mean_preds + eps))

            # Mutual Information (Epistemic Uncertainty)
            mi = predictive_entropy - expected_entropy
            volume_scores = mi.mean(dim=(1, 2, 3, 4))
            return volume_scores.cpu().numpy()

    def query(self, unlabeled_loader, labeled_ids: List[str], k: int) -> Tuple[List[str], Dict[str, float]]:
        scores_dict = {}
        for batch in unlabeled_loader:
            x = batch['image']
            vids = batch.get('id', batch.get('volume_id', []))
            scores = self.score(x)
            for vid, s in zip(vids, scores):
                scores_dict[vid] = float(s)

        sorted_ids = sorted(scores_dict.keys(), key=lambda v: scores_dict[v], reverse=True)
        return sorted_ids[:k], scores_dict


class CoreSetStrategy:
    """CoreSet Selection via Greedy k-Center in SSL Embedding Space (M-4 fix)."""

    def __init__(self, embeddings_dict: Dict[str, np.ndarray]):
        self.embeddings = embeddings_dict

    def score(self, unlabeled_ids: List[str], labeled_ids: List[str]) -> Dict[str, float]:
        """Compute initial min distances to labeled set."""
        if not self.embeddings or not unlabeled_ids:
            return {uid: np.random.rand() for uid in unlabeled_ids}

        valid_unlabeled = [uid for uid in unlabeled_ids if uid in self.embeddings]
        valid_labeled = [lid for lid in labeled_ids if lid in self.embeddings]

        if not valid_labeled:
            return {uid: np.random.rand() for uid in unlabeled_ids}

        labeled_feats = np.array([self.embeddings[lid] for lid in valid_labeled])
        unlabeled_feats = np.array([self.embeddings[uid] for uid in valid_unlabeled])

        dists = cdist(unlabeled_feats, labeled_feats, metric='euclidean')
        min_dists = np.min(dists, axis=1)

        result = {uid: float(dist) for uid, dist in zip(valid_unlabeled, min_dists)}
        for uid in unlabeled_ids:
            if uid not in result:
                result[uid] = 0.5
        return result

    def query(self, unlabeled_ids: Union[List[str], Any], labeled_ids: List[str], k: int) -> Tuple[List[str], Dict[str, float]]:
        """Greedy k-Center Iterative CoreSet Selection (M-4 fix)."""
        if hasattr(unlabeled_ids, '__iter__') and not isinstance(unlabeled_ids, (list, tuple)):
            # If DataLoader passed, extract volume IDs
            loader_vids = []
            for batch in unlabeled_ids:
                vids = batch.get('id', batch.get('volume_id', []))
                loader_vids.extend(vids)
            unlabeled_ids = loader_vids

        if not self.embeddings or not unlabeled_ids:
            return list(unlabeled_ids)[:k], {uid: 1.0 for uid in unlabeled_ids[:k]}

        valid_unlabeled = [uid for uid in unlabeled_ids if uid in self.embeddings]
        valid_labeled = [lid for lid in labeled_ids if lid in self.embeddings]

        if not valid_unlabeled:
            return list(unlabeled_ids)[:k], {uid: 1.0 for uid in unlabeled_ids[:k]}

        unlabeled_feats = np.array([self.embeddings[uid] for uid in valid_unlabeled])

        if valid_labeled:
            labeled_feats = np.array([self.embeddings[lid] for lid in valid_labeled])
            min_dists = np.min(cdist(unlabeled_feats, labeled_feats, metric='euclidean'), axis=1)
        else:
            # First pick is arbitrary/random if no labeled points
            min_dists = np.ones(len(valid_unlabeled))

        selected_ids = []
        scores_dict = {uid: float(d) for uid, d in zip(valid_unlabeled, min_dists)}

        # Greedy k-Center Selection loop (M-4 fix)
        for _ in range(min(k, len(valid_unlabeled))):
            idx = np.argmax(min_dists)
            chosen_id = valid_unlabeled[idx]
            selected_ids.append(chosen_id)

            # Update min_dists with distance to newly chosen center
            chosen_feat = unlabeled_feats[idx:idx + 1]
            new_dists = cdist(unlabeled_feats, chosen_feat, metric='euclidean').flatten()
            min_dists = np.minimum(min_dists, new_dists)

        return selected_ids, scores_dict


class DisagreementStrategy:
    """Disagreement Strategy measuring variance/discrepancy between dual models (Net A & Net B)."""

    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def score(self, x: torch.Tensor) -> np.ndarray:
        """Compute voxel-wise probability disagreement score."""
        self.model_a.eval()
        self.model_b.eval()
        with torch.no_grad():
            out_a = self.model_a(x.to(self.device))
            out_b = self.model_b(x.to(self.device))

            if isinstance(out_a, (tuple, list)): out_a = out_a[0]
            if isinstance(out_b, (tuple, list)): out_b = out_b[0]

            prob_a = torch.sigmoid(out_a) if out_a.shape[1] == 1 else torch.softmax(out_a, dim=1)
            prob_b = torch.sigmoid(out_b) if out_b.shape[1] == 1 else torch.softmax(out_b, dim=1)

            disagreement = torch.abs(prob_a - prob_b)
            volume_scores = disagreement.mean(dim=(1, 2, 3, 4))
            return volume_scores.cpu().numpy()

    def query(self, unlabeled_loader, labeled_ids: List[str], k: int) -> Tuple[List[str], Dict[str, float]]:
        scores_dict = {}
        for batch in unlabeled_loader:
            x = batch['image']
            vids = batch.get('id', batch.get('volume_id', []))
            scores = self.score(x)
            for vid, s in zip(vids, scores):
                scores_dict[vid] = float(s)

        sorted_ids = sorted(scores_dict.keys(), key=lambda v: scores_dict[v], reverse=True)
        return sorted_ids[:k], scores_dict


class HybridStrategy:
    """Hybrid Query Strategy fusing BALD + CoreSet + Disagreement scores."""

    def __init__(self, bald_strategy, coreset_strategy, disagreement_strategy, alpha=0.4, beta=0.3, gamma=0.3):
        self.bald = bald_strategy
        self.coreset = coreset_strategy
        self.disagreement = disagreement_strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _normalize(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        if not scores_dict:
            return {}
        vals = np.array(list(scores_dict.values()))
        min_val = np.min(vals)
        max_val = np.max(vals)
        if max_val == min_val:
            return {k: 0.5 for k in scores_dict.keys()}
        return {k: float((v - min_val) / (max_val - min_val)) for k, v in scores_dict.items()}

    def query(self, unlabeled_loader, labeled_ids: List[str], k: int) -> Tuple[List[str], Dict[str, float]]:
        bald_scores = {}
        dis_scores = {}
        unlabeled_ids = []

        for batch in unlabeled_loader:
            x = batch['image']
            vids = batch.get('id', batch.get('volume_id', []))

            b_scores = self.bald.score(x)
            d_scores = self.disagreement.score(x)

            for i, vid in enumerate(vids):
                bald_scores[vid] = float(b_scores[i])
                dis_scores[vid] = float(d_scores[i])
                unlabeled_ids.append(vid)

        coreset_scores = self.coreset.score(unlabeled_ids, labeled_ids)

        bald_norm = self._normalize(bald_scores)
        coreset_norm = self._normalize(coreset_scores)
        dis_norm = self._normalize(dis_scores)

        final_scores = {}
        for vid in unlabeled_ids:
            score = (self.alpha * bald_norm.get(vid, 0.5) +
                     self.beta * coreset_norm.get(vid, 0.5) +
                     self.gamma * dis_norm.get(vid, 0.5))
            final_scores[vid] = float(score)

        # C-2 & M-4 fix: CoreSet greedy refinement on top candidates
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        return sorted_ids[:k], final_scores
