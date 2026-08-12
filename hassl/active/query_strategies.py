import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

class BALDStrategy:
    def __init__(self, model, T=5):
        self.model = model
        self.T = T
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def score(self, x):
        self.model.train() # Enable dropout
        with torch.no_grad():
            preds = []
            for _ in range(self.T):
                out = self.model(x.to(self.device))
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                prob = torch.sigmoid(out) # Assuming binary segmentation
                preds.append(prob)
                
            preds = torch.stack(preds) # [T, B, C, D, H, W]
            
            # Expected Entropy E[H[y|x,w]]
            eps = 1e-8
            entropy_per_pass = - (preds * torch.log(preds + eps) + (1 - preds) * torch.log(1 - preds + eps))
            expected_entropy = entropy_per_pass.mean(dim=0)
            
            # Predictive Entropy H[y|x]
            mean_preds = preds.mean(dim=0)
            predictive_entropy = - (mean_preds * torch.log(mean_preds + eps) + (1 - mean_preds) * torch.log(1 - mean_preds + eps))
            
            # Mutual Information
            mi = predictive_entropy - expected_entropy
            
            # Aggregate to volume score via mean
            volume_scores = mi.mean(dim=(1, 2, 3, 4)) # Mean across voxels
            return volume_scores.cpu().numpy()

class CoreSetStrategy:
    def __init__(self, embeddings_dict):
        self.embeddings = embeddings_dict
        
    def score(self, unlabeled_ids, labeled_ids):
        if not labeled_ids:
            # If no labeled data, return random scores
            return {uid: np.random.rand() for uid in unlabeled_ids}
            
        labeled_feats = np.array([self.embeddings[lid] for lid in labeled_ids])
        unlabeled_feats = np.array([self.embeddings[uid] for uid in unlabeled_ids])
        
        # Calculate distances from all unlabeled to all labeled
        dists = cdist(unlabeled_feats, labeled_feats, metric='euclidean')
        
        # Minimum distance to any labeled sample
        min_dists = np.min(dists, axis=1)
        
        return {uid: dist for uid, dist in zip(unlabeled_ids, min_dists)}

class DisagreementStrategy:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def score(self, x):
        self.model_a.eval()
        self.model_b.eval()
        with torch.no_grad():
            out_a = self.model_a(x.to(self.device))
            out_b = self.model_b(x.to(self.device))
            
            if isinstance(out_a, tuple) or isinstance(out_a, list):
                out_a = out_a[0]
            if isinstance(out_b, tuple) or isinstance(out_b, list):
                out_b = out_b[0]
                
            prob_a = torch.sigmoid(out_a)
            prob_b = torch.sigmoid(out_b)
            
            disagreement = torch.abs(prob_a - prob_b)
            volume_scores = disagreement.mean(dim=(1, 2, 3, 4))
            return volume_scores.cpu().numpy()

class HybridStrategy:
    def __init__(self, bald_strategy, coreset_strategy, disagreement_strategy, alpha=0.4, beta=0.3, gamma=0.3):
        self.bald = bald_strategy
        self.coreset = coreset_strategy
        self.disagreement = disagreement_strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def _normalize(self, scores_dict):
        if not scores_dict:
            return {}
        vals = np.array(list(scores_dict.values()))
        min_val = np.min(vals)
        max_val = np.max(vals)
        if max_val == min_val:
            return {k: 0.5 for k in scores_dict.keys()}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

    def query(self, unlabeled_loader, labeled_ids, k):
        bald_scores = {}
        dis_scores = {}
        
        unlabeled_ids = []
        for batch in unlabeled_loader:
            x = batch['image']
            vids = batch['volume_id']
            
            b_scores = self.bald.score(x)
            d_scores = self.disagreement.score(x)
            
            for i, vid in enumerate(vids):
                bald_scores[vid] = b_scores[i]
                dis_scores[vid] = d_scores[i]
                unlabeled_ids.append(vid)
                
        coreset_scores = self.coreset.score(unlabeled_ids, labeled_ids)
        
        bald_norm = self._normalize(bald_scores)
        coreset_norm = self._normalize(coreset_scores)
        dis_norm = self._normalize(dis_scores)
        
        final_scores = {}
        for vid in unlabeled_ids:
            score = (self.alpha * bald_norm[vid] + 
                     self.beta * coreset_norm[vid] + 
                     self.gamma * dis_norm[vid])
            final_scores[vid] = score
            
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        return sorted_ids[:k], final_scores
