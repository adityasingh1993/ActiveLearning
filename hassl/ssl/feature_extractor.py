import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

class FeatureExtractor:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # We assume the deepest encoder layer has 256 channels based on the config.
        self.proj = nn.Linear(256, 128).to(self.device)
        
        self.deepest_features = None
        self._register_hook()
        
    def _register_hook(self):
        def hook(module, input, output):
            # Capture the tensor if it has 256 channels (deepest layer in our configs)
            if isinstance(output, torch.Tensor) and output.size(1) == 256:
                self.deepest_features = output
                
        for module in self.model.modules():
            module.register_forward_hook(hook)

    @torch.no_grad()
    def extract_all(self):
        embeddings = {}
        for batch in self.dataloader:
            x = batch['image'].to(self.device)
            vol_id = batch['volume_id']
            
            # Forward pass to trigger hook
            self.model(x)
            
            if self.deepest_features is not None:
                # Global Average Pooling
                pooled = F.adaptive_avg_pool3d(self.deepest_features, 1).view(x.size(0), -1)
                # Linear projection
                proj_out = self.proj(pooled)
                # L2 normalize
                norm_out = F.normalize(proj_out, p=2, dim=1)
                
                for i, vid in enumerate(vol_id):
                    embeddings[vid] = norm_out[i].cpu().numpy()
                    
            self.deepest_features = None
            
        return embeddings

    def save_embeddings(self, embeddings, output_path):
        np.savez(output_path, **embeddings)
        
    def visualize_embeddings(self, embeddings, labeled_ids, save_path):
        ids = list(embeddings.keys())
        feats = np.array([embeddings[i] for i in ids])
        
        labels = [1 if i in labeled_ids else 0 for i in ids]
        colors = ['red' if l == 1 else 'blue' for l in labels]
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        feats_tsne = tsne.fit_transform(feats)
        
        # UMAP
        reducer = umap.UMAP(random_state=42)
        feats_umap = reducer.fit_transform(feats)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=colors, alpha=0.6)
        ax1.set_title('t-SNE Embeddings')
        
        ax2.scatter(feats_umap[:, 0], feats_umap[:, 1], c=colors, alpha=0.6)
        ax2.set_title('UMAP Embeddings')
        
        # Legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)]
        ax1.legend(custom_lines, ['Labeled', 'Unlabeled'])
        ax2.legend(custom_lines, ['Labeled', 'Unlabeled'])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
