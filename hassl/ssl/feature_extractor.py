import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Union, Optional

try:
    import umap
except ImportError:
    umap = None


class FeatureExtractor:
    """Extracts bottleneck feature embeddings and visualizes embedding space (H-6 & M-14 fix)."""

    def __init__(self, encoder, dataloader, embedding_dim: int = 128, device: str = 'cuda', model=None):
        self.model = encoder if encoder is not None else model
        self.dataloader = dataloader
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.proj = nn.LazyLinear(embedding_dim).to(self.device)
        self.embeddings_cache = {}

    def _extract_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deep bottleneck feature map directly (M-14 fix)."""
        if hasattr(self.model, 'input_block') and hasattr(self.model, 'bottleneck'):
            # DynUNet bottleneck
            h = self.model.input_block(x)
            for down in self.model.downsamples:
                h = down(h)
            return self.model.bottleneck(h)
        elif hasattr(self.model, 'model'):
            # UNet bottleneck
            h = x
            for block in self.model.model:
                h = block(h)
                if h.shape[1] >= 128:
                    break
            return h
        else:
            out = self.model(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    @torch.no_grad()
    def extract_all((self) -> Dict[str, np.ndarray]:
        """Extract 128-dim normalized feature embeddings for all volumes."""
        self.embeddings_cache = {}

        for batch in self.dataloader:
            x = batch['image'].to(self.device)
            vol_ids = batch.get('id', batch.get('volume_id', []))

            bottleneck = self._extract_bottleneck(x)
            pooled = F.adaptive_avg_pool3d(bottleneck, 1).view(x.size(0), -1)

            proj_out = self.proj(pooled)
            norm_out = F.normalize(proj_out, p=2, dim=1)

            for i, vid in enumerate(vol_ids):
                self.embeddings_cache[vid] = norm_out[i].cpu().numpy()

        return self.embeddings_cache

    def save_embeddings(self, output_path_or_embeddings, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save extracted embeddings to .npz file supporting both positional conventions (H-6 fix)."""
        if output_path is None:
            path = Path(output_path_or_embeddings)
            embeddings = self.embeddings_cache
        else:
            path = Path(output_path)
            embeddings = output_path_or_embeddings

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **embeddings)
        print(f"  Embeddings saved to {path}")

    def visualize_embeddings(self, embeddings: Dict[str, np.ndarray], labeled_ids: List[str], save_path: Union[str, Path]) -> None:
        """Generate t-SNE and UMAP scatter plots colored by labeled/unlabeled status."""
        if not embeddings:
            print("  No embeddings available for visualization.")
            return

        ids = list(embeddings.keys())
        feats = np.array([embeddings[i] for i in ids])
        labeled_set = set(labeled_ids)

        labels = [1 if i in labeled_set else 0 for i in ids]
        colors = ['#3fb950' if l == 1 else '#58a6ff' for l in labels]

        # t-SNE
        perplexity = min(30, max(2, len(ids) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        feats_tsne = tsne.fit_transform(feats)

        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

        ax1.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=colors, alpha=0.7, edgecolors='none', s=40)
        ax1.set_title('t-SNE Embedding Space (HASSL Features)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3fb950', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#58a6ff', markersize=10),
        ]
        ax1.legend(custom_lines, ['Labeled', 'Unlabeled'])

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close()
