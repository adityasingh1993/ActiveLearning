import pytest
import numpy as np
from hassl.active.query_strategies import CoreSetStrategy, BALDStrategy, HybridStrategy

def test_coreset_strategy():
    strategy = CoreSetStrategy()
    # Mock embeddings: 10 unlabeled volumes, 128-dim
    unlabeled_embeddings = {f"vol_{i}": np.random.rand(128) for i in range(10)}
    labeled_embeddings = {f"lvol_{i}": np.random.rand(128) for i in range(2)}
    
    selected = strategy.query(unlabeled_embeddings, k=3, labeled_embeddings=labeled_embeddings)
    assert len(selected) == 3
    for s in selected:
        assert s in unlabeled_embeddings

def test_bald_strategy():
    strategy = BALDStrategy()
    # Mock probabilities from T passes
    # Format: dict mapping vol_id to array of shape (T, C, D, H, W)
    probs = {
        f"vol_{i}": np.random.rand(5, 2, 32, 32, 32) for i in range(4)
    }
    
    scores = strategy.score(probs)
    assert len(scores) == 4
    for k, v in scores.items():
        assert isinstance(v, float)
        assert v >= 0

def test_hybrid_strategy():
    strategy = HybridStrategy(alpha=0.4, beta=0.3, gamma=0.3)
    # Provide mock scores from BALD, CoreSet distances, and Disagreement
    scores = strategy.query(
        unlabeled_ids=[f"vol_{i}" for i in range(5)],
        k=2,
        bald_scores={f"vol_{i}": np.random.rand() for i in range(5)},
        coreset_distances={f"vol_{i}": np.random.rand() for i in range(5)},
        disagreement_scores={f"vol_{i}": np.random.rand() for i in range(5)}
    )
    assert len(scores) == 2
    assert scores[0] in [f"vol_{i}" for i in range(5)]
