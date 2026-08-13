import pytest
import numpy as np
import torch

from hassl.active.query_strategies import CoreSetStrategy, BALDStrategy, HybridStrategy, DisagreementStrategy


def test_coreset_strategy():
    # Mock embeddings: 10 unlabeled volumes, 128-dim
    unlabeled_embeddings = {f"vol_{i}": np.random.rand(128) for i in range(10)}
    labeled_embeddings = {f"lvol_{i}": np.random.rand(128) for i in range(2)}

    embeddings_dict = {**unlabeled_embeddings, **labeled_embeddings}
    strategy = CoreSetStrategy(embeddings_dict=embeddings_dict)

    selected_ids, scores = strategy.query(
        unlabeled_ids=list(unlabeled_embeddings.keys()),
        labeled_ids=list(labeled_embeddings.keys()),
        k=3
    )

    assert len(selected_ids) == 3
    for s in selected_ids:
        assert s in unlabeled_embeddings


def test_bald_strategy(sample_volume):
    from hassl.training.trainer import build_network
    model = build_network('unet', num_classes=1, dropout=0.2)

    strategy = BALDStrategy(model=model, T=3)
    scores = strategy.score(sample_volume)

    assert len(scores) == 1
    assert float(scores[0]) >= 0.0


def test_hybrid_strategy():
    from hassl.training.trainer import build_network
    model_a = build_network('unet', num_classes=1, dropout=0.2)
    model_b = build_network('unet', num_classes=1, dropout=0.2)

    embeddings = {f"vol_{i}": np.random.rand(128) for i in range(5)}
    embeddings.update({f"lvol_{i}": np.random.rand(128) for i in range(2)})

    bald_strat = BALDStrategy(model=model_a, T=2)
    coreset_strat = CoreSetStrategy(embeddings_dict=embeddings)
    disagreement_strat = DisagreementStrategy(model_a=model_a, model_b=model_b)

    hybrid = HybridStrategy(
        bald_strategy=bald_strat,
        coreset_strategy=coreset_strat,
        disagreement_strategy=disagreement_strat,
        alpha=0.4, beta=0.3, gamma=0.3
    )

    # Mock DataLoader-like dataset batch
    class DummyLoader:
        def __iter__(self):
            yield {'image': torch.zeros(2, 1, 64, 64, 64), 'id': ['vol_0', 'vol_1']}

    selected_ids, final_scores = hybrid.query(
        unlabeled_loader=DummyLoader(),
        labeled_ids=['lvol_0', 'lvol_1'],
        k=2
    )

    assert len(selected_ids) == 2
    assert 'vol_0' in selected_ids or 'vol_1' in selected_ids
