"""
HASSL Pipeline Orchestrator.

Main entry point for running the HASSL pipeline phases:
    - pretrain: Self-supervised pre-training on all volumes
    - train: Semi-supervised training (UA-MT or CPS)
    - query: Active learning query to select informative volumes
    - export-preseg: Export AI pre-segmentation masks for 3D Slicer
    - al-round: Run a complete active learning round (query + retrain)
    - all: Run the full pipeline end-to-end

Usage:
    python -m hassl.pipeline --config config.yaml --phase pretrain
    python -m hassl.pipeline --config config.yaml --phase train
    python -m hassl.pipeline --config config.yaml --phase query
    python -m hassl.pipeline --config config.yaml --phase al-round --round 1
    python -m hassl.pipeline --config config_full.yaml --phase train  # 24GB mode
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

from hassl.config import HASSLConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_pretrain(config: HASSLConfig) -> None:
    """Phase 2: Self-supervised pre-training on all volumes.

    Pre-trains the encoder (UNet or DynUNet) on all ~300 volumes using:
        - Masked Volume Inpainting
        - Contrastive Learning
        - Rotation Prediction

    Also extracts and caches feature embeddings for active learning.
    """
    from hassl.tracking import ExperimentTracker
    from hassl.data.data_engine import build_unlabeled_dataset
    from hassl.ssl.ssl_pretrainer import SSLPretrainer
    from hassl.ssl.feature_extractor import FeatureExtractor

    print("=" * 60)
    print("HASSL Phase 2: Self-Supervised Pre-training")
    print(f"  Compute mode: {config.compute_mode}")
    print(f"  Backbone: {config.unet_backbone}")
    print(f"  Epochs: {config.ssl_epochs}")
    print("=" * 60)

    tracker = ExperimentTracker(
        backend=config.tracker,
        project=config.project_name,
        run_name=f"{config.experiment_name}_ssl_pretrain",
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_config(config.to_dict())

    # Build dataloader for all volumes (no labels needed)
    from hassl.data.data_engine import build_all_volumes_loader
    all_loader = build_all_volumes_loader(config)
    print(f"  Total volumes for SSL: {len(all_loader.dataset)}")

    # Pre-train
    pretrainer = SSLPretrainer(config=config, dataloader=all_loader, tracker=tracker)
    pretrainer.train(num_epochs=config.ssl_epochs)

    # Extract embeddings
    print("\nExtracting feature embeddings for active learning...")
    extractor = FeatureExtractor(
        encoder=pretrainer.get_encoder(),
        dataloader=all_loader,
        embedding_dim=config.ssl_embedding_dim,
        device=config.device,
    )
    embeddings = extractor.extract_all()
    extractor.save_embeddings(Path(config.embedding_dir) / "ssl_embeddings.npz")

    # Visualize embeddings with labeled/unlabeled distinction
    from hassl.active.query_engine import QueryEngine
    engine = QueryEngine(config=config)
    engine.initialize_pool()
    labeled_ids = engine.get_labeled_ids()

    extractor.visualize_embeddings(
        embeddings=embeddings,
        labeled_ids=labeled_ids,
        save_path=Path(config.embedding_dir) / "embedding_tsne.png",
    )
    tracker.log_artifact(
        str(Path(config.embedding_dir) / "embedding_tsne.png"),
        name="embedding_visualization",
    )

    tracker.finish()
    print("\n✓ SSL pre-training complete. Embeddings saved.")


def run_train(config: HASSLConfig, round_num: int = 0) -> None:
    """Phase 3: Semi-supervised training.

    Trains using UA-Mean Teacher (prototype) or CPS (full mode)
    on labeled + unlabeled data.

    Args:
        config: HASSL configuration.
        round_num: Active learning round number (0 = initial training).
    """
    from hassl.tracking import ExperimentTracker
    from hassl.data.data_engine import build_dataloaders
    from hassl.training.trainer import HASSLTrainer

    print("=" * 60)
    print(f"HASSL Phase 3: Semi-Supervised Training (Round {round_num})")
    print(f"  Compute mode: {config.compute_mode}")
    mode_str = "UA-Mean Teacher" if config.compute_mode == "prototype" else "Cross-Pseudo Supervision"
    print(f"  Training mode: {mode_str}")
    print(f"  Backbone: {config.unet_backbone}")
    print(f"  Epochs: {config.train_epochs}")
    print("=" * 60)

    tracker = ExperimentTracker(
        backend=config.tracker,
        project=config.project_name,
        run_name=f"{config.experiment_name}_train_round{round_num}",
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_config(config.to_dict())
    tracker.log_metrics({"al_round": round_num}, step=0)

    # Build dataloaders
    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)
    print(f"  Labeled volumes: {len(labeled_loader.dataset)}")
    print(f"  Unlabeled volumes: {len(unlabeled_loader.dataset)}")
    print(f"  Validation volumes: {len(val_loader.dataset)}")

    # Load SSL pre-trained weights if available
    ssl_checkpoint = Path(config.checkpoint_dir) / "ssl_pretrained.pth"
    pretrained_weights = str(ssl_checkpoint) if ssl_checkpoint.exists() else None

    # Train
    trainer = HASSLTrainer(
        config=config,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        tracker=tracker,
        pretrained_weights=pretrained_weights,
    )

    # Resume from checkpoint if available for this round
    resume_path = Path(config.checkpoint_dir) / f"round{round_num}_latest.pth"
    if resume_path.exists():
        print(f"  Resuming from {resume_path}")
        trainer.resume(str(resume_path))

    trainer.train(num_epochs=config.train_epochs)

    tracker.finish()
    print(f"\n✓ Training round {round_num} complete.")


def run_query(config: HASSLConfig, round_num: int = 1) -> None:
    """Phase 4: Active learning query (H-3 fix).

    Selects the most informative unlabeled volumes using Hybrid Strategy
    (BALD + CoreSet + Disagreement) and exports AI pre-segmentations.
    """
    from hassl.tracking import ExperimentTracker
    from hassl.active.query_engine import QueryEngine
    from hassl.active.query_strategies import (
        BALDStrategy, CoreSetStrategy, DisagreementStrategy, HybridStrategy
    )
    from hassl.training.trainer import HASSLTrainer
    from hassl.data.data_engine import build_dataloaders

    print("=" * 60)
    print(f"HASSL Phase 4: Active Learning Query (Round {round_num})")
    print(f"  Strategy: {config.al_strategy}")
    print(f"  Query size: {config.al_query_size}")
    print("=" * 60)

    tracker = ExperimentTracker(
        backend=config.tracker,
        project=config.project_name,
        run_name=f"{config.experiment_name}_query_round{round_num}",
        tracking_uri=config.mlflow_tracking_uri,
    )

    # Load dataloaders
    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)

    if unlabeled_loader is None or len(unlabeled_loader.dataset) == 0:
        print("  No unlabeled volumes remaining to query.")
        tracker.finish()
        return

    # Load trained models
    trainer = HASSLTrainer(config=config, labeled_loader=labeled_loader,
                           unlabeled_loader=unlabeled_loader, val_loader=val_loader,
                           tracker=tracker)

    best_ckpt = Path(config.checkpoint_dir) / "best_checkpoint.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))

    model_A, model_B = trainer.get_models()

    # Load SSL embeddings for CoreSet
    embedding_path = Path(config.embedding_dir) / "ssl_embeddings.npz"
    embeddings_dict = {}
    if embedding_path.exists():
        data = np.load(str(embedding_path), allow_pickle=True)
        embeddings_dict = {k: v for k, v in data.items()}

    # Build strategies (H-3 fix)
    bald_strat = BALDStrategy(model=model_A, num_passes=config.mc_dropout_passes)
    coreset_strat = CoreSetStrategy(embeddings_dict=embeddings_dict)
    disagreement_strat = DisagreementStrategy(model_a=model_A, model_b=model_B)

    w_bald, w_core, w_dis = config.al_hybrid_weights
    strategy = HybridStrategy(
        bald_strategy=bald_strat,
        coreset_strategy=coreset_strat,
        disagreement_strategy=disagreement_strat,
        alpha=w_bald, beta=w_core, gamma=w_dis,
    )

    engine = QueryEngine(config=config, tracker=tracker)
    engine.initialize_pool()

    # Run query with strategy & unlabeled_loader (H-3 fix)
    queried_ids, scores = engine.run_query(
        strategy=strategy,
        unlabeled_loader=unlabeled_loader,
        round_num=round_num,
        k=config.al_query_size,
    )

    # Display results
    print(f"\n{'─' * 40}")
    print(f"Top {len(queried_ids)} volumes queried by {config.al_strategy} strategy:")
    print(f"{'─' * 40}")
    for i, (vol_id, score) in enumerate(zip(queried_ids, scores)):
        print(f"  {i+1}. {vol_id}  (score: {score:.4f})")
    print(f"{'─' * 40}")

    # Export AI pre-segmentation masks with trained model
    print("\nExporting AI pre-segmentation masks for review...")
    engine.export_presegmentation(
        model=model_A,
        dataloader=unlabeled_loader,
        volume_ids=queried_ids,
    )
    print(f"  Saved to: {config.preseg_dir}/")

    tracker.finish()


def run_al_round(config: HASSLConfig, round_num: int) -> None:
    """Run a complete active learning round.

    1. Detect new labels added since last round
    2. Retrain with expanded label pool
    3. Query next batch of informative volumes

    Args:
        config: HASSL configuration.
        round_num: Active learning round number.
    """
    from hassl.active.query_engine import QueryEngine

    print("=" * 60)
    print(f"HASSL Active Learning Round {round_num}")
    print("=" * 60)

    # Step 1: Detect new human labels
    engine = QueryEngine(config=config)
    new_labels = engine.detect_new_labels()
    if new_labels:
        print(f"  Detected {len(new_labels)} new human-labeled volumes: {new_labels}")
    else:
        print("  No new human labels detected. Proceeding with existing labels.")

    # Step 2: Retrain
    print(f"\n  Retraining with expanded label pool...")
    run_train(config, round_num=round_num)

    # Step 3: Query next batch
    if round_num < config.al_rounds:
        print(f"\n  Querying next batch of informative volumes...")
        run_query(config, round_num=round_num + 1)
    else:
        print(f"\n  All {config.al_rounds} AL rounds complete!")


def run_export_preseg(config: HASSLConfig) -> None:
    """Export AI pre-segmentation for all unlabeled volumes."""
    from hassl.active.query_engine import QueryEngine
    from hassl.training.trainer import HASSLTrainer
    from hassl.data.data_engine import build_dataloaders
    from hassl.tracking import ExperimentTracker

    print("=" * 60)
    print("HASSL: Exporting AI Pre-segmentation Masks")
    print("=" * 60)

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)

    if unlabeled_loader is None or len(unlabeled_loader.dataset) == 0:
        print("  No unlabeled volumes available.")
        return

    tracker = ExperimentTracker(backend="none")
    trainer = HASSLTrainer(config=config, labeled_loader=labeled_loader,
                           unlabeled_loader=unlabeled_loader, val_loader=val_loader,
                           tracker=tracker)

    best_ckpt = Path(config.checkpoint_dir) / "best_checkpoint.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))

    model_A, _ = trainer.get_models()

    engine = QueryEngine(config=config)
    print(f"  Exporting predictions for {len(unlabeled_loader.dataset)} unlabeled volumes...")

    engine.export_presegmentation(model=model_A, dataloader=unlabeled_loader)
    print(f"  Saved to: {config.preseg_dir}/")


def run_auto_loop(config: HASSLConfig) -> None:
    """Run fully automated iterative self-training (Zero Slicer / Zero Manual Labor).

    Pre-trains on all volumes (SSL), trains initial model, and then automatically
    promotes high-confidence pseudo-labeled volumes into training pool over multiple rounds.
    """
    from hassl.active.query_engine import QueryEngine
    from hassl.data.data_engine import build_dataloaders

    start_time = time.time()

    print("=" * 60)
    print("HASSL: Fully Automated Iterative Self-Training (Zero Manual Annotation)")
    print(f"  Compute mode: {config.compute_mode}")
    print(f"  Backbone: {config.unet_backbone}")
    print(f"  Automated Rounds: {config.al_rounds}")
    print("=" * 60)

    # 1. Pre-train (SSL)
    run_pretrain(config)

    # 2. Initial Training on 50 labels
    run_train(config, round_num=0)

    # 3. Automated Pseudo-Labeling & Retraining Rounds
    engine = QueryEngine(config=config)
    engine.initialize_pool()

    for r in range(1, config.al_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"  Automated Self-Training Round {r}/{config.al_rounds}")
        print(f"{'=' * 60}")

        # Build current dataloaders
        labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)

        if unlabeled_loader and len(unlabeled_loader.dataset) > 0:
            # Load best model from previous round
            from hassl.training.trainer import HASSLTrainer
            from hassl.tracking import ExperimentTracker

            tracker = ExperimentTracker(backend="none")
            trainer = HASSLTrainer(config=config, labeled_loader=labeled_loader,
                                   unlabeled_loader=unlabeled_loader, val_loader=val_loader,
                                   tracker=tracker)
            best_ckpt = Path(config.checkpoint_dir) / "best_checkpoint.pth"
            if best_ckpt.exists():
                trainer.load_checkpoint(str(best_ckpt))

            model, _ = trainer.get_models()
            promoted = engine.auto_promote_pseudo_labels(model=model, dataloader=unlabeled_loader, k=config.al_query_size)
            print(f"  Auto-promoted {len(promoted)} pseudo-labeled volumes to pool: {promoted}")

        # Retrain on expanded dataset
        run_train(config, round_num=r)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✓ Automated Self-Training Complete! Total time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")



def run_serve(config: HASSLConfig, port: int = 8000) -> None:
    """Launch the Web-Based Annotation UI server (Option C).

    Starts FastAPI web server on http://localhost:8000 for reviewing slices,
    AI pre-segmentations, and accepting labels in the browser.
    """
    from hassl.app.server import start_server
    start_server(config=config, port=port)


def run_all(config: HASSLConfig) -> None:
    """Run the full HASSL pipeline end-to-end.

    Phase 1: Data engine (automatic via dataloaders)
    Phase 2: SSL pre-training
    Phase 3: Initial semi-supervised training
    Phase 4: Active learning query (Round 1)
    """
    start_time = time.time()

    print("=" * 60)
    print("HASSL: Full Pipeline Execution")
    print(f"  Compute mode: {config.compute_mode}")
    print(f"  Backbone: {config.unet_backbone}")
    print("=" * 60)

    # Phase 2
    run_pretrain(config)

    # Phase 3: Initial training
    run_train(config, round_num=0)

    # Phase 4: First AL query
    run_query(config, round_num=1)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"HASSL Pipeline Complete! Total time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Open queried volumes in 3D Slicer with AI pre-segmentation")
    print(f"  2. Correct predictions and save to {config.data_dir}/labels/")
    print(f"  3. Run: python -m hassl.pipeline --phase al-round --round 1")


def main():
    """CLI entry point for the HASSL pipeline."""
    parser = argparse.ArgumentParser(
        description="HASSL: Hybrid Active Semi-Supervised Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (prototype mode)
  python -m hassl.pipeline --config config.yaml

  # Individual phases
  python -m hassl.pipeline --config config.yaml --phase pretrain
  python -m hassl.pipeline --config config.yaml --phase train
  python -m hassl.pipeline --config config.yaml --phase query
  python -m hassl.pipeline --config config.yaml --phase export-preseg

  # Active learning round (after annotating queried volumes)
  python -m hassl.pipeline --config config.yaml --phase al-round --round 1

  # Switch to 24GB full mode
  python -m hassl.pipeline --config config_full.yaml --phase train

  # Override compute mode from CLI
  python -m hassl.pipeline --config config.yaml --compute-mode full
        """,
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "pretrain", "train", "query", "export-preseg", "al-round", "auto-loop", "serve"],
        help="Pipeline phase to run (default: all)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port number for web server when phase=serve (default: 8000)",
    )

    parser.add_argument(
        "--round", type=int, default=1,
        help="Active learning round number (for al-round and query phases)",
    )
    parser.add_argument(
        "--compute-mode", type=str, choices=["prototype", "full"],
        help="Override compute mode from config",
    )
    parser.add_argument(
        "--backbone", type=str, choices=["unet", "dynunet"],
        help="Override UNet backbone from config",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Create one with: python -c \"from hassl.config import HASSLConfig; HASSLConfig().to_yaml('{args.config}')\"")
        sys.exit(1)

    config = HASSLConfig.from_yaml(str(config_path))

    # Apply CLI overrides
    if args.compute_mode:
        config.compute_mode = args.compute_mode
        config.__post_init__()  # Re-validate
    if args.backbone:
        config.unet_backbone = args.backbone

    # Set seed
    set_seed(config.seed)

    # Print config summary
    print(f"\nHASSL v0.1.0")
    print(f"Config: {config_path}")
    print(f"Device: {config.device} ({config.compute_mode} mode)")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    # Dispatch to phase
    phase_dispatch = {
        "all": lambda: run_all(config),
        "pretrain": lambda: run_pretrain(config),
        "train": lambda: run_train(config, round_num=args.round if args.phase == "al-round" else 0),
        "query": lambda: run_query(config, round_num=args.round),
        "export-preseg": lambda: run_export_preseg(config),
        "al-round": lambda: run_al_round(config, round_num=args.round),
        "auto-loop": lambda: run_auto_loop(config),
        "serve": lambda: run_serve(config, port=args.port),
    }

    try:
        phase_dispatch[args.phase]()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoints saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
