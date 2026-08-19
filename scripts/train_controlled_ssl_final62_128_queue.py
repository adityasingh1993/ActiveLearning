#!/usr/bin/env python3
"""Controlled Final62 SSL v3: cross-volume memory-queue contrastive learning.

Frozen experiment inputs remain unchanged:
- 62 audited HUMAN_GOLD + 41 legitimate unlabeled = 103 SSL images
- external31 explicitly excluded
- spacing 0.1 mm, whole-volume resize 128^3
- DynUNet, batch size 1, fixed 100 epochs, no SSL early stopping

SSL v3 objective:
    1.0 * masked inpainting
  + 0.1 * rotation prediction
  + 1.0 * cross-volume queue InfoNCE

The queue stores detached global bottleneck embeddings from prior volumes. Same-case entries
are masked when a volume reappears in a later epoch.
"""

import os
import sys


def _consume_gpu_argument(argv):
    args = list(argv)
    gpu = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == "--gpu":
            if i + 1 >= len(args):
                raise SystemExit("--gpu requires 0 or 1")
            gpu = args[i + 1]
            i += 2
            continue
        if token.startswith("--gpu="):
            gpu = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if gpu is not None:
        if gpu not in {"0", "1"}:
            raise SystemExit(f"--gpu must be 0 or 1, got {gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu, cleaned


SELECTED_GPU, CLEAN_ARGV = _consume_gpu_argument(sys.argv)
sys.argv = CLEAN_ARGV

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
from hassl.ssl.controlled import build_controlled_ssl_loader
from hassl.ssl.queue_pretrainer import QueueSSLPretrainer
from hassl.tracking import ExperimentTracker


DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_EXTERNAL_CASES = Path(
    "experiments/external31_final62_inference_modes/external31_inference_mode_case_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/ssl_pretrain_final62_pool_128_v1")


def git_commit_or_none():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Leakage-safe queue SSL pretraining for Final62")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--external-case-manifest", default=str(DEFAULT_EXTERNAL_CASES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--queue-size", type=int, default=256)
    p.add_argument("--lambda-inpainting", type=float, default=1.0)
    p.add_argument("--lambda-rotation", type=float, default=0.1)
    p.add_argument("--lambda-contrastive", type=float, default=1.0)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs < 1:
        p.error("--epochs must be >=1")
    if args.resize_size != 128:
        p.error("This controlled experiment is frozen to --resize-size 128")
    if args.queue_size < 1:
        p.error("--queue-size must be >=1")
    if min(args.lambda_inpainting, args.lambda_rotation, args.lambda_contrastive) < 0:
        p.error("SSL task weights must be non-negative")
    if args.lambda_inpainting + args.lambda_rotation + args.lambda_contrastive <= 0:
        p.error("At least one SSL task weight must be positive")

    audit_path = Path(args.audit_metadata)
    external_manifest = Path(args.external_case_manifest)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    legacy_checkpoint = checkpoint_dir / "ssl_pretrained.pth"
    controlled_checkpoint = checkpoint_dir / "ssl_pretrained_controlled.pth"

    if not args.audit_only and output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif controlled_checkpoint.exists():
            print(f"Controlled SSL checkpoint already exists: {controlled_checkpoint}")
            print("Use --overwrite only if intentionally retraining.")
            return
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")

    config = HASSLConfig.from_yaml(args.config)
    config.compute_mode = "prototype"
    config.batch_size = 1
    config.unet_backbone = "dynunet"
    config.dynunet_deep_supervision = True
    config.dropout = 0.0
    config.spacing = (0.1, 0.1, 0.1)
    config.preprocessing_mode = "resize"
    config.spatial_size = (128, 128, 128)
    config.ssl_epochs = int(args.epochs)
    config.ssl_use_early_stopping = False
    config.seed = int(args.seed)

    # Runtime-only v3 experiment settings.
    config.ssl_lambda_inpainting = float(args.lambda_inpainting)
    config.ssl_lambda_rotation = float(args.lambda_rotation)
    config.ssl_lambda_contrastive = float(args.lambda_contrastive)
    config.ssl_contrastive_queue_size = int(args.queue_size)
    config.ssl_contrastive_gamma_min = 0.8
    config.ssl_contrastive_gamma_max = 1.2
    config.ssl_contrastive_scale_min = 0.9
    config.ssl_contrastive_scale_max = 1.1
    config.ssl_contrastive_shift_abs = 0.05
    config.ssl_contrastive_noise_std_max = 0.05

    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.embedding_dir = str(output_dir / "embeddings")
    config.experiment_name = f"{config.experiment_name}-controlled-ssl-final62-128-queue-v3"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    audit_cache = (
        Path(tempfile.gettempdir()) / "hassl_controlled_ssl_final62_128_queue_audit"
        if args.audit_only
        else Path(config.cache_dir) / "ssl128"
    )
    ssl_config, loader, pool_meta = build_controlled_ssl_loader(
        config=config,
        audit_path=audit_path,
        external_case_manifest=external_manifest,
        output_cache_dir=audit_cache,
        resize_size=128,
        expected_human_count=62,
        expected_external_count=31,
    )
    config = ssl_config

    print("=" * 124)
    print("CONTROLLED FINAL62 SSL PRETRAINING — 128^3 — CROSS-VOLUME QUEUE V3")
    print(f"Discovered images:        {pool_meta['n_discovered_images']}")
    print(f"Audited HUMAN_GOLD:       {pool_meta['n_audited_human_gold']}")
    print(f"Non-human pool images:    {pool_meta['n_nonhuman_pool_images']}")
    print(f"External IDs manifest:    {pool_meta['n_external_ids_manifest']}")
    print(f"External overlap:         {len(pool_meta['external_overlap_after_filter'])}")
    print(f"Leakage audit:            {pool_meta['external_overlap_status']}")
    print(f"Final SSL pool:           {pool_meta['n_ssl_pool']}")
    print("Preprocessing:            spacing .1mm -> resize128^3 -> percentile [1,99] to [0,1]")
    print("Contrastive:              global bottleneck embedding + cross-volume FIFO queue")
    print(f"Queue size:               {config.ssl_contrastive_queue_size}")
    print("Same-case negatives:      MASKED")
    print(
        "SSL task weights:          "
        f"Inp={config.ssl_lambda_inpainting:g} | "
        f"Rot={config.ssl_lambda_rotation:g} | "
        f"Cont={config.ssl_lambda_contrastive:g}"
    )
    print(f"SSL epochs:               {config.ssl_epochs} fixed; early stopping OFF")
    print(f"Physical GPU:             {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("=" * 124)

    if args.audit_only:
        audit_output = output_dir.parent / f"{output_dir.name}_queue_pool_audit_preview.json"
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(pool_meta, indent=2), encoding="utf-8")
        print("AUDIT-ONLY: no SSL training was started.")
        print(f"Audit preview: {audit_output}")
        return

    tracker = ExperimentTracker(
        backend=config.tracker,
        project=config.project_name,
        run_name=f"{config.experiment_name}_ssl_pretrain",
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_config(config.to_dict())
    tracker.log_metrics(
        {
            "ssl_lambda_inpainting": config.ssl_lambda_inpainting,
            "ssl_lambda_rotation": config.ssl_lambda_rotation,
            "ssl_lambda_contrastive": config.ssl_lambda_contrastive,
            "ssl_contrastive_queue_size": float(config.ssl_contrastive_queue_size),
        },
        step=0,
    )

    pretrainer = QueueSSLPretrainer(config=config, dataloader=loader, tracker=tracker)
    pretrainer.train(num_epochs=config.ssl_epochs)
    tracker.finish()

    if not legacy_checkpoint.exists():
        raise FileNotFoundError(f"QueueSSLPretrainer did not create {legacy_checkpoint}")

    state = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "controlled_final62_ssl128_queue_v3",
        "purpose": "Final62 SSL-vs-no-SSL controlled comparison",
        "git_commit": git_commit_or_none(),
        "source_config": str(args.config),
        "source_human_audit": str(audit_path),
        "external_case_manifest": str(external_manifest),
        "seed": int(args.seed),
        "ssl_epochs": int(config.ssl_epochs),
        "ssl_early_stopping": False,
        "ssl_tasks": [
            "masked_volume_inpainting",
            "rotation_prediction",
            "cross_volume_queue_infonce",
        ],
        "ssl_task_weights": {
            "inpainting": float(config.ssl_lambda_inpainting),
            "rotation": float(config.ssl_lambda_rotation),
            "contrastive": float(config.ssl_lambda_contrastive),
        },
        "contrastive": {
            "mode": "global_bottleneck_cross_volume_memory_queue",
            "queue_size": int(config.ssl_contrastive_queue_size),
            "same_case_negatives_masked": True,
            "temperature": float(config.ssl_contrastive_temp),
            "embedding_dim": int(config.ssl_embedding_dim),
            "view_gamma": [0.8, 1.2],
            "view_scale": [0.9, 1.1],
            "view_shift_abs": 0.05,
            "view_noise_std_max": 0.05,
        },
        "preprocessing": {
            "spacing": [0.1, 0.1, 0.1],
            "mode": "resize",
            "spatial_size": [128, 128, 128],
            "intensity_percentiles": [1, 99],
        },
        "pool_audit": pool_meta,
        "transfer_rule": (
            "Downstream Final62 transfers compatible SSL feature weights but skips "
            "output_block/deep_supervision_heads, then synchronizes the EMA teacher to the "
            "SSL-initialized student."
        ),
    }
    torch.save({"model": state, "metadata": metadata}, controlled_checkpoint)
    metadata_path = output_dir / "ssl_pretraining_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCONTROLLED QUEUE SSL PRETRAINING COMPLETE")
    print(f"Legacy state dict:      {legacy_checkpoint}")
    print(f"Controlled checkpoint: {controlled_checkpoint}")
    print(f"Metadata:              {metadata_path}")


if __name__ == "__main__":
    main()
