#!/usr/bin/env python3
"""Controlled 128^3 SSL pretraining for the Final62 comparison.

Purpose
-------
Train SSL on all legitimate images under config.data_dir while explicitly excluding the
frozen external31 development IDs. The 62 audited HUMAN_GOLD images must all be present.

Frozen SSL experiment choices
-----------------------------
- DynUNet architecture compatible with Final62
- spacing 0.1 mm
- whole-volume resize to 128^3
- batch size 1
- 100 epochs by default
- SSL early stopping disabled
- masked inpainting + rotation + strengthened spatial-token InfoNCE
- InfoNCE uses 4x4x4=64 aligned bottleneck tokens per volume
- contrastive views use independent gamma/scale/shift/noise but no spatial transform

This script does not use external31 images or labels and does not run active learning.
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
from hassl.ssl.ssl_pretrainer import SSLPretrainer
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
    p = argparse.ArgumentParser(description="Leakage-safe controlled SSL pretraining for Final62")
    p.add_argument("--config", required=True)
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--external-case-manifest", default=str(DEFAULT_EXTERNAL_CASES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-only", action="store_true", help="Audit SSL pool/exclusions and exit before training")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.epochs < 1:
        p.error("--epochs must be >=1")
    if args.resize_size != 128:
        p.error("This controlled experiment is intentionally frozen to --resize-size 128")

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
    config.device = "cuda" if config.device == "cuda" else config.device
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

    # Frozen strengthened contrastive settings. These are set explicitly rather than relying
    # on SSLPretrainer defaults so the experiment remains reproducible if defaults later move.
    config.ssl_contrastive_grid_size = 4
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
    config.experiment_name = f"{config.experiment_name}-controlled-ssl-final62-128"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    audit_cache = (
        Path(tempfile.gettempdir()) / "hassl_controlled_ssl_final62_128_audit"
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

    print("=" * 116)
    print("CONTROLLED FINAL62 SSL PRETRAINING — 128^3")
    print(f"Discovered images:        {pool_meta['n_discovered_images']}")
    print(f"Audited HUMAN_GOLD:       {pool_meta['n_audited_human_gold']}")
    print(f"Non-human pool images:    {pool_meta['n_nonhuman_pool_images']}")
    print(f"External IDs manifest:    {pool_meta['n_external_ids_manifest']}")
    print(f"External present in dir:  {pool_meta['n_external_images_present_under_data_dir']}")
    print(f"Final SSL pool:           {pool_meta['n_ssl_pool']}")
    print(f"External overlap:         {len(pool_meta['external_overlap_after_filter'])}")
    print(f"Leakage audit:            {pool_meta['external_overlap_status']}")
    print("Preprocessing:            spacing .1mm -> resize128^3 -> percentile [1,99] to [0,1]")
    print("SSL tasks:                inpainting + rotation + strengthened spatial-token InfoNCE")
    print("Contrastive tokens:       4x4x4 = 64 per volume")
    print("Contrastive views:        gamma .8-1.2 | scale .9-1.1 | shift +/-0.05 | noise std 0-.05")
    print("Contrastive geometry:     unchanged/aligned between the two views")
    print(f"SSL epochs:               {config.ssl_epochs} fixed; early stopping OFF")
    print(f"Physical GPU:             {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("=" * 116)

    if args.audit_only:
        audit_output = output_dir.parent / f"{output_dir.name}_pool_audit_preview.json"
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
    pretrainer = SSLPretrainer(config=config, dataloader=loader, tracker=tracker)
    pretrainer.train(num_epochs=config.ssl_epochs)
    tracker.finish()

    if not legacy_checkpoint.exists():
        raise FileNotFoundError(f"SSLPretrainer did not create {legacy_checkpoint}")

    state = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "version": "controlled_final62_ssl128_v2_stronger_contrastive",
        "purpose": "Final62 SSL-vs-no-SSL controlled comparison",
        "git_commit": git_commit_or_none(),
        "source_config": str(args.config),
        "source_human_audit": str(audit_path),
        "external_case_manifest": str(external_manifest),
        "seed": int(args.seed),
        "ssl_epochs": int(config.ssl_epochs),
        "ssl_early_stopping": False,
        "ssl_tasks": ["masked_volume_inpainting", "rotation_prediction", "spatial_token_infonce"],
        "ssl_task_weights": {"inpainting": 1.0, "rotation": 1.0, "contrastive": 1.0},
        "ssl_mask_ratio": float(config.ssl_mask_ratio),
        "ssl_mask_cube_size": int(config.ssl_mask_cube_size),
        "ssl_contrastive_temp": float(config.ssl_contrastive_temp),
        "ssl_embedding_dim": int(config.ssl_embedding_dim),
        "ssl_contrastive": {
            "grid_size": int(config.ssl_contrastive_grid_size),
            "tokens_per_volume": int(config.ssl_contrastive_grid_size ** 3),
            "symmetric_infonce": True,
            "spatial_geometry_changed": False,
            "gamma_range": [float(config.ssl_contrastive_gamma_min), float(config.ssl_contrastive_gamma_max)],
            "intensity_scale_range": [float(config.ssl_contrastive_scale_min), float(config.ssl_contrastive_scale_max)],
            "intensity_shift_range": [-float(config.ssl_contrastive_shift_abs), float(config.ssl_contrastive_shift_abs)],
            "noise_std_range": [0.0, float(config.ssl_contrastive_noise_std_max)],
        },
        "architecture": "DynUNet",
        "preprocessing": {
            "spacing": [0.1, 0.1, 0.1],
            "mode": "resize",
            "spatial_size": [128, 128, 128],
            "intensity_percentiles": [1, 99],
        },
        "pool_audit": pool_meta,
        "transfer_rule": (
            "Downstream Final62 must transfer compatible SSL feature weights but skip "
            "output_block/deep_supervision_heads, then synchronize EMA teacher to the SSL-initialized student."
        ),
    }
    torch.save({"model": state, "metadata": metadata}, controlled_checkpoint)
    metadata_path = output_dir / "ssl_pretraining_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCONTROLLED SSL PRETRAINING COMPLETE")
    print(f"Legacy state dict:      {legacy_checkpoint}")
    print(f"Controlled checkpoint: {controlled_checkpoint}")
    print(f"Metadata:              {metadata_path}")


if __name__ == "__main__":
    main()
