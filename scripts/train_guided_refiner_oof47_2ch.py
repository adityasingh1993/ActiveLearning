#!/usr/bin/env python3
"""Train the first 2-channel guided ROI refiner using existing Round2 OOF47 guidance.

Input channels
--------------
0: ultrasound ROI
1: leakage-safe OOF Round2 EMA probability ROI

The 47 training ROIs are proposed from OOF predictions; GT does not repair the ROI. The model
is an OFFLINE active-learning/refinement experiment only. Production inference remains a
single-stage one-channel segmentation model.
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
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.networks.nets import DynUNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    Resized,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import set_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import ORIENTATIOND_RAS_LABELS
from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
import hassl.training.trainer as trainer_module
import scripts.train_supervised_cv as cv
from scripts.train_final_supervised_round2_from_audit import read_cv_best_epochs

DEFAULT_DATASET_DIR = Path("experiments/guided_refiner_oof47_roi_dataset_v1")
DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_OUTPUT_DIR = Path("experiments/guided_refiner_oof47_2ch_v1")
EXPECTED_CASES = 47


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ConcatGuidedChannels:
    """Replace image with [ultrasound, coarse_probability] channels."""

    def __call__(self, data):
        d = dict(data)
        image = d["image"]
        coarse = d["coarse"]
        if tuple(image.shape[1:]) != tuple(coarse.shape[1:]):
            raise RuntimeError(f"Guided channel shape mismatch: image={image.shape}, coarse={coarse.shape}")
        d["image"] = torch.cat([image, coarse], dim=0)
        d.pop("coarse", None)
        return d


def build_guided_transform(config, spatial_size, training):
    keys = ["image", "coarse", "label"]
    steps = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=ORIENTATIOND_RAS_LABELS),
    ]
    if config.spacing and all(float(x) >= 0.1 for x in config.spacing):
        steps.append(
            Spacingd(
                keys=keys,
                pixdim=config.spacing,
                mode=("bilinear", "bilinear", "nearest"),
            )
        )
    steps.extend([
        AsDiscreted(keys=["label"], threshold=0.5),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True, channel_wise=True
        ),
        Resized(
            keys=keys,
            spatial_size=(int(spatial_size),) * 3,
            mode=("trilinear", "trilinear", "nearest"),
        ),
    ])
    if training:
        steps.append(
            RandAffined(
                keys=keys,
                prob=0.8,
                rotate_range=(0.0, 0.0, 0.0),
                translate_range=(12.0, 12.0, 12.0),
                scale_range=(0.0, 0.0, 0.0),
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="zeros",
            )
        )
    steps.append(ConcatGuidedChannels())
    return Compose(steps)


def build_guided_dynunet(backbone, num_classes, dropout):
    if backbone != "dynunet":
        raise RuntimeError("Guided feasibility experiment is frozen to DynUNet")
    return DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=num_classes,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 4,
        filters=[16, 32, 64, 128, 256],
        dropout=dropout,
        norm_name="instance",
        deep_supervision=True,
    )


def install_guided_loader_hook(cases, config, resize_size):
    ordered = sorted(cases, key=lambda x: str(x["id"]))
    train_t = build_guided_transform(config, resize_size, training=True)
    val_t = build_guided_transform(config, resize_size, training=False)

    def build_guided_dataloaders(runtime_config):
        train_ds = Dataset(ordered, transform=train_t)
        val_ds = Dataset(ordered, transform=val_t)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(runtime_config, "batch_size", 1)),
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_guided_dataloaders
    pipeline_module.build_dataloaders = build_guided_dataloaders


def main():
    p = argparse.ArgumentParser(description="Train 2-channel guided OOF47 ROI refiner")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "guided_refiner_oof47_roi_manifest.csv"
    metadata_path = dataset_dir / "guided_refiner_oof47_roi_metadata.json"
    rows = read_csv(manifest_path)
    if len(rows) != EXPECTED_CASES or len({str(r["case_id"]) for r in rows}) != EXPECTED_CASES:
        raise RuntimeError("Guided refiner dataset must contain exactly 47 unique OOF cases")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Guided refiner requires prototype DynUNet configuration")

    epoch_rows = read_cv_best_epochs(Path(args.round2_cv_dir))
    best_epochs = np.asarray([int(r["best_epoch"]) for r in epoch_rows], dtype=int)
    median_best_epoch = int(round(float(np.median(best_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_best_epoch
    if final_epochs < 1:
        p.error("--epochs must be >=1")

    cases = []
    for row in rows:
        for key in ["image_path", "coarse_probability_path", "label_path"]:
            if not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
        cases.append({
            "id": str(row["case_id"]),
            "image": str(row["image_path"]),
            "coarse": str(row["coarse_probability_path"]),
            "label": str(row["label_path"]),
        })

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif final_checkpoint.exists():
            print(f"Guided refiner checkpoint already exists: {final_checkpoint}")
            print("Use --overwrite only if intentionally retraining.")
            return
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")

    cv.apply_baseline(config, args.resize_size, final_epochs)
    config.seed = int(args.seed)
    config.use_early_stopping = False
    config.checkpoint_dir = str(checkpoint_dir)
    config.cache_dir = str(output_dir / "cache")
    config.log_dir = str(output_dir / "logs")
    config.experiment_name = f"{config.experiment_name}-guided-oof47-2ch"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_determinism(seed=args.seed)

    # Patch only this process. The repository's normal one-channel production architecture is untouched.
    trainer_module.build_network = build_guided_dynunet
    cv.build_network = build_guided_dynunet
    install_guided_loader_hook(cases, config, args.resize_size)

    print("=" * 116)
    print("2-CHANNEL GUIDED ROI REFINER — OOF47 FEASIBILITY TRAINING")
    print(f"Cases:               {len(cases)} true-OOF guided ROIs")
    print("Channels:            ultrasound + OOF EMA probability")
    print(f"CV best epochs:      {best_epochs.tolist()}")
    print(f"Median best epoch:   {median_best_epoch}")
    print(f"Final epochs:        {final_epochs}")
    print(f"Physical GPU:        {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("Recipe:              2ch DynUNet | resize128 | DiceCE | AdamW 1e-4 | EMA | dropout=0")
    print("Augmentation:        paired image/coarse/label translation +/-12 vox, p=0.8")
    print("Validation:          same-data diagnostic only; fixed epoch selects final checkpoint")
    print("=" * 116)

    pipeline_module.run_train(config, round_num=0, pretrained_weights=None)

    latest = checkpoint_dir / "round0_latest.pth"
    if not latest.exists():
        raise FileNotFoundError(f"Expected fixed-epoch latest checkpoint: {latest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, final_checkpoint)

    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    training_meta = {
        "version": "guided_refiner_oof47_2ch_v1",
        "purpose": "offline guided ROI feasibility experiment",
        "n_training_cases": len(cases),
        "source_dataset": str(dataset_dir),
        "input_channels": ["ultrasound", "leakage_safe_oof_ema_probability"],
        "in_channels": 2,
        "seed": int(args.seed),
        "cv_best_epochs_from_round2": [int(x) for x in best_epochs.tolist()],
        "final_training_epochs": final_epochs,
        "checkpoint_saved_epoch": int(state.get("epoch", final_epochs)),
        "deployment_checkpoint": str(final_checkpoint),
        "prediction_threshold": 0.50,
        "dataset_localization_diagnostics": {
            "median_gt_coverage": dataset_meta.get("median_gt_coverage"),
            "gt_coverage_lt_090": dataset_meta.get("gt_coverage_lt_090"),
            "gt_coverage_lt_050": dataset_meta.get("gt_coverage_lt_050"),
        },
        "warning": (
            "This 2-channel model is an OFFLINE refiner experiment. It does not replace or alter the "
            "single-stage production segmentation model requirement."
        ),
    }
    train_meta_path = output_dir / "guided_refiner_training_metadata.json"
    train_meta_path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    print("\nGUIDED REFINER TRAINING COMPLETE")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Metadata:   {train_meta_path}")


if __name__ == "__main__":
    main()
