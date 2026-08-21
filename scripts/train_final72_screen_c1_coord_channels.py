#!/usr/bin/env python3
"""Screen C1: coordinate-aware DynUNet on top of the locked Final72 A3 recipe.

C1 changes exactly one modeling variable: the network input is expanded from one channel
(ultrasound intensity) to four channels:
    [ultrasound, coord_LR, coord_AP, coord_SI]
where each coordinate channel is a fixed normalized grid in [-1, +1] after deterministic
RAS/spacing/resize preprocessing and AFTER A3 spatial augmentation. Therefore the coordinates
represent the final absolute location in the model input, rather than moving with the anatomy.

Everything else stays matched to A3:
- Final72 HUMAN_GOLD, frozen original47 OOF folds
- DynUNet, resize128, DiceCE
- translation +/-4 vox p=.5
- LR flip p=.5 (RAS spatial axis 0)
- AdamW 1e-4, dropout0, lambda_unsup0, 100 epochs
- random initialization with same fold seeds
- raw Student+EMA 50/50 ensemble at threshold .50
- no LCC, no external31

Default screening folds are 1 and 2. Fixed validation SMALL/MEDIUM/LARGE groups are diagnostic
only and do not affect training.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DynUNet
from monai.transforms import Compose, RandAffined, RandFlipd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import hassl.data.data_engine as data_engine
import hassl.pipeline as pipeline_module
import hassl.training.trainer as trainer_module
import scripts.train_supervised_cv as cv
import scripts.train_final72_screen_spatial_folds12 as spatial_screen
from scripts.train_final72_screen_small_bladder_oversampling import (
    compare_to_a3,
    read_csv,
)

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
A3_CV = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
SIZE_PROFILE = Path("experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv")
DEFAULT_OUTPUT = Path("experiments/final72_screen_c1_coord_channels_a3")
INPUT_CHANNELS = 4


class AddCoordinateChannelsd:
    """Append fixed normalized LR/AP/SI coordinate channels to ``image``.

    The transform must be placed after all random spatial augmentation. The first spatial axis
    is treated as LR because the upstream deterministic transform has already oriented the
    volume to RAS, matching the A3 LR-flip convention used in this experiment series.
    """

    def __init__(self, key="image"):
        self.key = key

    def __call__(self, data):
        out = dict(data)
        image = out[self.key]
        is_meta = isinstance(image, MetaTensor)
        tensor = image.as_tensor() if is_meta else torch.as_tensor(image)
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise RuntimeError(
                f"C1 expects image [1,S0,S1,S2] before coordinates, got {tuple(tensor.shape)}"
            )

        dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
        device = tensor.device
        s0, s1, s2 = [int(x) for x in tensor.shape[1:]]
        a0 = torch.linspace(-1.0, 1.0, s0, dtype=dtype, device=device)
        a1 = torch.linspace(-1.0, 1.0, s1, dtype=dtype, device=device)
        a2 = torch.linspace(-1.0, 1.0, s2, dtype=dtype, device=device)
        c0, c1, c2 = torch.meshgrid(a0, a1, a2, indexing="ij")
        stacked = torch.cat(
            [tensor.to(dtype=dtype), c0.unsqueeze(0), c1.unsqueeze(0), c2.unsqueeze(0)],
            dim=0,
        )
        if tuple(stacked.shape) != (4, s0, s1, s2):
            raise RuntimeError(f"Coordinate concatenation produced invalid shape {tuple(stacked.shape)}")

        if is_meta:
            out[self.key] = MetaTensor(
                stacked,
                affine=image.affine.clone(),
                meta=dict(image.meta),
            )
        else:
            out[self.key] = stacked
        return out


def c1_train_transform(base_transform):
    """Exact A3 augmentation followed by fixed coordinate channels."""
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        AddCoordinateChannelsd("image"),
    ])


def c1_val_transform(base_transform):
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [AddCoordinateChannelsd("image")])


def make_dataset(items, transform, use_cache):
    if use_cache and items:
        return CacheDataset(items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(items, transform=transform)


def install_c1_loader_hook():
    def build_cv_dataloaders(config):
        cases = {c["id"]: c for c in cv.collect_cases(config)}
        train_ids = sorted(list(config._cv_train_ids))
        val_ids = sorted(list(config._cv_val_ids))
        missing = sorted((set(train_ids) | set(val_ids)) - set(cases))
        if missing:
            raise RuntimeError(f"Missing labeled cases for C1 fold: {missing}")

        train_base = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        val_base = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )
        train_t = c1_train_transform(train_base)
        val_t = c1_val_transform(val_base)

        use_cache = bool(getattr(config, "use_cache_dataset", True))
        train_ds = make_dataset([cases[x] for x in train_ids], train_t, use_cache)
        val_ds = make_dataset([cases[x] for x in val_ids], val_t, use_cache)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(getattr(config, "batch_size", 1)),
            shuffle=True,
            num_workers=int(getattr(config, "num_workers", 0)),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(Dataset([]), batch_size=1, shuffle=False, num_workers=0)
        return train_loader, unlabeled_loader, val_loader, val_t

    data_engine.build_dataloaders = build_cv_dataloaders
    pipeline_module.build_dataloaders = build_cv_dataloaders


def build_coord_network(backbone: str, num_classes: int, dropout: float):
    """Exact frozen DynUNet except in_channels=4."""
    if backbone != "dynunet":
        raise RuntimeError(f"C1 is locked to DynUNet, got backbone={backbone!r}")
    return DynUNet(
        spatial_dims=3,
        in_channels=INPUT_CHANNELS,
        out_channels=num_classes,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 4,
        filters=[16, 32, 64, 128, 256],
        dropout=dropout,
        norm_name="instance",
        deep_supervision=True,
    )


@torch.no_grad()
def evaluate_fold_c1(config, val_ids, checkpoint, source, threshold):
    """Final held-out evaluation with the same 4-channel deterministic input."""
    by_id = {c["id"]: c for c in cv.collect_cases(config)}
    base = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    transform = c1_val_transform(base)
    loader = DataLoader(
        Dataset([by_id[x] for x in sorted(val_ids)], transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    student = build_coord_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = build_coord_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()
    if source in ("teacher", "ensemble") and teacher is None:
        raise RuntimeError(f"Checkpoint has no EMA teacher; cannot evaluate source={source}")

    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].float().to(device)
        if int(image.shape[1]) != INPUT_CHANNELS:
            raise RuntimeError(f"Expected 4 input channels during evaluation, got {tuple(image.shape)}")
        case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            s_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
            if source == "student":
                prob = s_prob
            else:
                t_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                prob = t_prob if source == "teacher" else 0.5 * (s_prob + t_prob)

        pred = (prob > float(threshold)).float()
        row = cv.case_metrics(pred, target, cv.transformed_spacing(image, config))
        row["case_id"] = case_id
        rows.append(row)

    del student
    if teacher is not None:
        del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description="C1 coordinate-channel DynUNet screening on Final72 A3")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", default="screen", help="screen (=1+2), all, or 0..4")
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--a3-cv-dir", default=str(A3_CV))
    p.add_argument("--size-profile", default=str(SIZE_PROFILE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    selected_folds = spatial_screen.parse_screen_fold(args.fold)
    output_dir = Path(args.output_dir)
    source_cv_dir = Path(args.source_cv_dir)
    source_manifest_path = source_cv_dir / "cv_splits.json"
    a3_results_path = Path(args.a3_cv_dir) / "cv_results.csv"

    a3_rows = read_csv(a3_results_path)
    size_profile = read_csv(Path(args.size_profile))
    if len({str(r["case_id"]) for r in a3_rows}) != spatial_screen.EXPECTED_SOURCE:
        raise RuntimeError("A3 reference must contain the exact 47 frozen held-out cases")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("C1 screening requires prototype Student+EMA mode")
    if config.unet_backbone != "dynunet":
        raise RuntimeError(f"C1 is locked to dynunet, got {config.unet_backbone}")

    _, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    # Runner-local patches only. Existing trainer/checkpoints remain untouched.
    trainer_module.build_network = build_coord_network
    cv.build_network = build_coord_network
    cv.evaluate_fold = evaluate_fold_c1
    install_c1_loader_hook()

    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=42,
        resize_size=128,
        epochs=100,
        output_dir=str(output_dir),
        split_manifest=str(source_manifest_path),
        eval_source="ensemble",
        eval_threshold=0.50,
        overwrite=bool(args.overwrite),
        regenerate_splits=False,
        spatial_aug=True,
        translate_voxels=4.0,
        rotate_degrees=0.0,
        scale_fraction=0.0,
        baseline_results=str(a3_results_path),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "final72_c1_coord_channels_v1",
        "purpose": "test explicit absolute spatial/anatomical coordinates",
        "screening_folds": selected_folds,
        "reference": "Final72 A3",
        "source_manifest": str(source_manifest_path),
        "audit_metadata": str(args.audit_metadata),
        "n_total_human_gold": spatial_screen.EXPECTED_TOTAL,
        "n_train_only_extra": len(extra_ids),
        "input_channels": ["ultrasound", "coord_LR", "coord_AP", "coord_SI"],
        "coordinate_range": [-1.0, 1.0],
        "coordinates_added_after_spatial_augmentation": True,
        "recipe": {
            "architecture": "DynUNet",
            "resize_size": [128, 128, 128],
            "loss": "dice_ce",
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip": True,
            "lr_flip_probability": 0.5,
            "lr_flip_axis_after_ras": 0,
            "rotation": "off",
            "scale": "off",
            "epochs": 100,
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "postprocessing": "raw_no_lcc",
        },
        "external31_access": False,
    }
    plan_path = output_dir / "screening_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError(f"Existing C1 plan differs at {plan_path}; use a fresh output dir")
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 122)
    print("C1 — FINAL72 A3 + XYZ COORDINATE CHANNELS")
    print(f"Running folds:         {selected_folds}")
    print("Input:                 ultrasound + normalized LR/AP/SI coordinates = 4 channels")
    print("Coordinates:           fixed [-1,+1], added AFTER A3 spatial augmentation")
    print("A3 augmentation:       +/-4 vox p=.5 + LR flip p=.5")
    print("Model/loss/resolution: DynUNet / DiceCE / 128^3")
    print("Evaluation:            raw Student+EMA 50/50 @ .50")
    print("External31:            NOT ACCESSED")
    print("=" * 122)

    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    selected = set(selected_folds)
    kept = [r for r in existing if int(r["fold"]) not in selected]
    combined = kept + new_rows
    combined.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    cv.write_results(results_path, combined)

    compare_to_a3(
        combined,
        a3_rows,
        selected_folds,
        size_profile,
        output_dir,
        "C1_COORD",
    )

    print(f"\nResults: {results_path}")
    print(f"Plan:    {plan_path}")
    print(f"Summary: {output_dir / 'screening_vs_a3_summary.json'}")


if __name__ == "__main__":
    main()
