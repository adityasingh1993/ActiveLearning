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
import copy
import csv
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

SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
AUDIT = Path("experiments/round3_supervised_72_translation12/round3_label_audit.json")
A3_CV = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")
SIZE_PROFILE = Path("experiments/final72_bladder_size_diagnostic/all72_bladder_size_profile.csv")
DEFAULT_OUTPUT = Path("experiments/final72_screen_c1_coord_channels_a3")
INPUT_CHANNELS = 4


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


class AddCoordinateChannelsd:
    """Append fixed normalized LR/AP/SI coordinate channels to ``image``.

    The transform must be placed after all random spatial augmentation. The first spatial axis
    is treated as LR because the upstream deterministic transform has already oriented the
    volume to RAS, matching the A3 LR-flip convention used in this experiment series.

    MONAI transform history is copied from the one-channel MetaTensor. This is important because
    HASSL validation uses Invertd to restore predictions to native geometry for volume metrics.
    Adding coordinates is intentionally not part of that inverse trace; only the underlying
    spatial preprocessing/augmentation history is preserved.
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
                meta=copy.deepcopy(image.meta),
                applied_operations=copy.deepcopy(image.applied_operations),
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


def preflight_coordinate_input(config, fold_spec):
    """Fail before training if the coordinate input contract is not exactly as intended."""
    by_id = {c["id"]: c for c in cv.collect_cases(config)}
    case_id = sorted(fold_spec["val_ids"])[0]
    base = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    sample = c1_val_transform(base)(by_id[case_id])
    image = sample["image"]
    if tuple(image.shape) != (4, 128, 128, 128):
        raise RuntimeError(f"C1 preflight expected [4,128,128,128], got {tuple(image.shape)}")
    coord = image[1:4].as_tensor() if isinstance(image, MetaTensor) else image[1:4]
    mins = [float(coord[i].min().item()) for i in range(3)]
    maxs = [float(coord[i].max().item()) for i in range(3)]
    if not all(abs(x + 1.0) < 1e-5 for x in mins) or not all(abs(x - 1.0) < 1e-5 for x in maxs):
        raise RuntimeError(f"C1 coordinate range failed: mins={mins}, maxs={maxs}")
    print(
        f"C1 preflight PASS | case={case_id} | shape={tuple(image.shape)} | "
        f"coord mins={mins} | maxs={maxs}"
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


def metric_mean(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def compare_to_a3_c1(candidate_rows, a3_rows, selected_folds, size_profile, output_dir):
    selected = set(int(x) for x in selected_folds)
    cand = {str(r["case_id"]): r for r in candidate_rows if int(r["fold"]) in selected}
    base = {str(r["case_id"]): r for r in a3_rows if int(r["fold"]) in selected}
    if set(cand) != set(base):
        raise RuntimeError("C1 and A3 must contain exact same held-out IDs for selected folds")

    size_by_id = {str(r["case_id"]): str(r["size_group"]) for r in size_profile}
    paired = []
    for case_id in sorted(cand):
        a = base[case_id]
        b = cand[case_id]
        if case_id not in size_by_id:
            raise RuntimeError(f"Missing fixed size group for {case_id}")
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "size_group": size_by_id[case_id],
            "a3_dice": float(a["dice"]),
            "c1_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a3_precision": float(a["precision"]),
            "c1_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a3_recall": float(a["recall"]),
            "c1_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a3_hd95": float(a["hd95"]),
            "c1_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
            "a3_rve": float(a["rve"]),
            "c1_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
        })
    write_csv(output_dir / "screening_vs_a3_case_comparison.csv", paired)

    def block(rows):
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "a3_mean_dice": metric_mean(rows, "a3_dice"),
            "c1_mean_dice": metric_mean(rows, "c1_dice"),
            "delta_mean_dice": metric_mean(rows, "delta_dice"),
            "a3_mean_precision": metric_mean(rows, "a3_precision"),
            "c1_mean_precision": metric_mean(rows, "c1_precision"),
            "delta_mean_precision": metric_mean(rows, "delta_precision"),
            "a3_mean_recall": metric_mean(rows, "a3_recall"),
            "c1_mean_recall": metric_mean(rows, "c1_recall"),
            "delta_mean_recall": metric_mean(rows, "delta_recall"),
            "a3_mean_hd95": metric_mean(rows, "a3_hd95"),
            "c1_mean_hd95": metric_mean(rows, "c1_hd95"),
            "improved": int(sum(r["delta_dice"] > 1e-6 for r in rows)),
            "worsened": int(sum(r["delta_dice"] < -1e-6 for r in rows)),
            "improved_ge_0p05": int(sum(r["delta_dice"] >= 0.05 for r in rows)),
            "worsened_le_minus_0p05": int(sum(r["delta_dice"] <= -0.05 for r in rows)),
        }

    folds = {str(f): block([r for r in paired if int(r["fold"]) == f]) for f in selected_folds}
    groups = {g: block([r for r in paired if r["size_group"] == g]) for g in ("SMALL", "MEDIUM", "LARGE")}
    overall = block(paired)
    summary = {
        "version": "final72_c1_coord_channels_screen_v1",
        "reference": "A3 = +/-4 vox p=.5 + LR flip p=.5 + DiceCE + 128^3",
        "candidate": "C1 = A3 + normalized LR/AP/SI coordinate channels",
        "screening_folds": list(selected_folds),
        "overall": overall,
        "folds": folds,
        "fixed_validation_size_groups": groups,
        "screening_only": True,
    }
    (output_dir / "screening_vs_a3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 122)
    print("C1_COORD vs A3 — COORDINATE-CHANNEL SCREEN")
    for f in selected_folds:
        s = folds[str(f)]
        print(
            f"Fold {f}: A3={s['a3_mean_dice']:.4f} -> C1={s['c1_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | PrecDelta={s['delta_mean_precision']:+.4f} | "
            f"RecDelta={s['delta_mean_recall']:+.4f}"
        )
    print(
        f"Combined: A3={overall['a3_mean_dice']:.4f} -> C1={overall['c1_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f}) | PrecDelta={overall['delta_mean_precision']:+.4f} | "
        f"RecDelta={overall['delta_mean_recall']:+.4f}"
    )
    print("\nFIXED VALIDATION SIZE GROUPS (diagnostic only)")
    for g in ("SMALL", "MEDIUM", "LARGE"):
        s = groups[g]
        if not s["n"]:
            continue
        print(
            f"  {g}: n={s['n']} | Dice {s['a3_mean_dice']:.4f}->{s['c1_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | Prec {s['a3_mean_precision']:.4f}->{s['c1_mean_precision']:.4f} "
            f"({s['delta_mean_precision']:+.4f}) | Rec {s['a3_mean_recall']:.4f}->{s['c1_mean_recall']:.4f} "
            f"({s['delta_mean_recall']:+.4f}) | HD95 {s['a3_mean_hd95']:.2f}->{s['c1_mean_hd95']:.2f}mm"
        )
    print(
        f"Case effects: improved={overall['improved']} | worsened={overall['worsened']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: full frozen CV is required before promotion.")
    print("=" * 122)


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

    # Match the actual run's deterministic settings before validating the 4-channel contract.
    cv.apply_baseline(config, resize_size=128, epochs=100)
    preflight_coordinate_input(config, fold_map[selected_folds[0]])

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

    compare_to_a3_c1(combined, a3_rows, selected_folds, size_profile, output_dir)

    print(f"\nResults: {results_path}")
    print(f"Plan:    {plan_path}")
    print(f"Summary: {output_dir / 'screening_vs_a3_summary.json'}")


if __name__ == "__main__":
    main()
