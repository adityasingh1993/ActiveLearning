#!/usr/bin/env python3
"""Train the final 3D CenterNet bladder localizer with an audited labeled-case set.

The epoch count is the rounded median selected epoch from the five completed Stage-1 CV
checkpoints. Defaults preserve the all-90 QC-clean Final91 experiment; explicit flags permit the
separate all-136 experiment. External31 is not accessed.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT = Path("experiments/round5_supervised_91_a3/final91_live_label_audit.json")
SOURCE_CV = Path("experiments/cv5_supervised_47_translation12")
STAGE1_CV = Path("experiments/final91_qc_stage1_centernet3d_cv")
STAGE2_CV = Path("experiments/final91_qc_stage2_centernet_dynunet_cv")
OUTPUT = Path("experiments/final91_qc_two_stage_all90/stage1")
EXPECTED_LIVE = 91
EXPECTED_TRAINABLE = 90
QUARANTINED_CASE_ID = (
    "9435b1b67a41b88f6084a3e750fc54d913213ea55f33d165a1f42b9b50dd237c"
)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def selected_training_ids(audited_ids, include_quarantined):
    if include_quarantined:
        return sorted(audited_ids)
    return sorted(set(audited_ids) - {QUARANTINED_CASE_ID})


def dry_run(args):
    training_count = args.expected_live if args.include_quarantined else args.expected_live - 1
    print("=" * 116)
    print(f"FINAL{args.expected_live} FINAL CENTERNET — ALL{training_count} DRY RUN")
    print("Gate:                 complete Stage-1 + Stage-2 five-fold OOF required")
    print(f"Data:                 {args.expected_live} live labels -> {training_count} training cases")
    print(f"Prior quarantine:     {'INCLUDED by explicit experiment flag' if args.include_quarantined else 'EXCLUDED'}")
    print("Epoch selection:      rounded median Stage-1 CV selected epoch")
    print("Architecture:         exact 3D CenterNet used in Stage-1 CV")
    print("Input:                full 128^3 image")
    print("External31:           NOT ACCESSED")
    print(f"Output:               {args.output_dir}")
    print("=" * 116)


def run(args):
    import torch
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import Compose, MapTransform, RandAffined, RandFlipd
    from monai.utils import set_determinism

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from hassl.models.centernet3d import CenterNet3D, box_iou, centernet_loss, decode_peak
    from scripts.audit_round1_labels import discover_round1_cases

    class BoundingBoxTargetd(MapTransform):
        def __init__(self):
            super().__init__(["label"])

        def __call__(self, data):
            result = dict(data)
            label = (result["label"] > 0.5).to(dtype=result["label"].dtype)
            spatial = label[0] if label.ndim == 4 else label
            coords = torch.nonzero(spatial > 0, as_tuple=False)
            if coords.numel() == 0:
                raise RuntimeError(f"Empty bladder after preprocessing: {result.get('id', '?')}")
            lo, hi = coords.min(dim=0).values.float(), coords.max(dim=0).values.float()
            shape = torch.as_tensor(spatial.shape, dtype=torch.float32, device=lo.device)
            result["bbox_center"] = 0.5 * (lo + hi) / torch.clamp(shape - 1.0, min=1.0)
            result["bbox_size"] = (hi - lo + 1.0) / shape
            return result

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    provenance_ok = bool(
        audit.get("selection_provenance_enforced", False)
        or audit.get("training_scope_provenance_enforced", False)
    )
    if (
        not audit.get("all_visible_labels_passed_audit", False)
        or not provenance_ok
        or len(audited_ids) != int(args.expected_live)
        or QUARANTINED_CASE_ID not in audited_ids
    ):
        raise RuntimeError("Final91 live-label audit/quarantine provenance is not valid")

    stage1_cv = Path(args.stage1_cv_dir)
    stage1_gate = read_json(stage1_cv / "centernet3d_gate_summary.json")
    if (
        not stage1_gate.get("complete_original46_qc_oof", False)
        or not stage1_gate.get("gate_pass", False)
        or stage1_gate.get("quarantined_case_ids") != [QUARANTINED_CASE_ID]
    ):
        raise RuntimeError("Complete passing Stage-1 original46-QC gate is required")
    stage2_summary = read_json(Path(args.stage2_cv_dir) / "stage2_vs_fullvolume_summary.json")
    if not stage2_summary.get("complete_original46_qc", False):
        raise RuntimeError("Complete Stage-2 original46-QC OOF comparison is required")

    cv_states, selected_epochs = [], []
    for fold in range(5):
        checkpoint = stage1_cv / "checkpoints" / f"fold_{fold}" / "best_centernet3d.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(state.get("fold", -1)) != fold:
            raise RuntimeError(f"Stage-1 fold {fold} checkpoint provenance differs")
        recipe = state.get("recipe", {})
        if (
            recipe.get("architecture") != "centernet3d_fpn"
            or recipe.get("resize_size") != [128, 128, 128]
            or abs(float(recipe.get("safety_margin_per_side", -1)) - 0.50) > 1e-8
            or recipe.get("external31_access") is not False
        ):
            raise RuntimeError(f"Stage-1 fold {fold} recipe differs from the locked definition")
        selected_epochs.append(int(state["epoch"]))
        cv_states.append((checkpoint, state))
    median_epoch = int(round(float(np.median(selected_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_epoch
    if final_epochs < 1:
        raise RuntimeError("Final epoch count must be >=1")
    first_recipe = cv_states[0][1]["recipe"]
    learning_rate = float(first_recipe["learning_rate"])
    weight_decay = float(first_recipe["weight_decay"])
    for _, state in cv_states[1:]:
        recipe = state["recipe"]
        if (
            float(recipe["learning_rate"]) != learning_rate
            or float(recipe["weight_decay"]) != weight_decay
        ):
            raise RuntimeError("Stage-1 CV optimization recipes differ across folds")

    config = HASSLConfig.from_yaml(args.config)
    config.preprocessing_mode = "resize"
    config.spatial_size = (128, 128, 128)
    source_manifest = Path(args.source_cv_dir) / "cv_splits.json"
    _, source_ids, by_id, _ = discover_round1_cases(config, source_manifest)
    if len(source_ids) != 47 or sorted(by_id) != audited_ids:
        raise RuntimeError("Live labels or frozen original47 source changed after audit")
    train_ids = selected_training_ids(audited_ids, args.include_quarantined)
    expected_trainable = int(args.expected_live) if args.include_quarantined else int(args.expected_live) - 1
    if len(train_ids) != expected_trainable:
        raise RuntimeError(
            f"Expected exactly {expected_trainable} training cases, found {len(train_ids)}"
        )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "final_centernet3d.pth"
    metadata_path = output_dir / "final_centernet3d_metadata.json"
    if checkpoint_path.exists() and not args.overwrite:
        print(f"Final CenterNet already exists: {checkpoint_path}")
        print("Use --overwrite only if intentionally retraining the same locked recipe.")
        return

    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    base = data_engine.get_base_transforms(
        config, keys=["image", "label"], is_training=True, apply_strong_aug=False
    )
    steps = list(getattr(base, "transforms", [base]))
    steps.extend([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.0), translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0), mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        BoundingBoxTargetd(),
    ])
    workers = int(getattr(config, "num_workers", 0))
    dataset = CacheDataset(
        [by_id[x] for x in train_ids], transform=Compose(steps), cache_rate=1.0,
        copy_cache=False, num_workers=workers,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final 3D CenterNet training requires CUDA")
    model = CenterNet3D().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=final_epochs, eta_min=learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=" * 116)
    print(f"FINAL{args.expected_live} FINAL 3D CENTERNET — TRAIN ALL{len(train_ids)}")
    print(f"CV selected epochs:      {selected_epochs}")
    print(f"Median / final epochs:   {median_epoch} / {final_epochs}")
    print(f"Training cases:          {len(train_ids)}")
    print(f"Prior quarantine:        {'INCLUDED' if args.include_quarantined else 'EXCLUDED'}: "
          f"{QUARANTINED_CASE_ID}")
    print(f"LR / weight decay:       {learning_rate:g} / {weight_decay:g}")
    print("External31:              NOT ACCESSED")
    print("=" * 116)

    for epoch in range(1, final_epochs + 1):
        model.train()
        losses, ious, confidences = [], [], []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target_center = batch["bbox_center"].float().to(device)
            target_size = batch["bbox_size"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=True):
                outputs = model(image)
                total, _, _, _, _ = centernet_loss(outputs, target_center, target_size)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                center, size, confidence, _ = decode_peak(outputs)
                iou = box_iou(center, size, target_center, target_size).mean()
            losses.append(float(total.item()))
            ious.append(float(iou.item()))
            confidences.append(float(confidence.mean().item()))
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{final_epochs} | TRAIN loss={np.mean(losses):.4f} "
            f"box_iou={np.mean(ious):.4f} center_conf={np.mean(confidences):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.7f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": final_epochs,
        "train_ids": train_ids,
        "quarantined_case_ids": [] if args.include_quarantined else [QUARANTINED_CASE_ID],
        "intentionally_included_prior_quarantine_ids": (
            [QUARANTINED_CASE_ID] if args.include_quarantined else []
        ),
        "cv_selected_epochs": selected_epochs,
        "median_cv_selected_epoch": median_epoch,
        "recipe": {
            "architecture": "centernet3d_fpn",
            "resize_size": [128, 128, 128],
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip_probability": 0.5,
            "safety_margin_per_side": 0.50,
            "external31_access": False,
        },
    }, checkpoint_path)
    metadata = {
        "version": (
            f"final{args.expected_live}_final_centernet_all{len(train_ids)}_v1"
        ),
        "checkpoint": str(checkpoint_path),
        "n_live_labels": int(args.expected_live),
        "n_qc_trainable": len(train_ids),
        "train_ids": train_ids,
        "quarantined_case_ids": [] if args.include_quarantined else [QUARANTINED_CASE_ID],
        "intentionally_included_prior_quarantine_ids": (
            [QUARANTINED_CASE_ID] if args.include_quarantined else []
        ),
        "cv_checkpoints": [str(path) for path, _ in cv_states],
        "cv_selected_epochs": selected_epochs,
        "median_cv_selected_epoch": median_epoch,
        "final_epochs": final_epochs,
        "epoch_selection": "override" if args.epochs is not None else "rounded_median_cv",
        "external31_access": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFINAL CENTERNET COMPLETE")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata:   {metadata_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train final Final91-QC CenterNet on all90")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--stage1-cv-dir", default=str(STAGE1_CV))
    parser.add_argument("--stage2-cv-dir", default=str(STAGE2_CV))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--expected-live", type=int, default=EXPECTED_LIVE)
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="Intentionally include the historically quarantined 9435... case in training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >=1")
    if args.expected_live < 2:
        parser.error("--expected-live must be >=2")
    if args.dry_run:
        dry_run(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
