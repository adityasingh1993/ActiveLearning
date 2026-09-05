#!/usr/bin/env python3
"""Train the final Stage-2 DynUNet with an audited labeled-case set.

Training uses safe randomized GT-derived crops exactly as Stage-2 CV. The fixed epoch count is the
rounded median selected epoch from the five completed Stage-2 folds. Defaults preserve the all-90
QC-clean Final91 experiment; explicit flags permit the separate all-136 experiment. External31 is
not accessed.
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
STAGE1_FINAL = Path("experiments/final91_qc_two_stage_all90/stage1")
STAGE2_CV = Path("experiments/final91_qc_stage2_centernet_dynunet_cv")
OUTPUT = Path("experiments/final91_qc_two_stage_all90/stage2")
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
    print(f"FINAL{args.expected_live} FINAL STAGE-2 DYNUNET — ALL{training_count} DRY RUN")
    print(f"Prerequisite:         final all{training_count} CenterNet + complete Stage-2 OOF")
    print(f"Data:                 {args.expected_live} live labels -> {training_count} training cases")
    print(f"Prior quarantine:     {'INCLUDED by explicit experiment flag' if args.include_quarantined else 'EXCLUDED'}")
    print("Epoch selection:      rounded median Stage-2 CV selected epoch")
    print("Training crop:        GT box + 40%-60% safe margin + 10% center jitter")
    print("Model:                DynUNet Student+EMA; DiceCE; threshold 0.50")
    print("External31:           NOT ACCESSED")
    print(f"Output:               {args.output_dir}")
    print("=" * 116)


def run(args):
    import torch
    import torch.nn.functional as F
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import Compose, RandAffined, RandFlipd, RandomizableTransform
    from monai.utils import set_determinism

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from hassl.training.ema import EMATeacher
    from hassl.training.losses import CombinedSegLoss
    from hassl.training.trainer import build_network, compute_multiscale_loss
    from scripts.audit_round1_labels import discover_round1_cases

    def plain_tensor(value):
        if hasattr(value, "as_tensor"):
            return value.as_tensor()
        return torch.as_tensor(value)

    def crop_and_resize(image, label, bounds, output_size):
        z0, z1, y0, y1, x0, x1 = [int(x) for x in bounds]
        image_tensor = plain_tensor(image).float()
        label_tensor = plain_tensor(label).float()
        image_crop = image_tensor[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
        label_crop = label_tensor[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
        image_out = F.interpolate(
            image_crop, size=(output_size,) * 3, mode="trilinear", align_corners=False
        )[0]
        label_out = F.interpolate(label_crop, size=(output_size,) * 3, mode="nearest")[0]
        return image_out, label_out

    class SafeGTBoxCropd(RandomizableTransform):
        def __init__(self, output_size, margin_min, margin_max, center_jitter):
            super().__init__(prob=1.0)
            self.output_size = int(output_size)
            self.margin_min = float(margin_min)
            self.margin_max = float(margin_max)
            self.center_jitter = float(center_jitter)

        def __call__(self, data):
            result = dict(data)
            self.randomize(None)
            label = plain_tensor(result["label"])
            spatial = label[0] if label.ndim == 4 else label
            coords = torch.nonzero(spatial > 0.5, as_tuple=False)
            if coords.numel() == 0:
                raise RuntimeError(f"Empty Stage-2 training GT: {result.get('id', '?')}")
            gt_lo = coords.min(dim=0).values.cpu().numpy().astype(float)
            gt_hi = coords.max(dim=0).values.cpu().numpy().astype(float)
            gt_size = gt_hi - gt_lo + 1.0
            gt_center = 0.5 * (gt_lo + gt_hi)
            margin = float(self.R.uniform(self.margin_min, self.margin_max))
            shift = self.R.uniform(
                low=-self.center_jitter, high=self.center_jitter, size=3
            ) * gt_size
            proposed_center = gt_center + shift
            proposed_size = gt_size * (1.0 + 2.0 * margin)
            lo = np.floor(proposed_center - 0.5 * proposed_size).astype(int)
            hi = np.ceil(proposed_center + 0.5 * proposed_size).astype(int)
            lo = np.minimum(lo, gt_lo.astype(int))
            hi = np.maximum(hi, gt_hi.astype(int))
            shape = np.asarray(spatial.shape, dtype=int)
            lo, hi = np.maximum(lo, 0), np.minimum(hi, shape - 1)
            bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            result["image"], result["label"] = crop_and_resize(
                result["image"], result["label"], bounds, self.output_size
            )
            return result

    def main_prediction(output):
        if isinstance(output, (list, tuple)):
            return output[0]
        if torch.is_tensor(output) and output.ndim == 6:
            return output[:, 0]
        return output

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    provenance_ok = bool(
        audit.get("selection_provenance_enforced", False)
        or audit.get("training_scope_provenance_enforced", False)
    )
    if bool(audit.get("missing_frozen_allowed", False)) != bool(args.allow_missing_frozen):
        raise RuntimeError(
            "Training --allow-missing-frozen policy differs from the audit metadata"
        )
    if (
        not audit.get("all_visible_labels_passed_audit", False)
        or not provenance_ok
        or len(audited_ids) != int(args.expected_live)
        or QUARANTINED_CASE_ID not in audited_ids
    ):
        raise RuntimeError("Final91 live-label audit/quarantine provenance is not valid")

    stage1_final = Path(args.stage1_final_dir)
    stage1_checkpoint = stage1_final / "final_centernet3d.pth"
    stage1_metadata = read_json(stage1_final / "final_centernet3d_metadata.json")
    expected_trainable = int(args.expected_live) if args.include_quarantined else int(args.expected_live) - 1
    if (
        not stage1_checkpoint.exists()
        or int(stage1_metadata.get("n_qc_trainable", -1)) != expected_trainable
    ):
        raise RuntimeError(
            f"Final all{expected_trainable} CenterNet must be completed before final Stage 2"
        )
    if stage1_metadata.get("external31_access") is not False:
        raise RuntimeError("Final CenterNet metadata indicates external-data access")

    stage2_cv = Path(args.stage2_cv_dir)
    summary = read_json(stage2_cv / "stage2_vs_fullvolume_summary.json")
    if not summary.get("complete_original46_qc", False):
        raise RuntimeError("Complete Stage-2 original46-QC OOF comparison is required")
    cv_states, selected_epochs = [], []
    for fold in range(5):
        checkpoint = stage2_cv / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        marker = stage2_cv / "checkpoints" / f"fold_{fold}" / "training_complete.json"
        if not marker.exists():
            raise FileNotFoundError(marker)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(state.get("fold", -1)) != fold:
            raise RuntimeError(f"Stage-2 fold {fold} checkpoint provenance differs")
        recipe = state.get("recipe", {})
        if (
            recipe.get("architecture") != "DynUNet"
            or recipe.get("prediction") != "Student+EMA 50/50 ensemble"
            or recipe.get("crop_size") != [128, 128, 128]
            or abs(float(recipe.get("threshold", -1)) - 0.50) > 1e-8
            or recipe.get("external31_access") is not False
        ):
            raise RuntimeError(f"Stage-2 fold {fold} recipe differs from the locked definition")
        selected_epochs.append(int(state["epoch"]))
        cv_states.append((checkpoint, state))
    median_epoch = int(round(float(np.median(selected_epochs))))
    final_epochs = int(args.epochs) if args.epochs is not None else median_epoch
    if final_epochs < 1:
        raise RuntimeError("Final epoch count must be >=1")
    first_recipe = cv_states[0][1]["recipe"]
    locked = {
        "learning_rate": float(first_recipe["learning_rate"]),
        "weight_decay": float(first_recipe["weight_decay"]),
        "ema_decay": float(first_recipe["ema_decay"]),
        "margin_min": float(first_recipe["training_crop"]["margin_per_side_range"][0]),
        "margin_max": float(first_recipe["training_crop"]["margin_per_side_range"][1]),
        "center_jitter": float(first_recipe["training_crop"]["center_jitter_fraction_of_gt_size"]),
    }
    for _, state in cv_states[1:]:
        recipe = state["recipe"]
        current = {
            "learning_rate": float(recipe["learning_rate"]),
            "weight_decay": float(recipe["weight_decay"]),
            "ema_decay": float(recipe["ema_decay"]),
            "margin_min": float(recipe["training_crop"]["margin_per_side_range"][0]),
            "margin_max": float(recipe["training_crop"]["margin_per_side_range"][1]),
            "center_jitter": float(recipe["training_crop"]["center_jitter_fraction_of_gt_size"]),
        }
        if current != locked:
            raise RuntimeError("Stage-2 CV recipes differ across folds")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Final Stage 2 requires prototype DynUNet")
    config.preprocessing_mode = "resize"
    config.spatial_size = (128, 128, 128)
    config.loss_type = "dice_ce"
    config.include_boundary = False
    config.dropout = 0.0
    source_manifest = Path(args.source_cv_dir) / "cv_splits.json"
    _, source_ids, by_id, _ = discover_round1_cases(
        config,
        source_manifest,
        require_all_frozen=not args.allow_missing_frozen,
    )
    if len(source_ids) != 47 or sorted(by_id) != audited_ids:
        raise RuntimeError("Live labels or frozen original47 source changed after audit")
    train_ids = selected_training_ids(audited_ids, args.include_quarantined)
    if len(train_ids) != expected_trainable:
        raise RuntimeError(
            f"Expected exactly {expected_trainable} training cases, found {len(train_ids)}"
        )
    if stage1_metadata.get("train_ids") != train_ids:
        raise RuntimeError("Final CenterNet and requested Stage-2 training IDs differ")

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "final_stage2_dynunet.pth"
    metadata_path = output_dir / "final_stage2_metadata.json"
    if checkpoint_path.exists() and not args.overwrite:
        print(f"Final Stage-2 checkpoint already exists: {checkpoint_path}")
        print("Use --overwrite only if intentionally retraining the same locked recipe.")
        return

    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    base = data_engine.get_base_transforms(
        config, keys=["image", "label"], is_training=False, apply_strong_aug=False
    )
    steps = list(getattr(base, "transforms", [base]))
    steps.extend([
        SafeGTBoxCropd(128, locked["margin_min"], locked["margin_max"], locked["center_jitter"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.0), translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0), mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
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
        raise RuntimeError("Final Stage-2 training requires CUDA")
    student = build_network("dynunet", 1, 0.0).to(device)
    ema = EMATeacher(student).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=locked["learning_rate"], weight_decay=locked["weight_decay"]
    )
    criterion = CombinedSegLoss(1, loss_type="dice_ce", include_boundary=False)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=" * 116)
    print(f"FINAL{args.expected_live} FINAL STAGE-2 DYNUNET — TRAIN ALL{len(train_ids)}")
    print(f"CV selected epochs:      {selected_epochs}")
    print(f"Median / final epochs:   {median_epoch} / {final_epochs}")
    print(f"Training cases:          {len(train_ids)}")
    print(f"Missing frozen allowed:  {bool(args.allow_missing_frozen)}")
    print(f"Crop margin / jitter:    {locked['margin_min']:.0%}-{locked['margin_max']:.0%} / "
          f"{locked['center_jitter']:.0%}")
    print("Prediction:              Student+EMA 50/50 @ 0.50")
    print("External31:              NOT ACCESSED")
    print("=" * 116)

    for epoch in range(1, final_epochs + 1):
        student.train()
        losses, dices, precisions, recalls = [], [], [], []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=True):
                logits = student(image)
                loss = compute_multiscale_loss(criterion, logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(student, decay=locked["ema_decay"])
            with torch.no_grad():
                prediction = (torch.sigmoid(main_prediction(logits)) > 0.50).float()
                truth = (target > 0.5).float()
                tp = float((prediction * truth).sum().item())
                pred_sum = float(prediction.sum().item())
                gt_sum = float(truth.sum().item())
                dices.append((2.0 * tp) / (pred_sum + gt_sum + 1e-8))
                precisions.append(tp / (pred_sum + 1e-8))
                recalls.append(tp / (gt_sum + 1e-8))
            losses.append(float(loss.item()))
        print(
            f"Epoch {epoch:03d}/{final_epochs} | TRAIN loss={np.mean(losses):.4f} "
            f"dice={np.mean(dices):.4f} prec={np.mean(precisions):.4f} "
            f"rec={np.mean(recalls):.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "net_A": student.state_dict(),
        "teacher": ema.state_dict(),
        "epoch": final_epochs,
        "train_ids": train_ids,
        "quarantined_case_ids": [] if args.include_quarantined else [QUARANTINED_CASE_ID],
        "intentionally_included_prior_quarantine_ids": (
            [QUARANTINED_CASE_ID] if args.include_quarantined else []
        ),
        "cv_selected_epochs": selected_epochs,
        "median_cv_selected_epoch": median_epoch,
        "recipe": {
            "architecture": "DynUNet",
            "input": "single CenterNet-localized crop resized to 128^3",
            "loss": "DiceCE",
            "prediction": "Student+EMA 50/50 ensemble",
            "threshold": 0.50,
            "crop_size": [128, 128, 128],
            **locked,
            "external31_access": False,
        },
    }, checkpoint_path)
    metadata = {
        "version": f"final{args.expected_live}_final_stage2_all{len(train_ids)}_v1",
        "checkpoint": str(checkpoint_path),
        "paired_final_centernet": str(stage1_checkpoint),
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
        "prediction": "Student+EMA 50/50 raw ensemble @ 0.50",
        "external31_access": False,
        "missing_frozen_case_ids": audit.get("missing_frozen_case_ids", []),
        "missing_frozen_allowed": bool(args.allow_missing_frozen),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFINAL STAGE-2 COMPLETE")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata:   {metadata_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train final Final91-QC Stage 2 on all90")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-metadata", default=str(AUDIT))
    parser.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    parser.add_argument("--stage1-final-dir", default=str(STAGE1_FINAL))
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
    parser.add_argument(
        "--allow-missing-frozen",
        action="store_true",
        help="Train the audited current cohort even if original47 cases are absent.",
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
