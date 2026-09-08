#!/usr/bin/env python3
"""Train the final Stage-2 DynUNet with an audited labeled-case set.

Training uses safe randomized GT-derived crops exactly as Stage-2 CV. The fixed epoch count is the
rounded median selected epoch from the five completed Stage-2 folds. Defaults preserve the all-90
QC-clean Final91 experiment; explicit flags permit larger audited cohorts with recorded case
exclusions. Any robust crop, sampling, and appearance settings are inherited from the selected CV
checkpoints rather than re-entered at final training. External31 is not accessed.
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


def selected_training_ids(audited_ids, include_quarantined, excluded_ids=()):
    excluded = set(str(x) for x in excluded_ids)
    if not include_quarantined:
        excluded.add(QUARANTINED_CASE_ID)
    return sorted(set(audited_ids) - excluded)


def dry_run(args):
    default_count = args.expected_live if args.include_quarantined else args.expected_live - 1
    training_count = args.expected_trainable or default_count
    print("=" * 116)
    print(f"FINAL{args.expected_live} FINAL STAGE-2 DYNUNET — ALL{training_count} DRY RUN")
    print(f"Prerequisite:         final all{training_count} CenterNet + complete Stage-2 OOF")
    print(f"Data:                 {args.expected_live} live labels -> {training_count} training cases")
    print(f"Prior quarantine:     {'INCLUDED by explicit experiment flag' if args.include_quarantined else 'EXCLUDED'}")
    print("Epoch selection:      rounded median Stage-2 CV selected epoch")
    print("Training data recipe: inherited exactly from the selected five-fold Stage-2 CV")
    print("Model:                DynUNet Student+EMA; DiceCE; threshold 0.50")
    print("External31:           NOT ACCESSED")
    print(f"Output:               {args.output_dir}")
    print("=" * 116)


def run(args):
    import torch
    from monai.data import CacheDataset, DataLoader
    from monai.utils import set_determinism

    from hassl.config import HASSLConfig
    import hassl.data.data_engine as data_engine
    from hassl.data.stage2_training import (
        add_training_recipe_extensions,
        build_small_bladder_sampler,
        build_stage2_training_transform,
        options_from_recipe,
    )
    from hassl.training.ema import EMATeacher
    from hassl.training.losses import CombinedSegLoss
    from hassl.training.trainer import build_network, compute_multiscale_loss
    from scripts.audit_round1_labels import discover_round1_cases

    def main_prediction(output):
        if isinstance(output, (list, tuple)):
            return output[0]
        if torch.is_tensor(output) and output.ndim == 6:
            return output[:, 0]
        return output

    audit = read_json(args.audit_metadata)
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    excluded_ids = sorted(str(x) for x in audit.get("excluded_training_case_ids", []))
    provenance_ok = bool(
        audit.get("selection_provenance_enforced", False)
        or audit.get("training_scope_provenance_enforced", False)
    )
    training_audit_ok = bool(
        audit.get(
            "all_training_labels_passed_audit",
            audit.get("all_visible_labels_passed_audit", False),
        )
    )
    if bool(audit.get("missing_frozen_allowed", False)) != bool(args.allow_missing_frozen):
        raise RuntimeError(
            "Training --allow-missing-frozen policy differs from the audit metadata"
        )
    if (
        not training_audit_ok
        or not provenance_ok
        or len(audited_ids) != int(args.expected_live)
        or QUARANTINED_CASE_ID not in audited_ids
    ):
        raise RuntimeError("Final91 live-label audit/quarantine provenance is not valid")
    unknown_exclusions = sorted(set(excluded_ids) - set(audited_ids))
    if unknown_exclusions:
        raise RuntimeError(
            "Audit exclusion list contains IDs outside the audited cohort: "
            + ", ".join(unknown_exclusions)
        )

    stage1_final = Path(args.stage1_final_dir)
    stage1_checkpoint = stage1_final / "final_centernet3d.pth"
    stage1_metadata = read_json(stage1_final / "final_centernet3d_metadata.json")
    expected_trainable = len(
        selected_training_ids(audited_ids, args.include_quarantined, excluded_ids)
    )
    if args.expected_trainable is not None and expected_trainable != int(args.expected_trainable):
        raise RuntimeError(
            f"Expected --expected-trainable {args.expected_trainable}, but audit selects "
            f"{expected_trainable} cases"
        )
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
    locked_optimization = {
        "learning_rate": float(first_recipe["learning_rate"]),
        "weight_decay": float(first_recipe["weight_decay"]),
        "ema_decay": float(first_recipe["ema_decay"]),
    }
    training_options = options_from_recipe(first_recipe)
    for _, state in cv_states[1:]:
        recipe = state["recipe"]
        current = {
            "learning_rate": float(recipe["learning_rate"]),
            "weight_decay": float(recipe["weight_decay"]),
            "ema_decay": float(recipe["ema_decay"]),
        }
        if current != locked_optimization or options_from_recipe(recipe) != training_options:
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
    train_ids = selected_training_ids(audited_ids, args.include_quarantined, excluded_ids)
    if len(train_ids) != expected_trainable:
        raise RuntimeError(
            f"Expected exactly {expected_trainable} training cases, found {len(train_ids)}"
        )
    if stage1_metadata.get("train_ids") != train_ids:
        raise RuntimeError("Final CenterNet and requested Stage-2 training IDs differ")
    non_training_ids = sorted(set(audited_ids) - set(train_ids))

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "final_stage2_dynunet.pth"
    metadata_path = output_dir / "final_stage2_metadata.json"
    if training_options.robust_enabled and output_dir.resolve() == OUTPUT.resolve():
        raise RuntimeError(
            "Robust Stage-2 training requires a fresh explicit --output-dir; "
            "the locked all90 output is protected"
        )
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
    train_transform = build_stage2_training_transform(base, training_options)
    workers = int(getattr(config, "num_workers", 0))
    train_items = [by_id[x] for x in train_ids]
    sampler, sampling_audit = build_small_bladder_sampler(
        train_items, training_options, seed=seed + 10000
    )
    dataset = CacheDataset(
        train_items, transform=train_transform, cache_rate=1.0,
        copy_cache=False, num_workers=workers,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=sampler is None, sampler=sampler, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final Stage-2 training requires CUDA")
    student = build_network("dynunet", 1, 0.0).to(device)
    ema = EMATeacher(student).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=locked_optimization["learning_rate"],
        weight_decay=locked_optimization["weight_decay"]
    )
    criterion = CombinedSegLoss(1, loss_type="dice_ce", include_boundary=False)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=" * 116)
    print(f"FINAL{args.expected_live} FINAL STAGE-2 DYNUNET — TRAIN ALL{len(train_ids)}")
    print(f"CV selected epochs:      {selected_epochs}")
    print(f"Median / final epochs:   {median_epoch} / {final_epochs}")
    print(f"Training cases:          {len(train_ids)}")
    print(f"Missing frozen allowed:  {bool(args.allow_missing_frozen)}")
    print(
        f"Crop margin / jitter:    {training_options.margin_min:.0%}-"
        f"{training_options.margin_max:.0%} / {training_options.center_jitter:.0%}"
    )
    if training_options.wide_crop_probability > 0.0:
        print(
            f"Wide crop:               p={training_options.wide_crop_probability:.0%} | "
            f"margin={training_options.wide_margin_min:.0%}-"
            f"{training_options.wide_margin_max:.0%} | "
            f"jitter={training_options.wide_center_jitter:.0%}"
        )
    if sampling_audit["enabled"]:
        print(
            f"Small oversampling:      n={sampling_audit['n_small_training']}/"
            f"{sampling_audit['n_training']} | weight="
            f"{training_options.small_bladder_weight:.2f}x | expected draws="
            f"{sampling_audit['expected_small_draw_fraction']:.0%}"
        )
    print(
        "Appearance augmentation:"
        + (
            " mild contrast/intensity/speckle"
            if training_options.mild_appearance_augmentation else " OFF"
        )
    )
    print("Prediction:              Student+EMA 50/50 @ 0.50")
    print("External31:              NOT ACCESSED")
    print("=" * 116)

    for epoch in range(1, final_epochs + 1):
        student.train()
        losses, dices, precisions, recalls = [], [], [], []
        wide_draws, gt_fractions = [], []
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
            ema.update(student, decay=locked_optimization["ema_decay"])
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
            wide_draws.append(float(batch["training_crop_is_wide"].float().mean().item()))
            gt_fractions.append(float(target.float().mean().item()))
        print(
            f"Epoch {epoch:03d}/{final_epochs} | TRAIN loss={np.mean(losses):.4f} "
            f"dice={np.mean(dices):.4f} prec={np.mean(precisions):.4f} "
            f"rec={np.mean(recalls):.4f} wide={np.mean(wide_draws):.2f} "
            f"gt_frac={np.mean(gt_fractions):.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_sampling.json").write_text(
        json.dumps(sampling_audit, indent=2), encoding="utf-8"
    )
    final_recipe = {
        "architecture": "DynUNet",
        "input": "single CenterNet-localized crop resized to 128^3",
        "loss": "DiceCE",
        "prediction": "Student+EMA 50/50 ensemble",
        "threshold": 0.50,
        "crop_size": [128, 128, 128],
        **locked_optimization,
        "margin_min": training_options.margin_min,
        "margin_max": training_options.margin_max,
        "center_jitter": training_options.center_jitter,
        "external31_access": False,
    }
    if training_options.robust_enabled:
        final_recipe.update({
            "version": "final_stage2_oversegmentation_robust_v1",
            "training_profile": "oversegmentation_robust_v1",
            "training_crop": {
                "source": "GT bounding box only",
                "margin_per_side_range": [
                    training_options.margin_min, training_options.margin_max
                ],
                "center_jitter_fraction_of_gt_size": training_options.center_jitter,
                "guarantee": "crop unioned with GT bounds so foreground is never cut",
            },
            "augmentation_after_crop": {
                "translation_voxels": 4.0,
                "translation_probability": 0.5,
                "lr_flip_probability": 0.5,
                "lr_flip_axis_after_ras": 0,
            },
        })
        final_recipe = add_training_recipe_extensions(final_recipe, training_options)
    torch.save({
        "net_A": student.state_dict(),
        "teacher": ema.state_dict(),
        "epoch": final_epochs,
        "train_ids": train_ids,
        "excluded_training_case_ids": excluded_ids,
        "quarantined_case_ids": non_training_ids,
        "intentionally_included_prior_quarantine_ids": (
            [QUARANTINED_CASE_ID] if args.include_quarantined else []
        ),
        "cv_selected_epochs": selected_epochs,
        "median_cv_selected_epoch": median_epoch,
        "training_sampling": sampling_audit,
        "recipe": final_recipe,
    }, checkpoint_path)
    metadata = {
        "version": (
            f"final{args.expected_live}_final_stage2_all{len(train_ids)}_robust_v1"
            if training_options.robust_enabled
            else f"final{args.expected_live}_final_stage2_all{len(train_ids)}_v1"
        ),
        "checkpoint": str(checkpoint_path),
        "paired_final_centernet": str(stage1_checkpoint),
        "n_live_labels": int(args.expected_live),
        "n_qc_trainable": len(train_ids),
        "train_ids": train_ids,
        "excluded_training_case_ids": excluded_ids,
        "quarantined_case_ids": non_training_ids,
        "intentionally_included_prior_quarantine_ids": (
            [QUARANTINED_CASE_ID] if args.include_quarantined else []
        ),
        "cv_checkpoints": [str(path) for path, _ in cv_states],
        "cv_selected_epochs": selected_epochs,
        "median_cv_selected_epoch": median_epoch,
        "final_epochs": final_epochs,
        "epoch_selection": "override" if args.epochs is not None else "rounded_median_cv",
        "prediction": "Student+EMA 50/50 raw ensemble @ 0.50",
        "training_profile": final_recipe.get("training_profile", "locked_legacy"),
        "training_sampling": sampling_audit,
        "training_recipe": final_recipe,
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
    parser.add_argument("--expected-trainable", type=int, default=None)
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
    if args.expected_trainable is not None and args.expected_trainable < 1:
        parser.error("--expected-trainable must be >=1")
    if args.dry_run:
        dry_run(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
