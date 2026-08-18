#!/usr/bin/env python3
"""Export leakage-safe EMA probability maps for the original 47 cases from existing Round-2 CV.

No new coarse-model training is performed. The controlled Round-2 CV already reused the
original 47 fold assignments and appended the 15 post-original HUMAN_GOLD labels to TRAIN ONLY.
Therefore each original-47 case can be inferred by the Round-2 EMA model for the fold in which
that case was held out.

Outputs are native-grid float32 MHA probability maps for the first 2-channel guided-refiner
feasibility experiment.
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
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("export_guided_refiner_existing_round2_oof47.py requires SimpleITK") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd
from hassl.config import HASSLConfig
from hassl.data.data_engine import get_base_transforms
from scripts.build_oof_qc_dataset import load_models
import scripts.train_supervised_cv as cv
from scripts.validate_external_threshold_31 import (
    binary_metrics,
    invert_probability_exact,
    normalize_native_probability,
    read_gt_binary,
)

DEFAULT_ROUND2_CV_DIR = Path("experiments/round2_cv_62_translation12")
DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/guided_refiner_existing_round2_oof47_probs_v1")
EXPECTED_OOF_CASES = 47


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    p = argparse.ArgumentParser(description="Export existing Round2 OOF47 EMA probabilities")
    p.add_argument("--config", required=True)
    p.add_argument("--round2-cv-dir", default=str(DEFAULT_ROUND2_CV_DIR))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cv_dir = Path(args.round2_cv_dir)
    plan_path = cv_dir / "round2_cv_plan.json"
    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    plan = read_json(plan_path)
    audit = read_json(audit_path)

    if int(plan.get("n_frozen_source_labels", -1)) != EXPECTED_OOF_CASES:
        raise RuntimeError("Round2 CV plan is not the controlled original-47 held-out design")
    if int(plan.get("n_total_human_labels", -1)) != 62:
        raise RuntimeError("Expected the existing controlled Final62 Round2 CV plan")
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round2 audit is not marked passing")

    folds = plan.get("folds", [])
    held_out = [str(x) for fold in folds for x in fold.get("val_ids", [])]
    if len(held_out) != EXPECTED_OOF_CASES or len(set(held_out)) != EXPECTED_OOF_CASES:
        raise RuntimeError("Round2 CV plan must hold out each original case exactly once")

    # Extra 15 must never appear in val_ids.
    extra_ids = set(str(x) for x in plan.get("round1_human_label_ids", [])) | set(
        str(x) for x in plan.get("round2_new_human_label_ids", [])
    )
    if extra_ids & set(held_out):
        raise RuntimeError("Post-original human label leaked into controlled held-out folds")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("Existing Round2 checkpoints require prototype student + EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    by_id = {str(c["id"]): c for c in cv.collect_cases(config)}
    missing = sorted(set(held_out) - set(by_id))
    if missing:
        raise RuntimeError(f"Original held-out cases missing from current dataset: {missing}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Output directory is not empty: {output_dir}; use --overwrite intentionally")
    prob_dir = output_dir / "probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"], transform=transform, orig_keys=["image"], nearest_interp=False, to_tensor=True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    print("=" * 116)
    print("GUIDED REFINER — REUSE EXISTING ROUND2 OOF47 EMA")
    print(f"Round2 CV:            {cv_dir}")
    print(f"True OOF cases:       {len(held_out)}")
    print(f"Post-original train-only labels: {len(extra_ids)}")
    print(f"Device:               {device}")
    print(f"Physical GPU request: {SELECTED_GPU if SELECTED_GPU is not None else '<environment/config>'}")
    print("No new coarse-model training is performed.")
    print("=" * 116)

    rows = []
    for fold_spec in folds:
        fold = int(fold_spec["fold"])
        checkpoint = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing existing Round2 checkpoint: {checkpoint}")

        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"Fold {fold} checkpoint has no EMA teacher")
        del student
        teacher.eval()

        val_ids = sorted(str(x) for x in fold_spec["val_ids"])
        items = [{"image": by_id[x]["image"], "id": x} for x in val_ids]
        loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
                case = by_id[case_id]
                image_t = batch["image"].to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    logits = cv.main_prediction(inferer(image_t, teacher))
                    prob_t = torch.sigmoid(logits)

                native_prob = invert_probability_exact(prob_t, batch, inverse_transform, index=0)
                source_image, prob_zyx = normalize_native_probability(native_prob, case["image"])
                gt = read_gt_binary(case["label"], source_image)
                metrics = binary_metrics(prob_zyx > float(args.threshold), gt)

                out_img = sitk.GetImageFromArray(np.asarray(prob_zyx, dtype=np.float32))
                out_img.CopyInformation(source_image)
                prob_path = prob_dir / f"{case_id}.ema_prob.mha"
                sitk.WriteImage(out_img, str(prob_path), useCompression=True)

                rows.append({
                    "case_id": case_id,
                    "fold": fold,
                    "checkpoint": str(checkpoint),
                    "image_path": str(case["image"]),
                    "label_path": str(case["label"]),
                    "probability_path": str(prob_path),
                    "oof_ema_dice_at_050": float(metrics["dice"]),
                    "oof_ema_precision_at_050": float(metrics["precision"]),
                    "oof_ema_recall_at_050": float(metrics["recall"]),
                    "oof_ema_signed_rve_pct_at_050": float(metrics["signed_rve_pct"]),
                })
                print(
                    f"fold={fold} {case_id} | OOF EMA Dice={metrics['dice']:.4f} | "
                    f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}"
                )

        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: str(r["case_id"]))
    if len(rows) != EXPECTED_OOF_CASES or set(str(r["case_id"]) for r in rows) != set(held_out):
        raise RuntimeError("OOF47 export did not produce exactly one probability map per original case")

    manifest_csv = output_dir / "existing_round2_oof47_ema_probability_manifest.csv"
    write_csv(manifest_csv, rows)
    dices = np.asarray([float(r["oof_ema_dice_at_050"]) for r in rows], dtype=float)
    metadata = {
        "version": "guided_refiner_existing_round2_oof47_ema_v1",
        "n_cases": len(rows),
        "source_round2_cv_dir": str(cv_dir),
        "source_round2_plan": str(plan_path),
        "source_audit": str(audit_path),
        "probability_maps_are_native_grid_float32": True,
        "mean_oof_ema_dice_at_050": float(np.mean(dices)),
        "median_oof_ema_dice_at_050": float(np.median(dices)),
        "dice_lt_070": int(np.sum(dices < 0.70)),
        "true_oof_case_ids": sorted(held_out),
        "excluded_from_guided_training_because_not_oof": sorted(extra_ids),
        "leakage_rule": (
            "Only original47 val_ids are exported. Each map comes from the existing Round2 EMA fold "
            "that held that original case out; the 15 post-original labels remain train-only."
        ),
    }
    metadata_path = output_dir / "existing_round2_oof47_ema_probability_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 116)
    print("EXISTING ROUND2 OOF47 EMA EXPORT COMPLETE")
    print(f"Cases:                 {len(rows)}")
    print(f"Mean OOF EMA Dice:     {np.mean(dices):.4f}")
    print(f"Median OOF EMA Dice:   {np.median(dices):.4f}")
    print(f"Dice < .70:            {int(np.sum(dices < 0.70))}/{len(rows)}")
    print("Held-out inference:    PASS — original47 only")
    print("Native-grid export:    PASS")
    print(f"Manifest:              {manifest_csv}")
    print(f"Metadata:              {metadata_path}")
    print("=" * 116)


if __name__ == "__main__":
    main()
