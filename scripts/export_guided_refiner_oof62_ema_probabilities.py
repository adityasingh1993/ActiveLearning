#!/usr/bin/env python3
"""Export leakage-safe native-grid EMA probability maps for all 62 HUMAN_GOLD cases.

Each case is inferred only by the fold model for which that case was held out. The output
probability map is inverted back to the exact native image grid and stored as float32 MHA.
These maps are intended as the second input channel for the offline guided ROI refiner.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("export_guided_refiner_oof62_ema_probabilities.py requires SimpleITK") from exc

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

DEFAULT_CV_DIR = Path("experiments/guided_refiner_oof62_coarse_v1")
DEFAULT_AUDIT = Path("experiments/round2_supervised_62_translation12/round2_label_audit.json")
DEFAULT_OUTPUT_DIR = Path("experiments/guided_refiner_oof62_probabilities_v1")


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
    p = argparse.ArgumentParser(description="Export all-62 OOF EMA native probability maps")
    p.add_argument("--config", required=True)
    p.add_argument("--cv-dir", default=str(DEFAULT_CV_DIR))
    p.add_argument("--audit-metadata", default=str(DEFAULT_AUDIT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cv_dir = Path(args.cv_dir)
    audit_path = Path(args.audit_metadata)
    output_dir = Path(args.output_dir)
    manifest_path = cv_dir / "cv_splits.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    audit = read_json(audit_path)
    if not audit.get("all_visible_labels_passed_audit", False):
        raise RuntimeError("Round-2 audit is not marked passing")
    audited_ids = sorted(str(x) for x in audit.get("all_current_human_label_ids", []))
    if len(audited_ids) != 62:
        raise RuntimeError(f"Expected 62 audited HUMAN_GOLD IDs, found {len(audited_ids)}")

    manifest = read_json(manifest_path)
    held_out = [str(x) for fold in manifest.get("folds", []) for x in fold.get("val_ids", [])]
    if len(held_out) != 62 or sorted(held_out) != audited_ids or len(set(held_out)) != 62:
        raise RuntimeError("OOF manifest does not hold out every audited case exactly once")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("OOF guidance export requires prototype checkpoints with EMA teacher")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    cases = cv.collect_cases(config)
    by_id = {str(c["id"]): c for c in cases}
    if sorted(by_id) != audited_ids:
        raise RuntimeError("Current labeled dataset no longer matches the frozen 62-case audit")

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
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)

    rows = []
    for fold_spec in manifest["folds"]:
        fold = int(fold_spec["fold"])
        checkpoint = cv_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing completed OOF fold checkpoint: {checkpoint}. Train all five folds first."
            )

        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"Fold {fold} checkpoint has no EMA teacher")
        del student
        teacher.eval()

        items = [{"image": by_id[case_id]["image"], "id": case_id} for case_id in sorted(fold_spec["val_ids"])]
        loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                raw_id = batch.get("id")
                case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
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
    if len(rows) != 62 or sorted(str(r["case_id"]) for r in rows) != audited_ids:
        raise RuntimeError("OOF export did not produce exactly one probability map for every audited case")

    manifest_csv = output_dir / "oof62_ema_probability_manifest.csv"
    write_csv(manifest_csv, rows)
    dices = np.asarray([float(r["oof_ema_dice_at_050"]) for r in rows], dtype=float)
    metadata = {
        "version": "guided_refiner_oof62_native_ema_probabilities_v1",
        "n_cases": len(rows),
        "source_cv_dir": str(cv_dir),
        "source_split_manifest": str(manifest_path),
        "source_audit": str(audit_path),
        "threshold_for_sanity_metrics": float(args.threshold),
        "mean_oof_ema_dice_at_050": float(np.mean(dices)),
        "median_oof_ema_dice_at_050": float(np.median(dices)),
        "dice_lt_070": int(np.sum(dices < 0.70)),
        "case_ids": audited_ids,
        "manifest_csv": str(manifest_csv),
        "probability_maps_are_native_grid_float32": True,
        "leakage_rule": "Each case probability map comes only from the EMA fold model that held that case out.",
    }
    metadata_path = output_dir / "oof62_ema_probability_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("GUIDED REFINER OOF62 EMA PROBABILITY EXPORT COMPLETE")
    print(f"Cases:                 {len(rows)}")
    print(f"Mean OOF EMA Dice:     {np.mean(dices):.4f}")
    print(f"Median OOF EMA Dice:   {np.median(dices):.4f}")
    print(f"Dice < .70:            {int(np.sum(dices < 0.70))}/{len(rows)}")
    print("Held-out inference:    PASS — exactly one fold per case")
    print("Native-grid export:    PASS")
    print(f"Manifest:              {manifest_csv}")
    print(f"Metadata:              {metadata_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
