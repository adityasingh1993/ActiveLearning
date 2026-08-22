#!/usr/bin/env python3
"""Materialize native-grid A3 committee presegmentations for the Round-4 annotation batch.

Run this AFTER scripts/run_round4_active_learning_a3_committee.py. It reads the selected
round4_annotation_batch.csv, averages the exact five Final72 A3 fold predictions where each fold
prediction is the raw Student+EMA 50/50 ensemble, thresholds the five-fold mean at 0.50, inverts
that mask through the exact deterministic MONAI preprocessing trace, and writes a native-grid
.seg.nrrd for 3D Slicer.

Only the selected annotation batch is processed, so this expensive committee export is limited
to roughly 10 cases rather than the entire unlabeled pool.

The output is AI_PRESEG only. It is never HUMAN_GOLD until a human reviews/corrects it.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


def _consume_option(argv, name):
    args = list(argv)
    value = None
    cleaned = [args[0]]
    i = 1
    while i < len(args):
        token = args[i]
        if token == name:
            if i + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[i + 1]
            i += 2
            continue
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return value, cleaned


GPU, CLEAN_ARGV = _consume_option(sys.argv, "--gpu")
if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.argv = CLEAN_ARGV

import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.compat import build_invertd  # noqa: E402
from hassl.config import HASSLConfig  # noqa: E402
from hassl.data.data_engine import get_base_transforms  # noqa: E402
from hassl.data.nrrd_utils import write_mask_with_spatial_geometry  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
from scripts.build_oof_qc_dataset import load_models  # noqa: E402
from scripts.run_auto_label_pool import (  # noqa: E402
    invert_prediction_exact,
    verify_native_mask_before_write,
    verify_saved_geometry,
)

DEFAULT_ROUND4_DIR = Path("experiments/round4_active_a3_committee_v1")
DEFAULT_A3_DIR = Path("experiments/final72_screen_a3_translation4_p05_lrflip_p05")


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Materialize A3 committee preseg for Round4 selected cases")
    p.add_argument("--config", required=True)
    p.add_argument("--round4-dir", default=str(DEFAULT_ROUND4_DIR))
    p.add_argument("--annotation-csv", default=None)
    p.add_argument("--a3-dir", default=str(DEFAULT_A3_DIR))
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--resize-size", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if abs(float(args.threshold) - 0.50) > 1e-8:
        p.error("Round4 A3 committee preseg is frozen at threshold 0.50")

    round4_dir = Path(args.round4_dir)
    csv_path = Path(args.annotation_csv) if args.annotation_csv else round4_dir / "round4_annotation_batch.csv"
    rows = read_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No selected Round4 annotation cases in {csv_path}")

    ids = [str(r["case_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate case IDs in Round4 annotation batch")
    items = []
    row_by_id = {}
    for row in rows:
        case_id = str(row["case_id"])
        image = Path(str(row["image_path"]))
        if not image.exists():
            raise FileNotFoundError(f"Missing selected image {case_id}: {image}")
        items.append({"image": str(image), "id": case_id})
        row_by_id[case_id] = row

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("Round4 committee preseg requires prototype DynUNet Student+EMA mode")
    cv.apply_baseline(config, args.resize_size, epochs=1)

    a3_dir = Path(args.a3_dir)
    checkpoints = []
    for fold in range(5):
        ckpt = a3_dir / "checkpoints" / f"fold_{fold}" / "best_checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing A3 Fold{fold} checkpoint: {ckpt}")
        checkpoints.append(ckpt)

    transform = get_base_transforms(config, keys=["image"], is_training=False, apply_strong_aug=False)
    inverse_transform = build_invertd(
        keys=["pred"],
        transform=transform,
        orig_keys=["image"],
        nearest_interp=True,
        to_tensor=True,
    )
    inferer = SlidingWindowInferer(tuple(config.spatial_size), sw_batch_size=1, overlap=0.25)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    # ~10 x 128^3 float32 ~= 80 MB. Keep only the selected-case probability sums in CPU RAM.
    prob_sum = {case_id: None for case_id in ids}

    print("=" * 116)
    print("ROUND4 A3 COMMITTEE PRESEG MATERIALIZATION")
    print(f"Selected cases: {len(ids)}")
    print(f"Device:         {device}")
    print("Committee:      five A3 folds; each fold = Student+EMA 50/50; fold mean threshold=.50")
    print("=" * 116)

    for fold, checkpoint in enumerate(checkpoints):
        print(f"\nFold {fold}/4: {checkpoint}")
        student, teacher = load_models(config, checkpoint, device)
        if teacher is None:
            raise RuntimeError(f"A3 Fold{fold} checkpoint has no EMA teacher")
        loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for idx, batch in enumerate(loader, start=1):
                raw_id = batch.get("id")
                case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
                image_t = batch["image"].to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    s_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, student)))
                    t_prob = torch.sigmoid(cv.main_prediction(inferer(image_t, teacher)))
                    fold_prob = 0.5 * (s_prob + t_prob)
                arr = fold_prob[0, 0].detach().float().cpu().numpy().astype(np.float32, copy=False)
                if prob_sum[case_id] is None:
                    prob_sum[case_id] = arr.copy()
                else:
                    prob_sum[case_id] += arr
                print(f"  {idx:02d}/{len(items):02d} {case_id}")
        del student, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pack = round4_dir / "annotation_pack"
    pack.mkdir(parents=True, exist_ok=True)

    # Re-run the exact same deterministic transform to obtain a fresh untouched inversion trace.
    loader = DataLoader(Dataset(items, transform=transform), batch_size=1, shuffle=False, num_workers=0)
    outputs = []
    for batch in loader:
        raw_id = batch.get("id")
        case_id = raw_id[0] if isinstance(raw_id, (list, tuple)) else str(raw_id)
        row = row_by_id[case_id]
        rank = int(row["selection_rank"])
        avg = prob_sum[case_id] / float(len(checkpoints))
        prob_tensor = torch.from_numpy(avg).unsqueeze(0).unsqueeze(0)

        native_pred = invert_prediction_exact(prob_tensor, batch, inverse_transform, index=0)
        source_path = str(row["image_path"])
        reference_image, native_arr = verify_native_mask_before_write(native_pred, source_path)

        case_dir = pack / f"{rank:02d}_{case_id}"
        image_dir = case_dir / "image"
        preseg_dir = case_dir / "AI_PRESEG"
        image_dir.mkdir(parents=True, exist_ok=True)
        preseg_dir.mkdir(parents=True, exist_ok=True)
        src_image = Path(source_path)
        dst_image = image_dir / src_image.name
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        seg_path = preseg_dir / f"{case_id}_a3_committee_pred{config.label_suffix}"
        if seg_path.exists() and not args.overwrite:
            raise RuntimeError(f"Preseg already exists: {seg_path}. Use --overwrite intentionally.")
        write_mask_with_spatial_geometry(str(seg_path), native_arr, reference_image_path=source_path)
        verify_saved_geometry(seg_path, reference_image)

        provenance = {
            "case_id": case_id,
            "selection_rank": rank,
            "round": 4,
            "selection_profile": row.get("selection_profile", ""),
            "suggested_review_action": row.get("suggested_review_action", ""),
            "prediction_status": "AI_PRESEG",
            "human_gold_status": "PENDING",
            "prediction_definition": "mean of five A3 fold Student+EMA 50/50 probabilities @ threshold 0.50",
            "warning": "AI_PRESEG is not HUMAN_GOLD. Human review/correction is required before promotion.",
        }
        (case_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        outputs.append({
            "selection_rank": rank,
            "case_id": case_id,
            "image_path": str(dst_image),
            "a3_committee_preseg_path": str(seg_path),
            "selection_profile": row.get("selection_profile", ""),
            "suggested_review_action": row.get("suggested_review_action", ""),
            "human_gold_status": "PENDING",
        })
        print(f"WROTE {case_id}: {seg_path}")

    fields = list(outputs[0].keys())
    with (round4_dir / "round4_annotation_pack_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(outputs)

    print("\n" + "=" * 116)
    print("ROUND4 ANNOTATION PACK READY")
    print(f"Pack: {pack}")
    print("Open each image + AI_PRESEG in Slicer, correct it, and save a HUMAN_GOLD .seg.nrrd separately.")
    print("Do not overwrite AI_PRESEG; preserve it for provenance/comparison.")
    print("=" * 116)


if __name__ == "__main__":
    main()
