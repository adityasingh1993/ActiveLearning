#!/usr/bin/env python3
"""Screen C2: auxiliary signed-distance-field boundary supervision on Final72 A3.

C2 changes the training objective/last output layer while keeping the A3 data/model recipe fixed.
The network predicts two channels during training:
    channel 0: bladder segmentation logit
    channel 1: normalized signed distance to the bladder wall
At validation/inference ``forward()`` exposes only channel 0, so segmentation evaluation remains
identical to A3 and the SDF output is discarded.

SDF target
----------
- generated from the deterministic 128^3 HUMAN_GOLD mask before A3 random spatial augmentation
- physical spacing is read from the post-preprocessing MetaTensor affine
- inside bladder is positive, outside is negative
- raw SDF is normalized by --sdf-band-mm and clipped to [-1, +1]
- Smooth-L1 is evaluated ONLY where |raw SDF| <= --sdf-band-mm
  so far background cannot dominate the auxiliary objective
- the SDF and its band mask receive the exact same A3 LR flip / translation as the label

Default C2:
    total_loss = DiceCE(seg) + 0.10 * SmoothL1(SDF within +/-2 mm band)

Everything else stays matched to A3:
- exact Final72 HUMAN_GOLD / frozen original47 held-out folds
- DynUNet, resize128
- translation +/-4 vox p=.5
- LR flip p=.5 on RAS spatial axis 0
- AdamW 1e-4, dropout0, lambda_unsup0, seed42, 100 epochs
- prototype Student+EMA; raw 50/50 ensemble @ .50
- no LCC and no external31

Default screening runs folds 1 and 2 only.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.networks.nets import DynUNet
from monai.transforms import Compose, RandAffined, RandFlipd
from scipy import ndimage

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
DEFAULT_OUTPUT = Path("experiments/final72_screen_c2_sdf_boundary_a3")


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


class AddSignedDistanceTargetd:
    """Create normalized physical SDF + boundary-band mask from a binary label."""

    def __init__(self, label_key="label", sdf_key="sdf", band_key="sdf_band", band_mm=2.0):
        if band_mm <= 0:
            raise ValueError("band_mm must be > 0")
        self.label_key = label_key
        self.sdf_key = sdf_key
        self.band_key = band_key
        self.band_mm = float(band_mm)

    @staticmethod
    def _spacing_from_meta(label):
        if isinstance(label, MetaTensor) and hasattr(label, "affine"):
            affine = label.affine.detach().cpu().numpy()
            if affine.shape == (4, 4):
                spacing = np.linalg.norm(affine[:3, :3], axis=0)
                if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                    return tuple(float(x) for x in spacing)
        return (1.0, 1.0, 1.0)

    def __call__(self, data):
        out = dict(data)
        label = out[self.label_key]
        is_meta = isinstance(label, MetaTensor)
        tensor = label.as_tensor() if is_meta else torch.as_tensor(label)
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise RuntimeError(f"C2 expects label [1,D,H,W], got {tuple(tensor.shape)}")

        mask = (tensor[0].detach().cpu().numpy() > 0.5)
        if not mask.any():
            raise RuntimeError("C2 cannot build SDF from an empty HUMAN_GOLD label")
        spacing = self._spacing_from_meta(label)

        inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        raw_sdf = inside - outside  # inside positive, outside negative
        normalized = np.clip(raw_sdf / self.band_mm, -1.0, 1.0).astype(np.float32)
        band = (np.abs(raw_sdf) <= self.band_mm).astype(np.float32)

        sdf_t = torch.from_numpy(normalized).unsqueeze(0).to(dtype=torch.float32)
        band_t = torch.from_numpy(band).unsqueeze(0).to(dtype=torch.float32)

        if is_meta:
            meta = dict(label.meta)
            affine = label.affine.clone()
            ops = list(label.applied_operations)
            out[self.sdf_key] = MetaTensor(
                sdf_t, affine=affine, meta=meta, applied_operations=list(ops)
            )
            out[self.band_key] = MetaTensor(
                band_t, affine=affine.clone(), meta=dict(meta), applied_operations=list(ops)
            )
        else:
            out[self.sdf_key] = sdf_t
            out[self.band_key] = band_t
        return out


def c2_train_transform(base_transform, band_mm):
    """Deterministic A3 preprocessing -> SDF target -> exact shared A3 spatial augmentation."""
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        AddSignedDistanceTargetd(band_mm=band_mm),
        RandFlipd(keys=["image", "label", "sdf", "sdf_band"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label", "sdf", "sdf_band"],
            prob=0.5,
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(4.0, 4.0, 4.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest", "bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


def make_dataset(items, transform, use_cache):
    if use_cache and items:
        return CacheDataset(items, transform=transform, cache_rate=1.0, copy_cache=False)
    return Dataset(items, transform=transform)


def install_c2_loader_hook(band_mm):
    def build_cv_dataloaders(config):
        cases = {c["id"]: c for c in cv.collect_cases(config)}
        train_ids = sorted(list(config._cv_train_ids))
        val_ids = sorted(list(config._cv_val_ids))
        missing = sorted((set(train_ids) | set(val_ids)) - set(cases))
        if missing:
            raise RuntimeError(f"Missing labeled cases for C2 fold: {missing}")

        train_base = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        train_t = c2_train_transform(train_base, band_mm=band_mm)
        # Validation remains exactly A3 deterministic preprocessing; no SDF is needed.
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )

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


def _seg_only(output):
    """Select segmentation channel while preserving DynUNet deep-supervision shape."""
    if isinstance(output, (list, tuple)):
        return [_seg_only(x) for x in output]
    if not torch.is_tensor(output):
        raise TypeError(f"Unexpected DynUNet output type: {type(output)}")
    if output.ndim == 6:  # [B, heads, C, D, H, W]
        return output[:, :, 0:1]
    if output.ndim == 5:  # [B, C, D, H, W]
        return output[:, 0:1]
    raise RuntimeError(f"Unexpected DynUNet output shape: {tuple(output.shape)}")


class SDFDynUNet(nn.Module):
    """Frozen DynUNet decoder with 2 task channels; inference exposes segmentation only."""

    def __init__(self, num_classes, dropout):
        super().__init__()
        if int(num_classes) != 1:
            raise RuntimeError("C2 is locked to binary bladder segmentation")
        self.core = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 4,
            filters=[16, 32, 64, 128, 256],
            dropout=dropout,
            norm_name="instance",
            deep_supervision=True,
        )

    def forward_multitask(self, x):
        return self.core(x)

    def forward(self, x):
        return _seg_only(self.core(x))


def build_sdf_network(backbone: str, num_classes: int, dropout: float):
    if backbone != "dynunet":
        raise RuntimeError(f"C2 is locked to DynUNet, got {backbone!r}")
    return SDFDynUNet(num_classes=num_classes, dropout=dropout)


def _multitask_heads(output):
    if isinstance(output, (list, tuple)):
        return list(output)
    if torch.is_tensor(output) and output.ndim == 6:
        return [output[:, i] for i in range(output.shape[1])]
    if torch.is_tensor(output) and output.ndim == 5:
        return [output]
    raise RuntimeError(f"Unexpected C2 multitask output: {type(output)} / {getattr(output, 'shape', None)}")


def _resize_like(tensor, reference, mode):
    if tuple(tensor.shape[2:]) == tuple(reference.shape[2:]):
        return tensor
    kwargs = {"size": reference.shape[2:], "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def c2_multitask_loss(trainer, output, target, sdf_target, sdf_band, sdf_lambda):
    """Deep-supervision weighted DiceCE + boundary-band SmoothL1."""
    heads = _multitask_heads(output)
    weights = [1.0 / (2 ** i) for i in range(len(heads))]
    total_w = sum(weights)
    seg_total = torch.tensor(0.0, device=target.device)
    sdf_total = torch.tensor(0.0, device=target.device)

    for head, weight in zip(heads, weights):
        if head.shape[1] != 2:
            raise RuntimeError(f"C2 expects 2 task channels, got {tuple(head.shape)}")
        seg_logits = head[:, 0:1]
        sdf_pred = head[:, 1:2]
        t = _resize_like(target, seg_logits, "nearest")
        s = _resize_like(sdf_target, sdf_pred, "trilinear")
        band = _resize_like(sdf_band, sdf_pred, "nearest") > 0.5

        seg_loss = trainer.criterion(seg_logits, t)
        if seg_loss.ndim > 0:
            seg_loss = seg_loss.mean()

        if band.any():
            sdf_voxel = F.smooth_l1_loss(sdf_pred.float(), s.float(), reduction="none")
            sdf_loss = sdf_voxel[band].mean()
        else:
            sdf_loss = torch.tensor(0.0, device=target.device)

        factor = float(weight / total_w)
        seg_total = seg_total + factor * seg_loss
        sdf_total = sdf_total + factor * sdf_loss

    total = seg_total + float(sdf_lambda) * sdf_total
    return total, seg_total, sdf_total


def make_c2_train_one_epoch(sdf_lambda):
    """Prototype supervised epoch matching A3 optimizer/EMA behavior + SDF auxiliary loss."""
    def train_one_epoch_uamt(self, epoch: int):
        if self.unlabeled_loader is not None and len(self.unlabeled_loader.dataset) != 0:
            raise RuntimeError("C2 screen is supervised-only; unlabeled loader must be empty")
        if abs(float(getattr(self.config, "lambda_unsup", 0.0))) > 1e-12:
            raise RuntimeError("C2 screen requires lambda_unsup=0")

        self.net_A.train()
        pseudo_weight = getattr(self.config, "pseudo_label_weight", 0.5)
        total_loss = 0.0
        total_seg = 0.0
        total_sdf = 0.0

        for batch_data in self.labeled_loader:
            inputs = batch_data["image"].to(self.device)
            targets = batch_data["label"].float().to(self.device)
            sdf_target = batch_data["sdf"].float().to(self.device)
            sdf_band = batch_data["sdf_band"].float().to(self.device)
            provenance = batch_data.get("provenance", ["human"] * inputs.size(0))
            sample_weights = torch.tensor(
                [1.0 if p in ["human", "human_corrected"] else pseudo_weight for p in provenance],
                device=self.device,
                dtype=torch.float32,
            )

            self.optimizer.zero_grad()
            sample_total = []
            sample_seg = []
            sample_sdf = []
            with torch.amp.autocast(self.device_type, enabled=(self.device_type == "cuda")):
                raw = self.net_A.forward_multitask(inputs)
                for b in range(inputs.size(0)):
                    if torch.is_tensor(raw):
                        raw_b = raw[b:b + 1]
                    else:
                        raw_b = [x[b:b + 1] for x in raw]
                    loss_b, seg_b, sdf_b = c2_multitask_loss(
                        self,
                        raw_b,
                        targets[b:b + 1],
                        sdf_target[b:b + 1],
                        sdf_band[b:b + 1],
                        sdf_lambda=sdf_lambda,
                    )
                    sample_total.append(loss_b)
                    sample_seg.append(seg_b)
                    sample_sdf.append(sdf_b)

                weights = sample_weights / (sample_weights.sum() + 1e-8)
                loss = (torch.stack(sample_total) * weights).sum()
                seg_loss = (torch.stack(sample_seg) * weights).sum()
                sdf_loss = (torch.stack(sample_sdf) * weights).sum()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net_A.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.teacher.update(self.net_A, decay=self.config.ema_decay)

            total_loss += float(loss.item())
            total_seg += float(seg_loss.item())
            total_sdf += float(sdf_loss.item())

        n = max(1, len(self.labeled_loader))
        self._c2_last_seg_loss = total_seg / n
        self._c2_last_sdf_loss = total_sdf / n
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # train() expects six values in prototype mode.
        return total_loss / n, total_seg / n, 0.0, 0.0, float("nan"), float("nan")

    return train_one_epoch_uamt


def metric_mean(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def compare_to_a3(candidate_rows, a3_rows, selected_folds, size_profile, output_dir):
    selected = set(int(x) for x in selected_folds)
    cand = {str(r["case_id"]): r for r in candidate_rows if int(r["fold"]) in selected}
    base = {str(r["case_id"]): r for r in a3_rows if int(r["fold"]) in selected}
    if set(cand) != set(base):
        raise RuntimeError("C2 and A3 must contain exact same held-out IDs on selected folds")
    size_by_id = {str(r["case_id"]): str(r["size_group"]) for r in size_profile}

    paired = []
    for case_id in sorted(cand):
        a, b = base[case_id], cand[case_id]
        paired.append({
            "case_id": case_id,
            "fold": int(b["fold"]),
            "size_group": size_by_id[case_id],
            "a3_dice": float(a["dice"]),
            "c2_dice": float(b["dice"]),
            "delta_dice": float(b["dice"]) - float(a["dice"]),
            "a3_precision": float(a["precision"]),
            "c2_precision": float(b["precision"]),
            "delta_precision": float(b["precision"]) - float(a["precision"]),
            "a3_recall": float(a["recall"]),
            "c2_recall": float(b["recall"]),
            "delta_recall": float(b["recall"]) - float(a["recall"]),
            "a3_hd95": float(a["hd95"]),
            "c2_hd95": float(b["hd95"]),
            "delta_hd95": float(b["hd95"]) - float(a["hd95"]),
            "a3_rve": float(a["rve"]),
            "c2_rve": float(b["rve"]),
            "delta_rve": float(b["rve"]) - float(a["rve"]),
        })
    write_csv(output_dir / "screening_vs_a3_case_comparison.csv", paired)

    def block(rows):
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "a3_mean_dice": metric_mean(rows, "a3_dice"),
            "c2_mean_dice": metric_mean(rows, "c2_dice"),
            "delta_mean_dice": metric_mean(rows, "delta_dice"),
            "a3_mean_precision": metric_mean(rows, "a3_precision"),
            "c2_mean_precision": metric_mean(rows, "c2_precision"),
            "delta_mean_precision": metric_mean(rows, "delta_precision"),
            "a3_mean_recall": metric_mean(rows, "a3_recall"),
            "c2_mean_recall": metric_mean(rows, "c2_recall"),
            "delta_mean_recall": metric_mean(rows, "delta_recall"),
            "a3_mean_hd95": metric_mean(rows, "a3_hd95"),
            "c2_mean_hd95": metric_mean(rows, "c2_hd95"),
            "improved": int(sum(r["delta_dice"] > 1e-6 for r in rows)),
            "worsened": int(sum(r["delta_dice"] < -1e-6 for r in rows)),
            "improved_ge_0p05": int(sum(r["delta_dice"] >= 0.05 for r in rows)),
            "worsened_le_minus_0p05": int(sum(r["delta_dice"] <= -0.05 for r in rows)),
        }

    folds = {str(f): block([r for r in paired if int(r["fold"]) == f]) for f in selected_folds}
    groups = {g: block([r for r in paired if r["size_group"] == g]) for g in ("SMALL", "MEDIUM", "LARGE")}
    overall = block(paired)
    summary = {
        "version": "final72_c2_sdf_boundary_screen_v1",
        "reference": "A3 = +/-4 vox p=.5 + LR flip p=.5 + DiceCE + 128^3",
        "screening_folds": list(selected_folds),
        "overall": overall,
        "folds": folds,
        "fixed_validation_size_groups": groups,
        "screening_only": True,
    }
    (output_dir / "screening_vs_a3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 124)
    print("C2_SDF vs A3 — SDF BOUNDARY-SUPERVISION SCREEN")
    for f in selected_folds:
        s = folds[str(f)]
        print(
            f"Fold {f}: A3={s['a3_mean_dice']:.4f} -> C2={s['c2_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | PrecDelta={s['delta_mean_precision']:+.4f} | "
            f"RecDelta={s['delta_mean_recall']:+.4f}"
        )
    print(
        f"Combined: A3={overall['a3_mean_dice']:.4f} -> C2={overall['c2_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f}) | PrecDelta={overall['delta_mean_precision']:+.4f} | "
        f"RecDelta={overall['delta_mean_recall']:+.4f}"
    )
    print("\nFIXED VALIDATION SIZE GROUPS (diagnostic only)")
    for g in ("SMALL", "MEDIUM", "LARGE"):
        s = groups[g]
        print(
            f"  {g}: n={s['n']} | Dice {s['a3_mean_dice']:.4f}->{s['c2_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | Prec {s['a3_mean_precision']:.4f}->{s['c2_mean_precision']:.4f} "
            f"({s['delta_mean_precision']:+.4f}) | Rec {s['a3_mean_recall']:.4f}->{s['c2_mean_recall']:.4f} "
            f"({s['delta_mean_recall']:+.4f}) | HD95 {s['a3_mean_hd95']:.2f}->{s['c2_mean_hd95']:.2f}mm"
        )
    print(
        f"Case effects: improved={overall['improved']} | worsened={overall['worsened']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: full frozen CV is required before promotion.")
    print("=" * 124)


def preflight(config, fold_spec, band_mm):
    cases = {c["id"]: c for c in cv.collect_cases(config)}
    case_id = sorted(fold_spec["train_ids"])[0]
    base = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=True, apply_strong_aug=False
    )
    sample = c2_train_transform(base, band_mm=band_mm)(cases[case_id])
    image = torch.as_tensor(sample["image"])
    label = torch.as_tensor(sample["label"])
    sdf = torch.as_tensor(sample["sdf"])
    band = torch.as_tensor(sample["sdf_band"])
    if tuple(image.shape) != (1, 128, 128, 128):
        raise RuntimeError(f"C2 preflight image shape mismatch: {tuple(image.shape)}")
    if label.shape != sdf.shape or label.shape != band.shape:
        raise RuntimeError(f"C2 target shape mismatch label={label.shape}, sdf={sdf.shape}, band={band.shape}")
    if float(band.sum()) <= 0:
        raise RuntimeError("C2 preflight found empty SDF boundary band")
    print(
        f"C2 preflight PASS | case={case_id} | image={tuple(image.shape)} | "
        f"SDF=[{float(sdf.min()):+.3f},{float(sdf.max()):+.3f}] | "
        f"band_fraction={100.0 * float(band.float().mean()):.3f}%"
    )


def main():
    p = argparse.ArgumentParser(description="C2 SDF boundary-supervision screening on Final72 A3")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", default="screen", help="screen (=1+2), all, or 0..4")
    p.add_argument("--sdf-lambda", type=float, default=0.10)
    p.add_argument("--sdf-band-mm", type=float, default=2.0)
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--a3-cv-dir", default=str(A3_CV))
    p.add_argument("--size-profile", default=str(SIZE_PROFILE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.sdf_lambda <= 0 or args.sdf_band_mm <= 0:
        p.error("--sdf-lambda and --sdf-band-mm must be > 0")

    selected_folds = spatial_screen.parse_screen_fold(args.fold)
    output_dir = Path(args.output_dir)
    source_manifest_path = Path(args.source_cv_dir) / "cv_splits.json"
    a3_results_path = Path(args.a3_cv_dir) / "cv_results.csv"
    a3_rows = read_csv(a3_results_path)
    size_profile = read_csv(Path(args.size_profile))
    if len({str(r["case_id"]) for r in a3_rows}) != spatial_screen.EXPECTED_SOURCE:
        raise RuntimeError("A3 reference must contain exact 47 frozen held-out cases")

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype":
        raise RuntimeError("C2 screening requires prototype Student+EMA mode")
    if config.unet_backbone != "dynunet":
        raise RuntimeError(f"C2 is locked to dynunet, got {config.unet_backbone!r}")

    _, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest_path, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    # Apply the exact frozen baseline before preflight so preprocessing is guaranteed 128^3.
    cv.apply_baseline(config, resize_size=128, epochs=100)
    preflight(config, fold_map[selected_folds[0]], band_mm=float(args.sdf_band_mm))

    # Runner-local monkey patches only; repository trainer behavior is not changed.
    trainer_module.build_network = build_sdf_network
    cv.build_network = build_sdf_network
    trainer_module.HASSLTrainer.train_one_epoch_uamt = make_c2_train_one_epoch(float(args.sdf_lambda))
    install_c2_loader_hook(band_mm=float(args.sdf_band_mm))

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
        "version": "final72_c2_sdf_boundary_v1",
        "purpose": "auxiliary SDF wall supervision to improve foreground-boundary discrimination",
        "reference": "Final72 A3",
        "screening_folds": selected_folds,
        "source_manifest": str(source_manifest_path),
        "audit_metadata": str(args.audit_metadata),
        "n_total_human_gold": spatial_screen.EXPECTED_TOTAL,
        "n_train_only_extra": len(extra_ids),
        "recipe": {
            "architecture": "DynUNet shared decoder with 2 training task channels",
            "inference_output": "segmentation channel only",
            "resize_size": [128, 128, 128],
            "segmentation_loss": "DiceCE",
            "sdf_loss": "SmoothL1 within physical boundary band only",
            "sdf_lambda": float(args.sdf_lambda),
            "sdf_band_mm": float(args.sdf_band_mm),
            "sdf_sign": "inside_positive_outside_negative",
            "sdf_normalization": "clip(raw_sdf / band_mm, -1, +1)",
            "translation_voxels": 4.0,
            "translation_probability": 0.5,
            "lr_flip": True,
            "lr_flip_probability": 0.5,
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
            raise RuntimeError(f"Existing C2 plan differs: {plan_path}; use a fresh output dir")
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 124)
    print("C2 — FINAL72 A3 + SDF BOUNDARY SUPERVISION")
    print(f"Running folds:         {selected_folds}")
    print("Segmentation:          DiceCE (unchanged)")
    print(f"Auxiliary SDF:         SmoothL1, lambda={args.sdf_lambda:g}, band=+/-{args.sdf_band_mm:g} mm")
    print("SDF scope:             boundary band only; far background excluded")
    print("Spatial recipe:        A3 = +/-4 vox p=.5 + LR flip p=.5")
    print("Evaluation:            raw Student+EMA 50/50 @ .50")
    print("External31:            NOT ACCESSED")
    print("=" * 124)

    new_rows = []
    for fold in selected_folds:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[fold], output_dir))

    results_path = output_dir / "cv_results.csv"
    existing = cv.read_results(results_path)
    kept = [r for r in existing if int(r["fold"]) not in set(selected_folds)]
    combined = kept + new_rows
    combined.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    cv.write_results(results_path, combined)

    compare_to_a3(combined, a3_rows, selected_folds, size_profile, output_dir)
    print(f"\nResults:    {results_path}")
    print(f"Plan:       {plan_path}")
    print(f"Comparison: {output_dir / 'screening_vs_a3_case_comparison.csv'}")
    print(f"Summary:    {output_dir / 'screening_vs_a3_summary.json'}")


if __name__ == "__main__":
    main()
