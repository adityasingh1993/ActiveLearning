#!/usr/bin/env python3
"""C2-v2: SDF boundary supervision on the exact Final72 A3 one-channel DynUNet.

Why v2
------
The first C2 prototype used a second network output channel to regress an SDF. On this tiny-
foreground bladder task that can materially disturb early segmentation optimization. C2-v2 keeps
the exact A3 architecture (one segmentation output channel) and uses the ground-truth signed
distance field only to add a small differentiable boundary loss to the segmentation probability.

Base recipe is unchanged from A3:
- exact Final72 HUMAN_GOLD / frozen original47 OOF folds
- DynUNet, in_channels=1, out_channels=1, resize128, DiceCE
- translation +/-4 vox p=.5
- LR flip p=.5 on RAS spatial axis 0
- AdamW 1e-4, dropout0, lambda_unsup0, seed42, 100 epochs
- prototype Student+EMA; raw 50/50 ensemble @ .50
- no LCC; no external31

Boundary target
---------------
- generated after deterministic 128^3 preprocessing and before A3 random spatial augmentation
- physical signed distance in mm from the HUMAN_GOLD mask
- negative inside bladder, positive outside bladder
- clipped/normalized to [-1,+1] using --sdf-band-mm (default 2 mm)
- only voxels within +/- band_mm contribute to the boundary term
- label, SDF and band mask receive the exact same LR flip / translation

Loss
----
    L = DiceCE(seg_logits, mask) + lambda_boundary * mean(sigmoid(seg_logits) * sdf_norm)

inside the narrow SDF band only. Because sdf_norm is negative inside and positive outside,
minimizing this term rewards high probability just inside the bladder and penalizes probability
just outside the wall. The default lambda is deliberately small (0.05).

Default screening fold is 1. Use --fold screen only after Fold1 behaves normally.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
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
DEFAULT_OUTPUT = Path("experiments/final72_screen_c2v2_sdf_boundary_a3")


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class AddBoundarySDFd:
    """Create normalized signed distance and narrow-band mask from label.

    Sign convention is negative inside / positive outside for direct surface-loss use.
    Metadata is copied without passing affine twice, avoiding MONAI's duplicate-affine warning.
    """

    def __init__(self, band_mm=2.0, label_key="label", sdf_key="boundary_sdf", band_key="boundary_band"):
        if float(band_mm) <= 0:
            raise ValueError("band_mm must be > 0")
        self.band_mm = float(band_mm)
        self.label_key = label_key
        self.sdf_key = sdf_key
        self.band_key = band_key

    @staticmethod
    def spacing_from_meta(label):
        if isinstance(label, MetaTensor) and hasattr(label, "affine"):
            affine = label.affine.detach().cpu().numpy()
            if affine.shape == (4, 4):
                spacing = np.linalg.norm(affine[:3, :3], axis=0)
                if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                    return tuple(float(x) for x in spacing)
        return (1.0, 1.0, 1.0)

    @staticmethod
    def make_meta_like(array_t, label):
        # MetaTensor can recover affine from copied metadata. Do not pass affine separately
        # when it is already present in meta; this avoids "Setting affine ... overwritten".
        meta = dict(label.meta)
        ops = list(label.applied_operations)
        return MetaTensor(array_t, meta=meta, applied_operations=ops)

    def __call__(self, data):
        out = dict(data)
        label = out[self.label_key]
        is_meta = isinstance(label, MetaTensor)
        t = label.as_tensor() if is_meta else torch.as_tensor(label)
        if t.ndim != 4 or int(t.shape[0]) != 1:
            raise RuntimeError(f"Expected label [1,D,H,W], got {tuple(t.shape)}")
        mask = t[0].detach().cpu().numpy() > 0.5
        if not mask.any():
            raise RuntimeError("Cannot form boundary SDF from empty HUMAN_GOLD mask")

        spacing = self.spacing_from_meta(label)
        dist_in = ndimage.distance_transform_edt(mask, sampling=spacing)
        dist_out = ndimage.distance_transform_edt(~mask, sampling=spacing)
        raw = dist_out - dist_in  # negative inside, positive outside
        sdf = np.clip(raw / self.band_mm, -1.0, 1.0).astype(np.float32)
        band = (np.abs(raw) <= self.band_mm).astype(np.float32)
        sdf_t = torch.from_numpy(sdf).unsqueeze(0)
        band_t = torch.from_numpy(band).unsqueeze(0)

        if is_meta:
            out[self.sdf_key] = self.make_meta_like(sdf_t, label)
            out[self.band_key] = self.make_meta_like(band_t, label)
        else:
            out[self.sdf_key] = sdf_t
            out[self.band_key] = band_t
        return out


def c2v2_train_transform(base_transform, band_mm):
    base_steps = list(getattr(base_transform, "transforms", [base_transform]))
    return Compose(base_steps + [
        AddBoundarySDFd(band_mm=band_mm),
        RandFlipd(
            keys=["image", "label", "boundary_sdf", "boundary_band"],
            prob=0.5,
            spatial_axis=0,
        ),
        RandAffined(
            keys=["image", "label", "boundary_sdf", "boundary_band"],
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


def install_loader_hook(band_mm):
    def build_cv_dataloaders(config):
        cases = {c["id"]: c for c in cv.collect_cases(config)}
        train_ids = sorted(list(config._cv_train_ids))
        val_ids = sorted(list(config._cv_val_ids))
        missing = sorted((set(train_ids) | set(val_ids)) - set(cases))
        if missing:
            raise RuntimeError(f"Missing labeled cases: {missing}")

        train_base = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=True, apply_strong_aug=False
        )
        val_t = cv.ORIGINAL_GET_TRANSFORMS(
            config, keys=["image", "label"], is_training=False, apply_strong_aug=False
        )
        train_t = c2v2_train_transform(train_base, band_mm)

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


def split_deep_supervision(output):
    if isinstance(output, (list, tuple)):
        return list(output)
    if torch.is_tensor(output) and output.ndim == 6:
        return [output[:, i] for i in range(output.shape[1])]
    if torch.is_tensor(output) and output.ndim == 5:
        return [output]
    raise RuntimeError(f"Unexpected DynUNet output: {type(output)} {getattr(output, 'shape', None)}")


def resize_like(t, ref, mode):
    if tuple(t.shape[2:]) == tuple(ref.shape[2:]):
        return t
    kwargs = {"size": ref.shape[2:], "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(t.float(), **kwargs)


def per_sample_loss(trainer, output, target, sdf, band, lambda_boundary):
    heads = split_deep_supervision(output)
    head_weights = [1.0 / (2 ** i) for i in range(len(heads))]
    denom = sum(head_weights)
    seg_total = torch.tensor(0.0, device=target.device)
    boundary_total = torch.tensor(0.0, device=target.device)

    for head, hw in zip(heads, head_weights):
        if int(head.shape[1]) != 1:
            raise RuntimeError(f"C2-v2 requires exact one-channel A3 output, got {tuple(head.shape)}")
        t = resize_like(target, head, "nearest")
        s = resize_like(sdf, head, "trilinear")
        b = resize_like(band, head, "nearest") > 0.5

        seg = trainer.criterion(head, t)
        if seg.ndim > 0:
            seg = seg.mean()

        prob = torch.sigmoid(head.float())
        if b.any():
            boundary = (prob[b] * s.float()[b]).mean()
        else:
            boundary = torch.tensor(0.0, device=target.device)

        f = float(hw / denom)
        seg_total = seg_total + f * seg
        boundary_total = boundary_total + f * boundary

    total = seg_total + float(lambda_boundary) * boundary_total
    return total, seg_total, boundary_total


def make_train_epoch(lambda_boundary):
    def train_one_epoch_uamt(self, epoch: int):
        if self.unlabeled_loader is not None and len(self.unlabeled_loader.dataset) != 0:
            raise RuntimeError("C2-v2 is supervised-only")
        if abs(float(getattr(self.config, "lambda_unsup", 0.0))) > 1e-12:
            raise RuntimeError("C2-v2 requires lambda_unsup=0")

        self.net_A.train()
        pseudo_weight = getattr(self.config, "pseudo_label_weight", 0.5)
        total_loss = total_seg = total_boundary = 0.0

        for batch in self.labeled_loader:
            image = batch["image"].to(self.device)
            target = batch["label"].float().to(self.device)
            sdf = batch["boundary_sdf"].float().to(self.device)
            band = batch["boundary_band"].float().to(self.device)
            provenance = batch.get("provenance", ["human"] * image.size(0))
            weights = torch.tensor(
                [1.0 if p in ["human", "human_corrected"] else pseudo_weight for p in provenance],
                dtype=torch.float32,
                device=self.device,
            )

            self.optimizer.zero_grad()
            losses = []
            seg_losses = []
            boundary_losses = []
            with torch.amp.autocast(self.device_type, enabled=(self.device_type == "cuda")):
                pred = self.net_A(image)
                for i in range(image.size(0)):
                    if torch.is_tensor(pred):
                        p_i = pred[i:i+1]
                    else:
                        p_i = [x[i:i+1] for x in pred]
                    l, l_seg, l_b = per_sample_loss(
                        self,
                        p_i,
                        target[i:i+1],
                        sdf[i:i+1],
                        band[i:i+1],
                        lambda_boundary,
                    )
                    losses.append(l)
                    seg_losses.append(l_seg)
                    boundary_losses.append(l_b)

                wn = weights / (weights.sum() + 1e-8)
                loss = (torch.stack(losses) * wn).sum()
                seg_loss = (torch.stack(seg_losses) * wn).sum()
                boundary_loss = (torch.stack(boundary_losses) * wn).sum()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net_A.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.teacher.update(self.net_A, decay=self.config.ema_decay)

            total_loss += float(loss.item())
            total_seg += float(seg_loss.item())
            total_boundary += float(boundary_loss.item())

        n = max(1, len(self.labeled_loader))
        self._c2v2_seg_loss = total_seg / n
        self._c2v2_boundary_loss = total_boundary / n
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss / n, total_seg / n, 0.0, 0.0, float("nan"), float("nan")

    return train_one_epoch_uamt


def metric_mean(rows, key):
    a = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def compare_to_a3(candidate_rows, a3_rows, selected_folds, size_profile, output_dir):
    selected = set(int(x) for x in selected_folds)
    cand = {str(r["case_id"]): r for r in candidate_rows if int(r["fold"]) in selected}
    base = {str(r["case_id"]): r for r in a3_rows if int(r["fold"]) in selected}
    if set(cand) != set(base):
        raise RuntimeError("C2-v2 and A3 must contain identical held-out IDs")
    size_by_id = {str(r["case_id"]): str(r["size_group"]) for r in size_profile}

    paired = []
    for cid in sorted(cand):
        a, c = base[cid], cand[cid]
        paired.append({
            "case_id": cid,
            "fold": int(c["fold"]),
            "size_group": size_by_id[cid],
            "a3_dice": float(a["dice"]), "c2_dice": float(c["dice"]),
            "delta_dice": float(c["dice"]) - float(a["dice"]),
            "a3_precision": float(a["precision"]), "c2_precision": float(c["precision"]),
            "delta_precision": float(c["precision"]) - float(a["precision"]),
            "a3_recall": float(a["recall"]), "c2_recall": float(c["recall"]),
            "delta_recall": float(c["recall"]) - float(a["recall"]),
            "a3_hd95": float(a["hd95"]), "c2_hd95": float(c["hd95"]),
            "delta_hd95": float(c["hd95"]) - float(a["hd95"]),
            "a3_rve": float(a["rve"]), "c2_rve": float(c["rve"]),
            "delta_rve": float(c["rve"]) - float(a["rve"]),
        })
    write_csv(output_dir / "screening_vs_a3_case_comparison.csv", paired)

    def block(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "a3_mean_dice": metric_mean(rs, "a3_dice"),
            "c2_mean_dice": metric_mean(rs, "c2_dice"),
            "delta_mean_dice": metric_mean(rs, "delta_dice"),
            "delta_mean_precision": metric_mean(rs, "delta_precision"),
            "delta_mean_recall": metric_mean(rs, "delta_recall"),
            "a3_mean_hd95": metric_mean(rs, "a3_hd95"),
            "c2_mean_hd95": metric_mean(rs, "c2_hd95"),
            "improved": int(sum(r["delta_dice"] > 1e-6 for r in rs)),
            "worsened": int(sum(r["delta_dice"] < -1e-6 for r in rs)),
            "improved_ge_0p05": int(sum(r["delta_dice"] >= 0.05 for r in rs)),
            "worsened_le_minus_0p05": int(sum(r["delta_dice"] <= -0.05 for r in rs)),
        }

    folds = {str(f): block([r for r in paired if int(r["fold"]) == f]) for f in selected_folds}
    groups = {g: block([r for r in paired if r["size_group"] == g]) for g in ("SMALL", "MEDIUM", "LARGE")}
    overall = block(paired)
    payload = {
        "version": "final72_c2v2_sdf_boundary_screen_v1",
        "screening_folds": list(selected_folds),
        "overall": overall,
        "folds": folds,
        "fixed_validation_size_groups": groups,
        "screening_only": True,
    }
    (output_dir / "screening_vs_a3_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 122)
    print("C2V2_SDF vs A3 — SDF BOUNDARY-LOSS SCREEN")
    for f in selected_folds:
        s = folds[str(f)]
        print(
            f"Fold {f}: A3={s['a3_mean_dice']:.4f} -> C2v2={s['c2_mean_dice']:.4f} "
            f"({s['delta_mean_dice']:+.4f}) | PrecDelta={s['delta_mean_precision']:+.4f} | "
            f"RecDelta={s['delta_mean_recall']:+.4f}"
        )
    print(
        f"Combined: A3={overall['a3_mean_dice']:.4f} -> C2v2={overall['c2_mean_dice']:.4f} "
        f"({overall['delta_mean_dice']:+.4f}) | PrecDelta={overall['delta_mean_precision']:+.4f} | "
        f"RecDelta={overall['delta_mean_recall']:+.4f}"
    )
    print("FIXED VALIDATION SIZE GROUPS (diagnostic only)")
    for g in ("SMALL", "MEDIUM", "LARGE"):
        s = groups[g]
        if s["n"]:
            print(
                f"  {g}: n={s['n']} | DiceDelta={s['delta_mean_dice']:+.4f} | "
                f"PrecDelta={s['delta_mean_precision']:+.4f} | RecDelta={s['delta_mean_recall']:+.4f} | "
                f"HD95 {s['a3_mean_hd95']:.2f}->{s['c2_mean_hd95']:.2f}mm"
            )
    print(
        f"Case effects: improved={overall['improved']} | worsened={overall['worsened']} | "
        f"+>=.05={overall['improved_ge_0p05']} | <=-.05={overall['worsened_le_minus_0p05']}"
    )
    print("SCREENING ONLY: full frozen CV is required before promotion.")
    print("=" * 122)


def preflight(config, fold_spec, band_mm, lambda_boundary):
    cases = {c["id"]: c for c in cv.collect_cases(config)}
    cid = sorted(fold_spec["train_ids"])[0]
    base = cv.ORIGINAL_GET_TRANSFORMS(
        config, keys=["image", "label"], is_training=True, apply_strong_aug=False
    )
    sample = c2v2_train_transform(base, band_mm)(cases[cid])
    img = sample["image"]
    lab = sample["label"]
    sdf = sample["boundary_sdf"]
    band = sample["boundary_band"]
    if tuple(img.shape) != (1, 128, 128, 128):
        raise RuntimeError(f"Preflight image shape failed: {tuple(img.shape)}")
    if tuple(lab.shape) != tuple(sdf.shape) or tuple(lab.shape) != tuple(band.shape):
        raise RuntimeError("Preflight target shapes differ")
    if int((lab > 0.5).sum()) <= 0 or int((band > 0.5).sum()) <= 0:
        raise RuntimeError("Preflight has empty label/boundary band")
    print(
        f"C2-v2 preflight PASS | case={cid} | image={tuple(img.shape)} | "
        f"FG={int((lab > 0.5).sum())} | sdf=[{float(sdf.min()):+.3f},{float(sdf.max()):+.3f}] | "
        f"band={100.0*float((band>0.5).float().mean()):.2f}% | lambda={lambda_boundary:.3f}"
    )


def main():
    p = argparse.ArgumentParser(description="C2-v2 stable SDF boundary supervision on Final72 A3")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", default="1", help="1 recommended first; screen=1+2; all; or 0..4")
    p.add_argument("--sdf-band-mm", type=float, default=2.0)
    p.add_argument("--lambda-boundary", type=float, default=0.05)
    p.add_argument("--source-cv-dir", default=str(SOURCE_CV))
    p.add_argument("--audit-metadata", default=str(AUDIT))
    p.add_argument("--a3-cv-dir", default=str(A3_CV))
    p.add_argument("--size-profile", default=str(SIZE_PROFILE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not 0.0 < args.lambda_boundary <= 0.25:
        p.error("--lambda-boundary must be in (0, 0.25]")
    if not 0.25 <= args.sdf_band_mm <= 5.0:
        p.error("--sdf-band-mm must be in [0.25, 5.0]")

    selected = spatial_screen.parse_screen_fold(args.fold)
    output_dir = Path(args.output_dir)
    source_manifest = Path(args.source_cv_dir) / "cv_splits.json"
    a3_results_path = Path(args.a3_cv_dir) / "cv_results.csv"
    a3_rows = read_csv(a3_results_path)
    size_profile = read_csv(Path(args.size_profile))

    config = HASSLConfig.from_yaml(args.config)
    if config.compute_mode != "prototype" or config.unet_backbone != "dynunet":
        raise RuntimeError("C2-v2 requires prototype DynUNet")
    _, extra_ids, fold_specs = spatial_screen.build_final72_fold_specs(
        config, source_manifest, Path(args.audit_metadata)
    )
    fold_map = {int(x["fold"]): x for x in fold_specs}

    # Exact A3 architecture remains untouched. Patch only train epoch + data loader locally.
    install_loader_hook(args.sdf_band_mm)
    trainer_module.HASSLTrainer.train_one_epoch_uamt = make_train_epoch(args.lambda_boundary)

    runtime_args = SimpleNamespace(
        config=args.config,
        fold=args.fold,
        folds=5,
        seed=42,
        resize_size=128,
        epochs=100,
        output_dir=str(output_dir),
        split_manifest=str(source_manifest),
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
        "version": "final72_c2v2_sdf_boundary_v1",
        "reference": "Final72 A3",
        "architecture": "EXACT A3 DynUNet one-channel output",
        "sdf_band_mm": float(args.sdf_band_mm),
        "lambda_boundary": float(args.lambda_boundary),
        "boundary_loss": "mean(sigmoid(seg_logit) * normalized_signed_distance) within band",
        "sign": "negative_inside_positive_outside",
        "screening_folds": selected,
        "n_total_human_gold": spatial_screen.EXPECTED_TOTAL,
        "n_train_only_extra": len(extra_ids),
        "external31_access": False,
    }
    plan_path = output_dir / "screening_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError(f"Existing plan differs: {plan_path}; use a fresh output dir")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    preflight(config, fold_map[selected[0]], args.sdf_band_mm, args.lambda_boundary)
    print("=" * 122)
    print("C2-v2 — EXACT A3 DYNUNET + SDF BOUNDARY LOSS")
    print(f"Folds:                 {selected}")
    print("Architecture:          exact A3 (1 input / 1 output segmentation channel)")
    print("Segmentation loss:     DiceCE")
    print(f"Boundary SDF band:     +/-{args.sdf_band_mm:g} mm")
    print(f"Boundary lambda:       {args.lambda_boundary:g}")
    print("A3 augmentation:       +/-4 vox p=.5 + LR flip p=.5")
    print("Evaluation:            raw Student+EMA 50/50 @ .50")
    print("External31:            NOT ACCESSED")
    print("=" * 122)

    new_rows = []
    for f in selected:
        new_rows.extend(cv.run_fold(runtime_args, fold_map[f], output_dir))

    results_path = output_dir / "cv_results.csv"
    old = cv.read_results(results_path)
    selected_set = set(selected)
    merged = [r for r in old if int(r["fold"]) not in selected_set] + new_rows
    merged.sort(key=lambda r: (int(r["fold"]), str(r["case_id"])))
    cv.write_results(results_path, merged)
    compare_to_a3(merged, a3_rows, selected, size_profile, output_dir)

    diagnostics = {
        "last_seg_loss": getattr(trainer_module.HASSLTrainer, "_c2v2_seg_loss", None),
        "note": "Per-run epoch losses are also visible in the normal trainer log.",
    }
    (output_dir / "c2v2_notes.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"\nResults: {results_path}")
    print(f"Plan:    {plan_path}")


if __name__ == "__main__":
    main()
