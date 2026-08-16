#!/usr/bin/env python3
"""Analyze false positives and connected components on frozen CV hold-outs.

Designed for the translation-only supervised CV experiment. This script is read-only:
it does not train, alter checkpoints, change the reported CV threshold, or overwrite
cv_results.csv.

For each held-out case and available prediction source (student/teacher/ensemble), it:
- reproduces deterministic model-space inference
- reports probability statistics inside vs outside GT
- analyzes connected components at the frozen reporting threshold (default 0.50)
- evaluates largest-connected-component (LCC) filtering
- identifies the GT-overlapping component and its size rank (oracle diagnostic only)
- writes a diagnostic-only threshold sweep without using it to re-report CV performance

Outputs:
  case_summary.csv
  components.csv
  threshold_diagnostics.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError(
        "scripts/analyze_cv_false_positives.py requires scipy for connected-component analysis."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hassl.config import HASSLConfig
import scripts.train_supervised_cv as cv


DEFAULT_EXPERIMENT_DIR = Path("experiments/cv5_supervised_resize128_translation12")
DEFAULT_SPLIT_MANIFEST = Path("experiments/cv5_supervised_resize128/cv_splits.json")
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "false_positive_analysis"
DEFAULT_FOCUS_CASE = "80d0955124466d9b82337e7a17a8a2b5de9f4ec9244be0daa6eeb6f5014989d6"


def scalar_case_metrics(pred, target):
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = int(np.logical_and(pred, target).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(target.sum())
    eps = 1e-8
    return {
        "dice": (2.0 * tp + eps) / (pred_sum + gt_sum + eps),
        "precision": (tp + eps) / (pred_sum + eps),
        "recall": (tp + eps) / (gt_sum + eps),
        "pred_fg": pred_sum / float(pred.size),
        "gt_fg": gt_sum / float(target.size),
        "rve": abs(pred_sum - gt_sum) / (gt_sum + eps) * 100.0,
        "gt_vox": gt_sum,
        "pred_vox": pred_sum,
        "tp_vox": tp,
    }


def probability_stats(prob, target):
    target = target.astype(bool, copy=False)
    inside = prob[target]
    outside = prob[~target]

    def stats(values, prefix):
        if values.size == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p95": float("nan"),
                f"{prefix}_p99": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_p50": float(np.percentile(values, 50)),
            f"{prefix}_p95": float(np.percentile(values, 95)),
            f"{prefix}_p99": float(np.percentile(values, 99)),
            f"{prefix}_max": float(np.max(values)),
        }

    result = {}
    result.update(stats(inside, "prob_gt"))
    result.update(stats(outside, "prob_bg"))
    return result


def centroid(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (float("nan"),) * 3
    values = coords.mean(axis=0)
    return tuple(float(x) for x in values)


def centroid_distance(a, b):
    if not all(math.isfinite(x) for x in (*a, *b)):
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def component_analysis(pred, target, prob):
    """Return per-component rows and LCC/oracle summary using 6-connectivity."""
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, n_components = ndimage.label(pred, structure=structure)

    gt_vox = int(target.sum())
    gt_centroid = centroid(target)
    if n_components == 0:
        empty = np.zeros_like(pred, dtype=bool)
        return [], {
            "component_count": 0,
            "components_ge_100": 0,
            "components_ge_1000": 0,
            "largest_component_vox": 0,
            "largest_component_fraction_of_pred": 0.0,
            "largest_component_gt_overlap_vox": 0,
            "largest_component_gt_recall": 0.0,
            "lcc_centroid_distance_gt_vox": float("nan"),
            "lcc_mask": empty,
            "best_overlap_component_rank": float("nan"),
            "best_overlap_component_vox": 0,
            "best_overlap_component_gt_overlap_vox": 0,
            "best_overlap_component_dice": 0.0,
            "best_overlap_component_centroid_distance_gt_vox": float("nan"),
        }

    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
    order = np.argsort(-sizes)
    rank_by_label = {int(label_id): rank for rank, label_id in enumerate(order + 1, start=1)}

    component_rows = []
    for label_id in range(1, n_components + 1):
        mask = labels == label_id
        voxels = int(sizes[label_id - 1])
        overlap = int(np.logical_and(mask, target).sum())
        comp_centroid = centroid(mask)
        metrics = scalar_case_metrics(mask, target)
        component_rows.append({
            "component_label": label_id,
            "size_rank": rank_by_label[label_id],
            "voxels": voxels,
            "fraction_of_prediction": voxels / max(int(pred.sum()), 1),
            "gt_overlap_vox": overlap,
            "gt_recall": overlap / max(gt_vox, 1),
            "component_dice_vs_gt": metrics["dice"],
            "mean_probability": float(prob[mask].mean()) if voxels else float("nan"),
            "max_probability": float(prob[mask].max()) if voxels else float("nan"),
            "centroid_d": comp_centroid[0],
            "centroid_h": comp_centroid[1],
            "centroid_w": comp_centroid[2],
            "centroid_distance_gt_vox": centroid_distance(comp_centroid, gt_centroid),
        })

    component_rows.sort(key=lambda row: row["size_rank"])
    largest = component_rows[0]
    largest_label = int(largest["component_label"])
    lcc_mask = labels == largest_label

    # Oracle diagnostic only: requires GT and must never become deployable post-processing.
    best_overlap = max(
        component_rows,
        key=lambda row: (row["gt_overlap_vox"], row["component_dice_vs_gt"], -row["size_rank"]),
    )

    return component_rows, {
        "component_count": n_components,
        "components_ge_100": int(np.sum(sizes >= 100)),
        "components_ge_1000": int(np.sum(sizes >= 1000)),
        "largest_component_vox": int(largest["voxels"]),
        "largest_component_fraction_of_pred": float(largest["fraction_of_prediction"]),
        "largest_component_gt_overlap_vox": int(largest["gt_overlap_vox"]),
        "largest_component_gt_recall": float(largest["gt_recall"]),
        "lcc_centroid_distance_gt_vox": float(largest["centroid_distance_gt_vox"]),
        "lcc_mask": lcc_mask,
        "best_overlap_component_rank": int(best_overlap["size_rank"]),
        "best_overlap_component_vox": int(best_overlap["voxels"]),
        "best_overlap_component_gt_overlap_vox": int(best_overlap["gt_overlap_vox"]),
        "best_overlap_component_dice": float(best_overlap["component_dice_vs_gt"]),
        "best_overlap_component_centroid_distance_gt_vox": float(
            best_overlap["centroid_distance_gt_vox"]
        ),
    }


def load_models(config, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "net_A" not in state:
        raise RuntimeError(f"{checkpoint_path} is not a HASSL checkpoint with net_A")

    student = cv.build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
    student.load_state_dict(state["net_A"])
    student.eval()

    teacher = None
    if "teacher" in state:
        teacher = cv.build_network(config.unet_backbone, config.num_classes, config.dropout).to(device)
        teacher.load_state_dict(state["teacher"])
        teacher.eval()

    return student, teacher


def requested_sources(value, teacher_available):
    if value == "all":
        sources = ["student"]
        if teacher_available:
            sources.extend(["teacher", "ensemble"])
        return sources
    if value in ("teacher", "ensemble") and not teacher_available:
        raise RuntimeError(f"Requested source={value}, but checkpoint contains no teacher")
    return [value]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    config = HASSLConfig.from_yaml(args.config)
    cv.apply_baseline(config, args.resize_size, epochs=100)

    cases = cv.collect_cases(config)
    by_id = {case["id"]: case for case in cases}

    manifest_path = Path(args.split_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cv.validate_manifest(manifest, cases, args.folds, manifest_path)

    transform = cv.ORIGINAL_GET_TRANSFORMS(
        config,
        keys=["image", "label"],
        is_training=False,
        apply_strong_aug=False,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    inferer = SlidingWindowInferer(
        roi_size=tuple(config.spatial_size),
        sw_batch_size=1,
        overlap=0.25,
    )

    thresholds = sorted(set(float(x) for x in args.thresholds))
    case_rows = []
    component_rows = []
    threshold_rows = []

    print("=" * 100)
    print("Frozen CV false-positive / component diagnostic")
    print(f"Experiment: {args.experiment_dir}")
    print(f"Split manifest: {manifest_path}")
    print(f"Reporting threshold (unchanged): {args.threshold:.2f}")
    print(f"Diagnostic thresholds: {', '.join(f'{x:.2f}' for x in thresholds)}")
    print("NOTE: threshold sweep and GT-overlap component are diagnostic only; they do not replace CV results.")
    print("=" * 100)

    for fold_spec in manifest["folds"]:
        fold_idx = int(fold_spec["fold"])
        checkpoint = (
            Path(args.experiment_dir) / "checkpoints" / f"fold_{fold_idx}" / "best_checkpoint.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")

        fold_cases = [by_id[case_id] for case_id in fold_spec["val_ids"]]
        loader = DataLoader(
            Dataset(fold_cases, transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        student, teacher = load_models(config, checkpoint, device)
        sources = requested_sources(args.source, teacher is not None)

        for batch in loader:
            image = batch["image"].to(device)
            target_t = batch["label"].float().to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(batch["id"])
            spacing = cv.transformed_spacing(image, config)

            with torch.no_grad(), torch.amp.autocast(
                device.type, enabled=(device.type == "cuda")
            ):
                s_prob = torch.sigmoid(cv.main_prediction(inferer(image, student)))
                probs = {"student": s_prob}
                if teacher is not None:
                    t_prob = torch.sigmoid(cv.main_prediction(inferer(image, teacher)))
                    probs["teacher"] = t_prob
                    probs["ensemble"] = 0.5 * (s_prob + t_prob)

            target_np = target_t[0, 0].detach().cpu().numpy() > 0.5
            gt_centroid = centroid(target_np)

            for source in sources:
                prob_t = probs[source]
                prob_np = prob_t[0, 0].detach().float().cpu().numpy()
                pred_np = prob_np > args.threshold

                raw_np_metrics = scalar_case_metrics(pred_np, target_np)
                raw_tensor = torch.from_numpy(pred_np[None, None].astype(np.float32)).to(device)
                raw_full_metrics = cv.case_metrics(raw_tensor, target_t, spacing)

                comps, comp_summary = component_analysis(pred_np, target_np, prob_np)
                lcc_np = comp_summary.pop("lcc_mask")
                lcc_np_metrics = scalar_case_metrics(lcc_np, target_np)
                lcc_tensor = torch.from_numpy(lcc_np[None, None].astype(np.float32)).to(device)
                lcc_full_metrics = cv.case_metrics(lcc_tensor, target_t, spacing)

                row = {
                    "fold": fold_idx,
                    "case_id": case_id,
                    "source": source,
                    "reporting_threshold": args.threshold,
                    "raw_dice": raw_np_metrics["dice"],
                    "raw_precision": raw_np_metrics["precision"],
                    "raw_recall": raw_np_metrics["recall"],
                    "raw_pred_fg": raw_np_metrics["pred_fg"],
                    "raw_rve": raw_np_metrics["rve"],
                    "raw_hd95": raw_full_metrics["hd95"],
                    "raw_pred_vox": raw_np_metrics["pred_vox"],
                    "gt_vox": raw_np_metrics["gt_vox"],
                    "lcc_dice": lcc_np_metrics["dice"],
                    "lcc_precision": lcc_np_metrics["precision"],
                    "lcc_recall": lcc_np_metrics["recall"],
                    "lcc_pred_fg": lcc_np_metrics["pred_fg"],
                    "lcc_rve": lcc_np_metrics["rve"],
                    "lcc_hd95": lcc_full_metrics["hd95"],
                    "lcc_pred_vox": lcc_np_metrics["pred_vox"],
                    "lcc_dice_delta": lcc_np_metrics["dice"] - raw_np_metrics["dice"],
                    "gt_centroid_d": gt_centroid[0],
                    "gt_centroid_h": gt_centroid[1],
                    "gt_centroid_w": gt_centroid[2],
                }
                row.update(probability_stats(prob_np, target_np))
                row.update(comp_summary)
                case_rows.append(row)

                for comp in comps:
                    component_rows.append({
                        "fold": fold_idx,
                        "case_id": case_id,
                        "source": source,
                        "threshold": args.threshold,
                        **comp,
                    })

                for threshold in thresholds:
                    sweep_pred = prob_np > threshold
                    sweep = scalar_case_metrics(sweep_pred, target_np)
                    threshold_rows.append({
                        "fold": fold_idx,
                        "case_id": case_id,
                        "source": source,
                        "threshold": threshold,
                        "dice": sweep["dice"],
                        "precision": sweep["precision"],
                        "recall": sweep["recall"],
                        "pred_fg": sweep["pred_fg"],
                        "rve": sweep["rve"],
                        "pred_vox": sweep["pred_vox"],
                        "diagnostic_only": True,
                    })

            print(f"[fold {fold_idx}] {case_id} complete")

        del student
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "case_summary.csv", case_rows)
    write_csv(output_dir / "components.csv", component_rows)
    write_csv(output_dir / "threshold_diagnostics.csv", threshold_rows)

    metadata = {
        "experiment_dir": str(args.experiment_dir),
        "split_manifest": str(args.split_manifest),
        "reporting_threshold": args.threshold,
        "diagnostic_thresholds": thresholds,
        "source": args.source,
        "connectivity": 6,
        "warning": (
            "Threshold sweep and best-GT-overlap component are diagnostics only. "
            "Do not tune on held-out CV cases and re-report as unbiased CV performance."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    focus = [row for row in case_rows if row["case_id"] == args.focus_case]
    print("\n" + "=" * 100)
    print(f"Outputs: {output_dir}")
    print(f"  {output_dir / 'case_summary.csv'}")
    print(f"  {output_dir / 'components.csv'}")
    print(f"  {output_dir / 'threshold_diagnostics.csv'}")
    print("=" * 100)

    if focus:
        print(f"\nFOCUS CASE: {args.focus_case}")
        for row in focus:
            print(
                f"{row['source']:<8} raw Dice={row['raw_dice']:.4f} "
                f"Prec={row['raw_precision']:.4f} Rec={row['raw_recall']:.4f} "
                f"PredVox={int(row['raw_pred_vox'])} Components={int(row['component_count'])}"
            )
            print(
                f"         LCC Dice={row['lcc_dice']:.4f} "
                f"(delta {row['lcc_dice_delta']:+.4f}) | "
                f"LCC vox={int(row['lcc_pred_vox'])} | "
                f"best-overlap component rank={row['best_overlap_component_rank']} "
                f"Dice={row['best_overlap_component_dice']:.4f}"
            )
            print(
                f"         P(GT) mean/p95/p99/max="
                f"{row['prob_gt_mean']:.3f}/{row['prob_gt_p95']:.3f}/"
                f"{row['prob_gt_p99']:.3f}/{row['prob_gt_max']:.3f} | "
                f"P(BG) mean/p95/p99/max="
                f"{row['prob_bg_mean']:.3f}/{row['prob_bg_p95']:.3f}/"
                f"{row['prob_bg_p99']:.3f}/{row['prob_bg_max']:.3f}"
            )
    else:
        print(f"\nFocus case {args.focus_case} was not found in the manifest.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze probability maps and connected components for frozen CV hold-outs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=128)
    parser.add_argument(
        "--source",
        choices=["all", "student", "teacher", "ensemble"],
        default="all",
        help="Analyze all available sources by default.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Frozen reporting threshold used for component/LCC analysis.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99],
        help="Diagnostic-only thresholds; do not use them to re-report unbiased CV performance.",
    )
    parser.add_argument("--focus-case", default=DEFAULT_FOCUS_CASE)
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("all --thresholds values must be between 0 and 1")

    analyze(args)


if __name__ == "__main__":
    main()
