#!/usr/bin/env python3
"""Controlled Final62 SwinUNETR architecture experiment.

This wrapper reuses the provenance-safe Round-2 62-HUMAN_GOLD CV runner and changes ONLY the
segmentation architecture relative to the frozen A0 baseline:

A0: DynUNet
B1: SwinUNETR, feature_size=24, gradient checkpointing enabled

Frozen conditions
-----------------
- 62 audited HUMAN_GOLD labels: original47 + Round1 8 + Round2 7
- exact original-47 held-out fold assignments
- Round1/Round2 added human labels are TRAIN ONLY in every fold
- resize 128^3 / spacing from the supplied config
- binary sigmoid output (out_channels=1 via config.num_classes)
- DiceCE / AdamW lr=1e-4 / weight decay from config
- dropout=0 / lambda_unsup=0
- random initialization / seed42
- paired translation +/-12 voxels, p=0.8
- 100 CV epochs
- EMA teacher / RAW student+EMA 50/50 ensemble evaluation @ threshold 0.50
- no SSL, no LCC, no external31 access

The script intentionally does not modify hassl/training/trainer.py. It patches the model factory
only for this process, keeping the experiment isolated from the stable HASSL path.

Examples
--------
# Recommended first: confirm memory/runtime on one fold
python scripts/train_round2_swinunetr_cv.py --config config_resize128.yaml --gpu 1 --fold 0

# Then run all remaining/completed folds; completed folds are reused
python scripts/train_round2_swinunetr_cv.py --config config_resize128.yaml --gpu 1 --fold all
"""

import csv
import inspect
import json
import os
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


def _option_value(argv, name):
    for i, token in enumerate(argv[1:], start=1):
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _has_option(argv, name):
    return any(x == name or x.startswith(name + "=") for x in argv[1:])


GPU, CLEAN_ARGV = _consume_option(sys.argv, "--gpu")
FEATURE_SIZE_RAW, CLEAN_ARGV = _consume_option(CLEAN_ARGV, "--feature-size")
FEATURE_SIZE = 24 if FEATURE_SIZE_RAW is None else int(FEATURE_SIZE_RAW)
if FEATURE_SIZE != 24:
    raise SystemExit(
        "This controlled B1 experiment is locked to --feature-size 24. "
        "Use a separate experiment branch for another SwinUNETR capacity."
    )

if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

DEFAULT_AUDIT = "experiments/round2_supervised_62_translation12/round2_label_audit.json"
DEFAULT_SOURCE_CV = "experiments/cv5_supervised_47_translation12"
DEFAULT_ROUND1_CV = "experiments/round1_cv_55_translation12"
DEFAULT_OUTPUT = "experiments/round2_cv_62_translation12_swinunetr_fs24_v1"
A0_BASELINE = Path("experiments/round2_cv_62_translation12")

LOCKED = {
    "--epochs": "100",
    "--seed": "42",
    "--resize-size": "128",
    "--eval-source": "ensemble",
    "--eval-threshold": "0.50",
    "--audit-metadata": DEFAULT_AUDIT,
    "--source-cv-dir": DEFAULT_SOURCE_CV,
    "--round1-cv-dir": DEFAULT_ROUND1_CV,
}
for name, expected in LOCKED.items():
    explicit = _option_value(CLEAN_ARGV, name)
    if explicit is None:
        continue
    if name in {"--epochs", "--seed", "--resize-size"}:
        same = int(explicit) == int(expected)
    elif name == "--eval-threshold":
        same = abs(float(explicit) - float(expected)) <= 1e-8
    else:
        same = explicit == expected
    if not same:
        raise SystemExit(
            f"Controlled SwinUNETR B1 locks {name}={expected}; got {explicit!r}."
        )

for name, expected in LOCKED.items():
    if not _has_option(CLEAN_ARGV, name):
        CLEAN_ARGV.extend([name, expected])
if not _has_option(CLEAN_ARGV, "--output-dir"):
    CLEAN_ARGV.extend(["--output-dir", DEFAULT_OUTPUT])

resolved_output = Path(_option_value(CLEAN_ARGV, "--output-dir"))
if resolved_output.resolve() == A0_BASELINE.resolve():
    raise SystemExit(f"Refusing to overwrite frozen A0 baseline: {A0_BASELINE}")

sys.argv = CLEAN_ARGV

# Heavy imports happen only after CUDA visibility is fixed.
import numpy as np  # noqa: E402
from monai.networks.nets import SwinUNETR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hassl.training.trainer as trainer_module  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
import scripts.train_active_learning_round2_cv_from_audit as round2  # noqa: E402

ORIGINAL_BUILD_NETWORK = trainer_module.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_controlled_network(backbone: str, num_classes: int, dropout: float):
    """Use a small binary SwinUNETR for B1; preserve the original factory otherwise."""
    if backbone != "swinunetr_b1_fs24":
        return ORIGINAL_BUILD_NETWORK(backbone, num_classes, dropout)

    if int(num_classes) != 1:
        raise RuntimeError(
            f"B1 is a binary experiment and requires num_classes=1, got {num_classes}"
        )

    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "feature_size": FEATURE_SIZE,
        "use_checkpoint": True,
    }
    signature = inspect.signature(SwinUNETR.__init__)
    if "spatial_dims" in signature.parameters:
        kwargs["spatial_dims"] = 3
    # MONAI releases differ on whether img_size is a required constructor argument.
    if "img_size" in signature.parameters:
        param = signature.parameters["img_size"]
        if param.default is inspect.Parameter.empty:
            kwargs["img_size"] = (128, 128, 128)

    model = SwinUNETR(**kwargs)
    return model


def apply_swinunetr_b1(config, resize_size, epochs):
    """Apply the frozen A0 training recipe, then change only the architecture."""
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    if int(config.num_classes) != 1:
        raise RuntimeError(
            f"Expected binary config num_classes=1 for controlled B1; got {config.num_classes}"
        )
    if tuple(int(x) for x in config.spatial_size) != (128, 128, 128):
        raise RuntimeError(f"Expected frozen 128^3 input, got {config.spatial_size}")
    config.unet_backbone = "swinunetr_b1_fs24"
    config.swinunetr_feature_size = FEATURE_SIZE
    config.use_gradient_checkpointing = True


# Patch both call sites. HASSLTrainer resolves trainer_module.build_network at runtime, while
# train_supervised_cv imported build_network directly for fold checkpoint evaluation.
trainer_module.build_network = build_controlled_network
cv.build_network = build_controlled_network
cv.apply_baseline = apply_swinunetr_b1


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_to_a0(output_dir: Path):
    """Emit a direct B1-vs-A0 paired OOF comparison after all five folds complete."""
    a0_path = A0_BASELINE / "cv_results.csv"
    b1_path = output_dir / "cv_results.csv"
    if not a0_path.exists() or not b1_path.exists():
        return

    a0_rows = read_csv(a0_path)
    b1_rows = read_csv(b1_path)
    a0 = {str(r["case_id"]): round2.enrich_volume_metrics(r) for r in a0_rows}
    b1 = {str(r["case_id"]): round2.enrich_volume_metrics(r) for r in b1_rows}
    if len(a0) != 47 or len(b1) != 47 or set(a0) != set(b1):
        return

    for case_id in a0:
        if int(a0[case_id]["fold"]) != int(b1[case_id]["fold"]):
            raise RuntimeError(f"A0/B1 fold mismatch for {case_id}")

    s0 = round2.summarize(list(a0.values()))
    s1 = round2.summarize(list(b1.values()))
    paired = []
    for case_id in sorted(a0):
        paired.append({
            "case_id": case_id,
            "fold": int(b1[case_id]["fold"]),
            "a0_dice": float(a0[case_id]["dice"]),
            "b1_dice": float(b1[case_id]["dice"]),
            "delta_dice": float(b1[case_id]["dice"]) - float(a0[case_id]["dice"]),
            "a0_precision": float(a0[case_id]["precision"]),
            "b1_precision": float(b1[case_id]["precision"]),
            "a0_recall": float(a0[case_id]["recall"]),
            "b1_recall": float(b1[case_id]["recall"]),
            "a0_signed_rve": float(a0[case_id]["signed_rve"]),
            "b1_signed_rve": float(b1[case_id]["signed_rve"]),
        })

    deltas = np.asarray([r["delta_dice"] for r in paired], dtype=float)
    summary = {
        "version": "final62_swinunetr_b1_vs_dynunet_a0_v1",
        "comparison": "paired exact original47 held-out cases/folds",
        "a0": s0,
        "b1": s1,
        "delta": {
            "mean_dice": s1["mean_dice"] - s0["mean_dice"],
            "median_dice": s1["median_dice"] - s0["median_dice"],
            "mean_precision": s1["mean_precision"] - s0["mean_precision"],
            "mean_recall": s1["mean_recall"] - s0["mean_recall"],
            "median_abs_rve_pct": s1["median_abs_rve_pct"] - s0["median_abs_rve_pct"],
            "dice_lt_0p70": s1["dice_lt_0p70"] - s0["dice_lt_0p70"],
            "dice_lt_0p50": s1["dice_lt_0p50"] - s0["dice_lt_0p50"],
            "dice_ge_0p80": s1["dice_ge_0p80"] - s0["dice_ge_0p80"],
        },
        "case_effects": {
            "improved": int(np.sum(deltas > 1e-6)),
            "worsened": int(np.sum(deltas < -1e-6)),
            "improved_ge_0p05": int(np.sum(deltas >= 0.05)),
            "worsened_le_minus_0p05": int(np.sum(deltas <= -0.05)),
        },
    }
    write_json(output_dir / "swinunetr_b1_vs_a0_summary.json", summary)

    case_path = output_dir / "swinunetr_b1_vs_a0_case_comparison.csv"
    with case_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)

    print("\n" + "=" * 112)
    print("FINAL62 CONTROLLED ARCHITECTURE COMPARISON — A0 DynUNet vs B1 SwinUNETR-small")
    print("=" * 112)
    print(f"Mean Dice:          {s0['mean_dice']:.4f} -> {s1['mean_dice']:.4f} ({summary['delta']['mean_dice']:+.4f})")
    print(f"Median Dice:        {s0['median_dice']:.4f} -> {s1['median_dice']:.4f}")
    print(f"Precision:          {s0['mean_precision']:.4f} -> {s1['mean_precision']:.4f}")
    print(f"Recall:             {s0['mean_recall']:.4f} -> {s1['mean_recall']:.4f}")
    print(f"Median |RVE|:       {s0['median_abs_rve_pct']:.2f}% -> {s1['median_abs_rve_pct']:.2f}%")
    print(f"Dice <0.70:         {s0['dice_lt_0p70']} -> {s1['dice_lt_0p70']}")
    print(f"Dice <0.50:         {s0['dice_lt_0p50']} -> {s1['dice_lt_0p50']}")
    print(f"Dice >=0.80:        {s0['dice_ge_0p80']} -> {s1['dice_ge_0p80']}")
    print(
        "Case effects:       "
        f"improved={summary['case_effects']['improved']} | "
        f"worsened={summary['case_effects']['worsened']} | "
        f"+>=.05={summary['case_effects']['improved_ge_0p05']} | "
        f"<=-.05={summary['case_effects']['worsened_le_minus_0p05']}"
    )
    print(f"Summary:            {output_dir / 'swinunetr_b1_vs_a0_summary.json'}")
    print(f"Case comparison:    {case_path}")
    print("=" * 112)


def main():
    output_dir = Path(_option_value(sys.argv, "--output-dir"))
    metadata = {
        "version": "final62_swinunetr_b1_fs24_v1",
        "controlled_reference": str(A0_BASELINE),
        "architecture_delta_only": True,
        "architecture": {
            "name": "SwinUNETR",
            "in_channels": 1,
            "out_channels": 1,
            "feature_size": FEATURE_SIZE,
            "use_checkpoint": True,
            "spatial_dims": 3,
            "random_initialization": True,
        },
        "frozen": {
            "human_gold_labels": 62,
            "held_out_cases": "exact original47 / exact A0 folds",
            "resize_size": [128, 128, 128],
            "epochs": 100,
            "seed": 42,
            "loss": "dice_ce",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "dropout": 0.0,
            "lambda_unsup": 0.0,
            "translation_voxels": 12.0,
            "translation_probability": 0.8,
            "eval_source": "ensemble",
            "eval_threshold": 0.50,
            "ssl": False,
            "postprocessing": "raw_no_lcc",
            "external31_used": False,
        },
        "warning": "Do not combine appearance augmentation or CPS in this B1 architecture-isolation run.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "swinunetr_b1_profile.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"Existing B1 metadata differs: {metadata_path}")
    else:
        write_json(metadata_path, metadata)

    print("=" * 118)
    print("FINAL62 CONTROLLED SWINUNETR B1")
    print(f"Output:              {output_dir}")
    print("Architecture delta:  DynUNet -> SwinUNETR feature_size=24")
    print("Output convention:   binary sigmoid, out_channels=1")
    print("Frozen:              62 labels | original47 held-out folds | 128^3 | translation +/-12")
    print("Training:            100 epochs | DiceCE | AdamW 1e-4 | dropout0 | lambda_unsup0 | seed42")
    print("Evaluation:          RAW Student+EMA 50/50 ensemble @ 0.50")
    print("SSL/CPS/A1/ext31:    OFF / OFF / OFF / NOT USED")
    print("Physical GPU:        " + (GPU if GPU is not None else "<environment/config>"))
    print("=" * 118)

    round2.main()
    compare_to_a0(output_dir)


if __name__ == "__main__":
    main()
