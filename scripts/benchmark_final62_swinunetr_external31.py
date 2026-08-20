#!/usr/bin/env python3
"""Benchmark the all-62 SwinUNETR-small model on frozen external31.

Evaluation only. No training, threshold tuning, checkpoint selection, or use of external labels
for fitting occurs here. Reuses the established strict native-grid Final62 benchmark and reports
STUDENT, EMA, and 50/50 ENSEMBLE at threshold 0.50.
"""

import inspect
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
    raise SystemExit("Final62 Swin B1 benchmark is locked to feature_size=24")

if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

CHECKPOINT = (
    "experiments/final_supervised_round2_62_translation12_swinunetr_fs24_v1/"
    "checkpoints/final_checkpoint.pth"
)
OUTPUT_DIR = "experiments/external31_final62_swinunetr_fs24_v1"

LOCKED = {
    "--checkpoint": CHECKPOINT,
    "--output-dir": OUTPUT_DIR,
    "--resize-size": "128",
    "--threshold": "0.50",
    "--expected-count": "31",
}
for name, expected in LOCKED.items():
    explicit = _option_value(CLEAN_ARGV, name)
    if explicit is not None:
        if name in {"--resize-size", "--expected-count"}:
            same = int(explicit) == int(expected)
        elif name == "--threshold":
            same = abs(float(explicit) - float(expected)) <= 1e-8
        else:
            same = explicit == expected
        if not same:
            raise SystemExit(f"Controlled Final62 Swin benchmark locks {name}={expected}; got {explicit!r}")
    else:
        CLEAN_ARGV.extend([name, expected])

sys.argv = CLEAN_ARGV

from monai.networks.nets import SwinUNETR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hassl.training.trainer as trainer_module  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
import scripts.benchmark_final62_student_ema_ensemble_external31 as benchmark  # noqa: E402

ORIGINAL_BUILD_NETWORK = trainer_module.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_swin_b1(backbone: str, num_classes: int, dropout: float):
    if backbone != "swinunetr_b1_fs24":
        return ORIGINAL_BUILD_NETWORK(backbone, num_classes, dropout)
    if int(num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {num_classes}")
    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "feature_size": FEATURE_SIZE,
        "use_checkpoint": True,
    }
    signature = inspect.signature(SwinUNETR.__init__)
    if "spatial_dims" in signature.parameters:
        kwargs["spatial_dims"] = 3
    if "img_size" in signature.parameters:
        param = signature.parameters["img_size"]
        if param.default is inspect.Parameter.empty:
            kwargs["img_size"] = (128, 128, 128)
    return SwinUNETR(**kwargs)


def apply_swin(config, resize_size, epochs):
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    config.unet_backbone = "swinunetr_b1_fs24"
    config.swinunetr_feature_size = FEATURE_SIZE
    config.use_gradient_checkpointing = True


# benchmark.load_models constructs through scripts.train_supervised_cv.build_network, while the
# benchmark itself calls cv.apply_baseline before loading. Patch those two exact runtime paths.
trainer_module.build_network = build_swin_b1
cv.build_network = build_swin_b1
cv.apply_baseline = apply_swin


def main():
    print("=" * 120)
    print("FINAL62 SWINUNETR B1 — FROZEN EXTERNAL31")
    print(f"Checkpoint:   {CHECKPOINT}")
    print("Architecture: SwinUNETR fs24 | binary | gradient checkpointing")
    print("Threshold:    0.50 fixed")
    print("Modes:        STUDENT | EMA | ENSEMBLE")
    print("Training:     NONE")
    print("=" * 120)
    benchmark.main()


if __name__ == "__main__":
    main()
