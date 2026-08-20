#!/usr/bin/env python3
"""Train one Final62 SwinUNETR-small model on all audited HUMAN_GOLD labels.

This is the 62-vs-62 follow-up to the controlled B1 architecture CV experiment.
It reuses scripts/train_final_supervised_round2_from_audit.py so label provenance,
translation-only augmentation, fixed-epoch training, and final-checkpoint semantics stay
identical to the established Final62 DynUNet path. The only intended model change is:

    DynUNet -> SwinUNETR(feature_size=24, use_checkpoint=True)

The final epoch budget is NOT hard-coded. By default the underlying final-training runner reads
the five B1 SwinUNETR CV best checkpoints and uses the median saved best epoch, exactly as was
done for Final62 DynUNet.

No external31 labels are read by this script.
"""

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
        "Final62 B1 is locked to --feature-size 24. Use a separate experiment for another capacity."
    )

if GPU is not None:
    if not GPU.isdigit():
        raise SystemExit(f"--gpu must be a non-negative physical GPU index, got {GPU!r}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

SWIN_CV_DIR = "experiments/round2_cv_62_translation12_swinunetr_fs24_v1"
AUDIT = "experiments/round2_supervised_62_translation12/round2_label_audit.json"
SOURCE_MANIFEST = "experiments/cv5_supervised_47_translation12/cv_splits.json"
OUTPUT_DIR = "experiments/final_supervised_round2_62_translation12_swinunetr_fs24_v1"

LOCKED_DEFAULTS = {
    "--round2-cv-dir": SWIN_CV_DIR,
    "--audit-metadata": AUDIT,
    "--source-manifest": SOURCE_MANIFEST,
    "--output-dir": OUTPUT_DIR,
    "--resize-size": "128",
    "--seed": "42",
}
for name, expected in LOCKED_DEFAULTS.items():
    explicit = _option_value(CLEAN_ARGV, name)
    if explicit is not None:
        if name in {"--resize-size", "--seed"}:
            same = int(explicit) == int(expected)
        else:
            same = explicit == expected
        if not same:
            raise SystemExit(f"Controlled Final62 Swin run locks {name}={expected}; got {explicit!r}")
    else:
        CLEAN_ARGV.extend([name, expected])

# --epochs remains intentionally optional. When omitted, the established final runner derives
# the fixed training budget from the median best epoch of the five Swin CV folds.
sys.argv = CLEAN_ARGV

# Heavy imports only after physical GPU visibility is fixed.
from monai.networks.nets import SwinUNETR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hassl.training.trainer as trainer_module  # noqa: E402
import scripts.train_supervised_cv as cv  # noqa: E402
import scripts.train_final_supervised_round2_from_audit as final_round2  # noqa: E402

ORIGINAL_BUILD_NETWORK = trainer_module.build_network
ORIGINAL_APPLY_BASELINE = cv.apply_baseline


def build_swin_b1(backbone: str, num_classes: int, dropout: float):
    if backbone != "swinunetr_b1_fs24":
        return ORIGINAL_BUILD_NETWORK(backbone, num_classes, dropout)
    if int(num_classes) != 1:
        raise RuntimeError(f"Final62 Swin B1 requires num_classes=1, got {num_classes}")

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


def apply_final_swin(config, resize_size, epochs):
    ORIGINAL_APPLY_BASELINE(config, resize_size, epochs)
    if int(config.num_classes) != 1:
        raise RuntimeError(f"Expected binary num_classes=1, got {config.num_classes}")
    if tuple(int(x) for x in config.spatial_size) != (128, 128, 128):
        raise RuntimeError(f"Expected frozen 128^3 input, got {config.spatial_size}")
    config.unet_backbone = "swinunetr_b1_fs24"
    config.swinunetr_feature_size = FEATURE_SIZE
    config.use_gradient_checkpointing = True


# HASSLTrainer resolves trainer_module.build_network at runtime. The final runner calls
# cv.apply_baseline before training, so patch both model construction and recipe application.
trainer_module.build_network = build_swin_b1
cv.build_network = build_swin_b1
cv.apply_baseline = apply_final_swin


def main():
    print("=" * 116)
    print("FINAL62 SWINUNETR B1 — ALL AUDITED HUMAN_GOLD")
    print(f"Architecture:       SwinUNETR feature_size={FEATURE_SIZE}, checkpointing=ON")
    print("Input/output:       128^3 | binary sigmoid out_channels=1")
    print("Recipe:             frozen A0 translation-only supervised recipe")
    print("Epoch selection:    median best epoch from the five completed Swin CV folds")
    print("External31 access:  NONE")
    print(f"CV source:          {SWIN_CV_DIR}")
    print(f"Output:             {OUTPUT_DIR}")
    print("=" * 116)

    final_round2.main()

    metadata_path = Path(OUTPUT_DIR) / "final_training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["architecture_experiment"] = {
            "name": "B1_SWINUNETR_SMALL",
            "architecture": "SwinUNETR",
            "feature_size": FEATURE_SIZE,
            "gradient_checkpointing": True,
            "binary_out_channels": 1,
            "controlled_reference": "Final62 DynUNet A0",
            "architecture_only_change": True,
        }
        metadata["prediction_source"] = (
            "student_ema_modes_available; choose deployment mode only after frozen external31 benchmark"
        )
        metadata["external31_used_for_training_or_epoch_selection"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Swin metadata amended: {metadata_path}")


if __name__ == "__main__":
    main()
